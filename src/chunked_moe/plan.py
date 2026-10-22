"""Slice planning for chunked expert-parallel dispatch.

Everything in this module runs inside `shard_map` over the expert-parallel
axis and only does integer bookkeeping. Both transports (ragged_all_to_all and
NVSHMEM) consume the same plan, so the addressing is tested once.

Layout. Let P be the EP size, L = E / P the local experts per rank, and
C[s, r, l] the number of token copies that rank s routes to local expert l of
rank r. Every rank all-gathers C, so every rank knows the full traffic matrix.

Sender view: a rank sorts its T*K token copies by global expert id. Copies for
(receiver r, local expert l) are then contiguous in this "send order", starting
at `send_start[s, r*L + l]`.

Receiver view: rank r lays out everything it will receive as one packed,
expert-major stream: all copies for its local expert 0 (from source 0, then
source 1, ...), then local expert 1, and so on. Segment (s, l) starts at
`recv_start[r, l, s]`. The stream has R[r] = sum_{s,l} C[s, r, l] rows and no
padding.

Slices. The stream is cut into fixed slices of S rows. Slice j covers stream
rows [j*S, (j+1)*S). Because the stream is expert-major, each slice arrives
already grouped by expert, so the grouped GEMM can run on it without a local
re-sort. All ranks loop for the same number of slices,
max_r ceil(R[r] / S); a rank whose stream is shorter just sends and receives
nothing in the tail iterations, and the last partial slice carries < S rows of
padding.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class DispatchPlan(NamedTuple):
  """Per-layer routing bookkeeping, identical in forward and backward."""

  # [T*K] flat assignment index (t*K + k) of the row at each send-order position.
  order: jax.Array
  # [P, P, L] token copies routed from source s to local expert l of rank r.
  counts: jax.Array
  # [P, P*L] exclusive prefix sum of counts[s].reshape(-1): where source s's
  # rows for global expert r*L + l start in its send order.
  send_start: jax.Array
  # [P, L, P] where segment (source s, local expert l) starts in rank r's stream.
  recv_start: jax.Array
  # [P, L] where local expert l's rows start in rank r's stream.
  expert_start: jax.Array
  # [P, L] rows for local expert l in rank r's stream.
  expert_rows: jax.Array
  # [] number of slice iterations, the same on every rank.
  num_slices: jax.Array


def _exclusive_cumsum(x, axis):
  return jnp.cumsum(x, axis=axis) - x


def make_plan(expert_ids: jax.Array, num_experts: int, slice_size: int, axis_name) -> DispatchPlan:
  """Builds the dispatch plan from this rank's top-k expert ids.

  Args:
    expert_ids: [T, K] int32 global expert ids for this rank's tokens.
    num_experts: Global expert count E. Must be divisible by the EP size.
    slice_size: Rows per slice S.
    axis_name: Expert-parallel mesh axis name.
  """
  num_ranks = jax.lax.axis_size(axis_name)
  if num_experts % num_ranks:
    raise ValueError(f"num_experts={num_experts} is not divisible by the EP size {num_ranks}")
  local_experts = num_experts // num_ranks

  flat = expert_ids.reshape(-1).astype(jnp.int32)
  order = jnp.argsort(flat, stable=True).astype(jnp.int32)
  my_counts = jnp.zeros((num_experts,), jnp.int32).at[flat].add(1)
  counts = jax.lax.all_gather(my_counts.reshape(num_ranks, local_experts), axis_name)  # [s, r, l]

  send_start = _exclusive_cumsum(counts.reshape(num_ranks, num_ranks * local_experts), axis=1)

  per_receiver = jnp.transpose(counts, (1, 2, 0))  # [r, l, s]
  expert_rows = per_receiver.sum(axis=2)  # [r, l]
  expert_start = _exclusive_cumsum(expert_rows, axis=1)  # [r, l]
  recv_start = expert_start[:, :, None] + _exclusive_cumsum(per_receiver, axis=2)  # [r, l, s]

  stream_rows = expert_rows.sum(axis=1)  # [r]
  num_slices = (jnp.max(stream_rows) + slice_size - 1) // slice_size
  return DispatchPlan(order, counts, send_start, recv_start, expert_start, expert_rows, num_slices.astype(jnp.int32))


def _overlap(lo, hi, start, size):
  """Intersection of [lo, hi) with [start, start + size) as (first row, length)."""
  first = jnp.clip(lo, start, start + size)
  last = jnp.clip(hi, start, start + size)
  return first, last - first


class SliceArgs(NamedTuple):
  """Arguments for one ragged exchange, laid out as [peer * L + l]."""

  input_offsets: jax.Array
  send_sizes: jax.Array
  output_offsets: jax.Array
  recv_sizes: jax.Array


def dispatch_args(plan: DispatchPlan, j, slice_size: int, axis_name) -> SliceArgs:
  """Arguments that move slice j from senders' send order into receivers' slice buffers."""
  me = jax.lax.axis_index(axis_name)
  num_ranks, _, local_experts = plan.counts.shape
  lo = j * slice_size
  hi = lo + slice_size

  # As a sender: one segment per (receiver r, local expert l) of that receiver.
  seg_start = plan.recv_start[:, :, me]  # [r, l] where my segment starts in r's stream
  seg_size = plan.counts[me]  # [r, l]
  first, size = _overlap(lo, hi, seg_start, seg_size)
  my_send_start = plan.send_start[me].reshape(num_ranks, local_experts)
  input_offsets = my_send_start + (first - seg_start)
  output_offsets = first - lo

  # As a receiver: one segment per (source s, my local expert l).
  in_start = jnp.transpose(plan.recv_start[me], (1, 0))  # [s, l]
  in_size = plan.counts[:, me, :]  # [s, l]
  _, recv_sizes = _overlap(lo, hi, in_start, in_size)

  flat = lambda a: a.reshape(-1).astype(jnp.int32)
  return SliceArgs(flat(input_offsets), flat(size), flat(output_offsets), flat(recv_sizes))


def combine_args(plan: DispatchPlan, j, slice_size: int, axis_name) -> SliceArgs:
  """Arguments that return slice j's expert outputs to the senders' send order.

  This is the exact reverse of `dispatch_args`: every receiver sends each
  (source, local expert) segment of its slice back to where it came from.
  """
  me = jax.lax.axis_index(axis_name)
  num_ranks, _, local_experts = plan.counts.shape
  lo = j * slice_size
  hi = lo + slice_size

  # As the returning rank: segments (source s, my local expert l) of my slice.
  in_start = jnp.transpose(plan.recv_start[me], (1, 0))  # [s, l]
  in_size = plan.counts[:, me, :]  # [s, l]
  first, size = _overlap(lo, hi, in_start, in_size)
  input_offsets = first - lo
  source_send_start = plan.send_start.reshape(num_ranks, num_ranks, local_experts)[:, me, :]  # [s, l]
  output_offsets = source_send_start + (first - in_start)

  # As the original sender: my dispatch send sizes for slice j come back.
  seg_start = plan.recv_start[:, :, me]  # [r, l]
  _, recv_sizes = _overlap(lo, hi, seg_start, plan.counts[me])

  flat = lambda a: a.reshape(-1).astype(jnp.int32)
  return SliceArgs(flat(input_offsets), flat(size), flat(output_offsets), flat(recv_sizes))


def slice_group_sizes(plan: DispatchPlan, j, slice_size: int, axis_name) -> jax.Array:
  """[L] rows of each local expert inside this rank's slice j (they are contiguous, in order)."""
  me = jax.lax.axis_index(axis_name)
  lo = j * slice_size
  _, rows = _overlap(lo, lo + slice_size, plan.expert_start[me], plan.expert_rows[me])
  return rows.astype(jnp.int32)
