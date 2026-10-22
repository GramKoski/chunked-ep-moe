"""Slice exchange built on `jax.lax.ragged_all_to_all`, plus a CPU emulation.

Both functions take the same arguments as `jax.lax.ragged_all_to_all`, with the
offset arrays laid out as [peer * segments_per_peer + segment].
"""

import jax
import jax.numpy as jnp

from chunked_moe import util
from chunked_moe.plan import SliceArgs


def ragged_exchange(operand: jax.Array, output: jax.Array, args: SliceArgs, axis_name) -> jax.Array:
  return jax.lax.ragged_all_to_all(
      operand, output, args.input_offsets, args.send_sizes, args.output_offsets, args.recv_sizes, axis_name=axis_name
  )


def emulated_exchange(operand: jax.Array, output: jax.Array, args: SliceArgs, axis_name) -> jax.Array:
  """Same semantics as `ragged_exchange`, using a padded dense all_to_all.

  XLA:CPU has no ragged-all-to-all thunk, so tests run the algorithm through
  this function. Every segment is padded to `output.shape[0]` rows, which is
  only affordable at test sizes.
  """
  num_ranks = jax.lax.axis_size(axis_name)
  num_segments = args.send_sizes.shape[0]
  per_peer = num_segments // num_ranks
  cap = min(output.shape[0], operand.shape[0])  # no segment is longer than either buffer
  rows = jnp.arange(cap, dtype=jnp.int32)

  # Pack every outgoing segment into a fixed [cap, ...] window.
  src_idx = args.input_offsets[:, None] + rows[None, :]
  windows = jnp.take(operand, jnp.clip(src_idx, 0, operand.shape[0] - 1), axis=0)
  keep = (rows[None, :] < args.send_sizes[:, None]).reshape((num_segments, cap) + (1,) * (operand.ndim - 1))
  windows = jnp.where(keep, windows, jnp.zeros_like(windows))

  def to_peers(x):
    x = x.reshape((num_ranks, per_peer) + x.shape[1:])
    return jax.lax.all_to_all(x, axis_name, 0, 0, tiled=False).reshape((num_segments,) + x.shape[2:])

  received = to_peers(windows)
  dest_offsets = to_peers(args.output_offsets)

  # Write each received segment at the offset its sender chose.
  def write(i, result):
    target = dest_offsets[i] + rows
    in_range = rows < args.recv_sizes[i]
    target = jnp.where(in_range, target, output.shape[0])  # out-of-range rows are dropped
    return result.at[target].set(received[i], mode="drop")

  return jax.lax.fori_loop(0, num_segments, write, util.varying(output, axis_name))
