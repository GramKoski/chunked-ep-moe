"""Slice exchange with one-sided NVSHMEM puts from a Mosaic GPU kernel.

Same contract as `jax.lax.ragged_all_to_all` (see transport/ragged.py), but the
data path is Cohere's design: each sender writes its rows straight into the
receiver's buffer through NVSHMEM peer pointers, then bumps a semaphore on that
receiver. A receiver does not take part in the transfer; it only waits until
every block of every sender has signalled. There is no NCCL call and no staging
copy on either side.

Synchronization, per launch:
  1. ready barrier: block 0 of every rank signals every peer, then waits for
     all peers. Remote writes ignore the receiver's stream order, so this is
     what keeps a fast sender from writing into memory the receiver is still
     using (or has not zero-filled yet).
  2. puts: every block copies its share of rows into peer buffers.
  3. done: every block signals every receiver; block 0 of each receiver waits
     for num_blocks * num_ranks signals before the kernel can finish.

Rows move through registers, one row per block iteration, so segment lengths
can be anything (TMA would need static box shapes and masking for the tails).
Blocks stride over the rows of each segment, so many SMs write in parallel.

Requirements on a real system:
  * XLA_FLAGS must contain --xla_gpu_experimental_enable_nvshmem, and the
    NVSHMEM host/device libraries must be importable (nvidia-nvshmem-cu12).
  * One process per GPU (the NVSHMEM PE id must equal the mesh device id).
  * `output` is updated in place (inout), so XLA:GPU must place it in NVSHMEM
    symmetric memory. That is how JAX's own Mosaic GPU collectives are set up,
    but it can only be confirmed on a multi-GPU run.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from chunked_moe.plan import SliceArgs

# Bytes each thread moves per vector access; 16 B is one 128-bit store.
_VECTOR_BYTES = 16
_WARPGROUP_THREADS = 128


def _row_chunk(hidden: int, dtype) -> tuple[int, int]:
  """Elements per register chunk of a row and the per-thread vector width."""
  vec = _VECTOR_BYTES // jnp.dtype(dtype).itemsize
  chunk = _WARPGROUP_THREADS * vec
  while hidden % chunk:
    vec //= 2
    if vec == 0:
      raise ValueError(f"hidden={hidden} must be a multiple of {_WARPGROUP_THREADS}")
    chunk = _WARPGROUP_THREADS * vec
  return chunk, vec


@functools.partial(jax.named_call, name="nvshmem_slice_exchange")
def nvshmem_exchange(operand: jax.Array, output: jax.Array, args: SliceArgs, axis_name, *, num_sms: int = 132) -> jax.Array:
  if operand.ndim != 2 or output.ndim != 2 or operand.shape[1] != output.shape[1]:
    raise ValueError(f"expected [rows, hidden] buffers, got {operand.shape} and {output.shape}")
  num_ranks = lax.axis_size(axis_name)
  num_segments = args.send_sizes.shape[0]
  per_peer = num_segments // num_ranks
  hidden = operand.shape[1]
  chunk, vec = _row_chunk(hidden, operand.dtype)
  num_blocks = num_sms

  def kernel(in_off_ref, size_ref, out_off_ref, x_ref, _, y_ref):
    block = pl.program_id(0)
    # Global semaphores persist across launches of this kernel. Each has exactly
    # one waiter that consumes exactly what one launch produces, so early
    # arrivals from a peer that is already in its next launch are just counted.
    ready = pl.get_global(plgpu.SemaphoreType.REGULAR)  # peers that entered this launch
    go = pl.get_global(plgpu.SemaphoreType.REGULAR)  # block 0 -> other local blocks
    done = pl.get_global(plgpu.SemaphoreType.REGULAR)  # (sender block) finished writing to me

    # Ready barrier: XLA orders this kernel after every local use of y's memory
    # (including the zero fill of a fresh buffer), but remote writers are not in
    # that order. Nobody writes into a peer until that peer has entered the kernel.
    @pl.when(block == 0)
    def _barrier():
      @pl.loop(0, num_ranks)
      def _announce(peer):
        pl.semaphore_signal(ready, device_id=peer)

      pl.semaphore_wait(ready, value=num_ranks, decrement=True)

      @pl.loop(1, num_blocks)
      def _release(_):
        pl.semaphore_signal(go)

    @pl.when(block != 0)
    def _wait_for_barrier():
      pl.semaphore_wait(go, value=1, decrement=True)

    @pl.loop(0, num_segments)
    def _segment(seg):
      dst = seg // per_peer
      src_row0 = in_off_ref[seg]
      dst_row0 = out_off_ref[seg]
      peer_y = plgpu.remote_ref(y_ref, dst)

      @pl.loop(block, size_ref[seg], step=num_blocks)
      def _row(r):
        for c in range(hidden // chunk):  # static unroll over the row's chunks
          cols = pl.ds(c * chunk, chunk)
          vals = plgpu.layout_cast(x_ref[src_row0 + r, cols], plgpu.Layout.WG_STRIDED((chunk,), vec_size=vec))
          peer_y[dst_row0 + r, cols] = vals

    # The signal is a release operation, so the puts above are visible to the
    # receiver before it sees the count go up.
    @pl.loop(0, num_ranks)
    def _signal(dst):
      pl.semaphore_signal(done, device_id=dst)

    # The kernel finishes only when block 0 has seen every block of every sender.
    @pl.when(block == 0)
    def _wait_for_data():
      pl.semaphore_wait(done, value=num_blocks * num_ranks, decrement=True)

  gmem = pl.BlockSpec(memory_space=plgpu.GMEM)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(output.shape, output.dtype),
      grid=(num_blocks,),
      in_specs=[gmem] * 5,
      out_specs=gmem,
      input_output_aliases={4: 0},  # `output` is updated in place by remote writes
      compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
  )(args.input_offsets, args.send_sizes, args.output_offsets, operand, output)
