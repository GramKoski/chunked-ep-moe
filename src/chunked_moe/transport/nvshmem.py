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
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from chunked_moe import util
from chunked_moe.plan import SliceArgs
from chunked_moe.transport import ragged

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


def enable_nvshmem_lowering(sm: int = 90) -> dict:
  """Makes Mosaic GPU lower peer refs to NVSHMEM and returns the matching compile options.

  Mosaic GPU decides between NVSHMEM and XLA's collective metadata by looking
  for --xla_gpu_experimental_enable_nvshmem in XLA_FLAGS, but in jaxlib 0.11.1
  the CPU client rejects that GPU-only flag at startup. So instead we tell
  Mosaic directly:
    compiled = jitted.lower(...).compile(enable_nvshmem_lowering())

  Args:
    sm: Target compute capability (90 = Hopper). nvidia-nvshmem-cu12 3.7 ships
      device bitcode per architecture (up to sm_90), while JAX 0.11.1 looks for a
      single libnvshmem_device.bc, so we point it at the right one.
  """
  # Importing Mosaic GPU sets MOSAIC_GPU_NVSHMEM_{BC,SO}_PATH from the nvidia-nvshmem wheel.
  from jax.experimental.mosaic.gpu import core as mgpu_core  # pylint: disable=import-outside-toplevel

  bc = os.environ.get("MOSAIC_GPU_NVSHMEM_BC_PATH", "")
  if not os.path.exists(bc):
    per_arch = os.path.join(os.path.dirname(bc), f"libnvshmem_device_sm_{sm}.bc")
    if not os.path.exists(per_arch):
      raise RuntimeError(f"no NVSHMEM device bitcode for sm_{sm} next to {bc!r}; install a matching nvidia-nvshmem")
    os.environ["MOSAIC_GPU_NVSHMEM_BC_PATH"] = per_arch
  mgpu_core.is_nvshmem_available = lambda: True
  # XLA:GPU also needs xla_gpu_experimental_enable_nvshmem at run time so it
  # initializes NVSHMEM and allocates these buffers symmetrically. jaxlib 0.11.1
  # does not register that flag (only the CUDA plugin does), so neither
  # XLA_FLAGS nor compile options can carry it here; this needs a jaxlib and
  # plugin built from the same XLA (e.g. NVIDIA's JAX containers). Compiling,
  # which is all this environment can do, does not need it.
  return {}


def _peer(axis_name, index):
  """Mesh device id of position `index` along the EP axis, keeping our own coordinates on every other axis.

  The EP axis may be several mesh axes (e.g. ("context", "expert")); their
  flat index is row-major, like lax.axis_index on a tuple of axes.
  """
  axes = tuple(axis_name) if isinstance(axis_name, (tuple, list)) else (axis_name,)
  coords = {}
  for axis in reversed(axes):
    size = lax.axis_size(axis)
    coords[axis] = lax.rem(index, size)
    index = lax.div(index, size)
  return coords


@functools.partial(jax.named_call, name="nvshmem_slice_exchange")
def nvshmem_exchange(operand: jax.Array, output: jax.Array, args: SliceArgs, axis_name, *, num_sms: int = 132) -> jax.Array:
  if operand.ndim != 2 or output.ndim != 2 or operand.shape[1] != output.shape[1]:
    raise ValueError(f"expected [rows, hidden] buffers, got {operand.shape} and {output.shape}")
  hidden = operand.shape[1]
  if hidden % _WARPGROUP_THREADS:
    # Narrow rows (the router weight and its gradient: one float per row, vs
    # ~14 KB of activations) are not worth a kernel; use the XLA collective.
    return ragged.ragged_exchange(operand, output, args, axis_name)
  num_ranks = lax.axis_size(axis_name)
  num_segments = args.send_sizes.shape[0]
  per_peer = num_segments // num_ranks
  chunk, vec = _row_chunk(hidden, operand.dtype)
  num_blocks = num_sms

  def kernel(in_off_ref, size_ref, out_off_ref, x_ref, _, y_ref):
    block = pl.program_id(0)
    # Global semaphores persist across launches of this kernel. Each has exactly
    # one waiter that consumes exactly what one launch produces, so early
    # arrivals from a peer that is already in its next launch are just counted.
    sems = pl.get_global(plgpu.SemaphoreType.REGULAR((3,)))
    ready = sems.at[0]  # peers that entered this launch
    go = sems.at[1]  # block 0 -> other local blocks
    done = sems.at[2]  # (sender block) finished writing to me

    # Ready barrier: XLA orders this kernel after every local use of y's memory
    # (including the zero fill of a fresh buffer), but remote writers are not in
    # that order. Nobody writes into a peer until that peer has entered the kernel.
    @pl.when(block == 0)
    def _barrier():
      @pl.loop(0, num_ranks)
      def _announce(peer):
        pl.semaphore_signal(ready, device_id=_peer(axis_name, peer))

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
      peer_y = plgpu.remote_ref(y_ref, _peer(axis_name, dst))

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
      pl.semaphore_signal(done, device_id=_peer(axis_name, dst))

    # The kernel finishes only when block 0 has seen every block of every sender.
    @pl.when(block == 0)
    def _wait_for_data():
      pl.semaphore_wait(done, value=num_blocks * num_ranks, decrement=True)

  output = util.varying(output, axis_name)
  out_type = jax.ShapeDtypeStruct(output.shape, output.dtype, manual_axis_type=jax.typeof(output).manual_axis_type)
  gmem = pl.BlockSpec(memory_space=plgpu.GMEM)
  return pl.pallas_call(
      kernel,
      out_shape=out_type,
      grid=(num_blocks,),
      in_specs=[gmem] * 5,
      out_specs=gmem,
      input_output_aliases={4: 0},  # `output` is updated in place by remote writes
      compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
  )(args.input_offsets, args.send_sizes, args.output_offsets, operand, output)
