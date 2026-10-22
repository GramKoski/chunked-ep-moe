"""Test-only patches that let the Mosaic GPU interpreter (JAX 0.11) run our kernels on CPU.

Neither patch changes what a kernel computes; they fill two gaps in the
interpreter for operations the real compiler handles:

  * layout_cast only picks a register layout for the real compiler. The
    interpreter has no rule for it; on CPU it is the identity.
  * wgmma with a transposed operand (plgpu.transpose_ref) is how Hopper reads a
    B tile in the other major order. The interpreter's wgmma callback assumes
    every ref transform is an indexer, so we drop a trailing transpose from the
    transform list and transpose the fetched tile instead.
"""

import jax
import numpy as np
from jax._src.interpreters import mlir
from jax._src.pallas.mosaic.interpret import utils as interpret_utils
from jax._src.pallas.mosaic_gpu import core as mgpu_core
from jax._src.pallas.mosaic_gpu.interpret import gpu_callbacks
from jax._src.pallas.mosaic_gpu.interpret import shared_memory as memory
from jax._src.state.types import TransposeTransform

_installed = False


def _split_transpose(transforms):
  transforms = tuple(transforms)
  if transforms and isinstance(transforms[-1], TransposeTransform):
    return transforms[:-1], transforms[-1].permutation
  if any(isinstance(t, TransposeTransform) for t in transforms):
    raise NotImplementedError("only a trailing transpose_ref is supported by this shim")
  return transforms, None


def _wgmma(
    *,
    token,
    mesh_location,
    thread,
    acc_allocation_key_as_array,
    acc_transforms,
    acc_dtype,
    a_allocation_key_as_array,
    a_transforms,
    b_allocation_key_as_array,
    b_transforms,
    source_info=None,
):
  key = memory.HostAllocationKey.from_array
  shared = gpu_callbacks._get_shared_memory()  # pylint: disable=protected-access
  log = memory.GPULoggingInfo(mesh_location, thread, source_info)

  def fetch(alloc, transforms):
    transforms, perm = _split_transpose(gpu_callbacks._remove_noop_transforms(transforms))  # pylint: disable=protected-access
    transforms = jax.tree.map(int, transforms)
    value, _, _ = shared.get_buffer_content(key(alloc), interpret_utils.to_range(transforms), thread, logging_info=log)
    return value if perm is None else np.transpose(value, perm)

  a = fetch(a_allocation_key_as_array, a_transforms)
  b = fetch(b_allocation_key_as_array, b_transforms)
  acc_transforms = jax.tree.map(int, gpu_callbacks._remove_noop_transforms(acc_transforms))  # pylint: disable=protected-access
  acc_range = interpret_utils.to_range(acc_transforms)
  acc, _, _ = shared.get_buffer_content(key(acc_allocation_key_as_array), acc_range, thread, logging_info=log)
  shared.store_buffer_content(
      key(acc_allocation_key_as_array), acc_range, acc + np.matmul(a, b, dtype=acc_dtype), thread, logging_info=log
  )
  return token


def install():
  global _installed
  if _installed:
    return
  mlir.register_lowering(mgpu_core.layout_cast_p, lambda ctx, x, **_: [x], platform="cpu")
  gpu_callbacks.wgmma = _wgmma
  _installed = True
