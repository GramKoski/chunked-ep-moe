"""Fused Mosaic GPU kernels vs. the reference expert math, run in the GPU interpreter on CPU."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic_gpu.interpret import params as interpret_params

from chunked_moe import expert
from chunked_moe.kernels import fused_expert

import interpret_shims

interpret_shims.install()

S, D, F, STRIPE = 128, 128, 256, 128
GROUP_SIZES = {
    "partial_with_empty_group": [40, 0, 70],  # 18 padding rows at the tail
    "full": [64, 1, 63],
}


@pytest.fixture(autouse=True)
def _interpret():
  with interpret_params.force_gpu_interpret_mode(interpret_params.InterpretGPUParams()):
    yield


def _data(sizes):
  keys = jax.random.split(jax.random.key(0), 5)
  dt = jnp.bfloat16
  x = jax.random.normal(keys[0], (S, D), dt)
  w_up = (jax.random.normal(keys[1], (len(sizes), D, 2 * F)) / np.sqrt(D)).astype(dt)
  w_down = (jax.random.normal(keys[2], (len(sizes), F, D)) / np.sqrt(F)).astype(dt)
  dy = jax.random.normal(keys[3], (S, D), dt)
  return x, w_up, w_down, dy, jnp.array(sizes, jnp.int32), sum(sizes)


def _close(got, want, valid_rows=None, rtol=3e-2):
  got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
  if valid_rows is not None:
    got, want = got[:valid_rows], want[:valid_rows]
  scale = np.abs(want).max()
  np.testing.assert_allclose(got, want, atol=rtol * scale, rtol=rtol)


@pytest.mark.parametrize("case", list(GROUP_SIZES))
def test_forward(case):
  x, w_up, w_down, _, gs, n = _data(GROUP_SIZES[case])
  y, res = fused_expert.expert_forward(x, w_up, w_down, gs, stripe=STRIPE, num_sms=4)
  y_ref, res_ref = expert.expert_forward(x, w_up, w_down, gs, stripe=STRIPE)
  _close(res.h, res_ref.h, n)
  _close(res.act, res_ref.act, n)
  _close(y, y_ref, n)


@pytest.mark.parametrize("case", list(GROUP_SIZES))
def test_backward(case):
  x, w_up, w_down, dy, gs, n = _data(GROUP_SIZES[case])
  _, res = expert.expert_forward(x, w_up, w_down, gs, stripe=STRIPE)
  got = fused_expert.expert_backward(x, w_up, w_down, gs, res, dy, stripe=STRIPE, num_sms=4)
  want = expert.expert_backward(x, w_up, w_down, gs, res, dy, stripe=STRIPE)
  _close(got.dx, want.dx, n)
  _close(got.dw_up, want.dw_up)
  _close(got.dw_down, want.dw_down)


def test_chunked_layer_with_fused_kernels():
  """End to end: shard_map + slice loop + remat/backward, with every expert GEMM a Mosaic GPU kernel."""
  from jax.sharding import NamedSharding, PartitionSpec as P

  from chunked_moe import ChunkedMoEConfig, chunked_moe, reference_moe

  ep, tokens, experts, top_k = 2, 64, 4, 2
  mesh = jax.make_mesh((ep,), ("ep",))
  cfg = ChunkedMoEConfig(
      num_experts=experts, top_k=top_k, slice_size=64, transport="emulated",
      expert_compute="fused", stripe_width=STRIPE, num_sms=4,
  )
  keys = jax.random.split(jax.random.key(1), 6)
  n = ep * tokens
  x = jax.random.normal(keys[0], (n, D), jnp.float32)
  logits = jax.random.normal(keys[1], (n, experts)).at[:, 0].add(2.0)  # skewed: slices fill unevenly
  weights, ids = jax.lax.top_k(jax.nn.softmax(logits), top_k)
  w_up = jax.random.normal(keys[2], (experts, D, 2 * F)) / np.sqrt(D)
  w_down = jax.random.normal(keys[3], (experts, F, D)) / np.sqrt(F)
  g = jax.random.normal(keys[4], (n, D))
  cast = lambda a: a.astype(jnp.bfloat16)
  x, w_up, w_down = cast(x), cast(w_up), cast(w_down)

  spec = P("ep")
  put = lambda a: jax.device_put(a, NamedSharding(mesh, spec))
  layer = jax.shard_map(
      functools.partial(chunked_moe, cfg=cfg, axis_name="ep"), mesh=mesh, in_specs=(spec,) * 5, out_specs=spec
  )
  ref = functools.partial(reference_moe, stripe=STRIPE)

  def loss(f, ids_):
    return lambda x, w, u, d: jnp.sum(f(x, ids_, w, u, d).astype(jnp.float32) * g)

  out = jax.jit(layer)(*map(put, (x, ids.astype(jnp.int32), weights, w_up, w_down)))
  _close(out, ref(x, ids, weights, w_up, w_down))
  got = jax.jit(jax.grad(loss(layer, put(ids.astype(jnp.int32))), argnums=(0, 1, 2, 3)))(*map(put, (x, weights, w_up, w_down)))
  want = jax.grad(loss(ref, ids), argnums=(0, 1, 2, 3))(x, weights, w_up, w_down)
  for a, b in zip(got, want):
    _close(a, b)
