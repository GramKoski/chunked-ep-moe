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
