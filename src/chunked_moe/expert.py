"""Per-slice expert MLP on an expert-major slice, with striped SwiGLU.

Weights use the striped layout: `w_up` is [L, D, 2F] with gate and up columns
interleaved (column 2i is gate_i, column 2i+1 is up_i). A fused GEMM epilogue
then finds each gate/up pair next to each other in its accumulator tile and can
apply SwiGLU before anything reaches HBM. `stripe_weights` converts the usual
separate (gate, up) weights into this layout.

The forward and backward here are written out by hand rather than with
`jax.vjp` because the chunked layer recomputes the forward inside its backward
loop and wants to choose what to keep; see `moe.py`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


def stripe_weights(w_gate: jax.Array, w_up: jax.Array) -> jax.Array:
  """[L, D, F] gate and [L, D, F] up -> [L, D, 2F] striped."""
  return jnp.stack([w_gate, w_up], axis=-1).reshape(*w_gate.shape[:-1], 2 * w_gate.shape[-1])


def unstripe_weights(w_striped: jax.Array) -> tuple[jax.Array, jax.Array]:
  return w_striped[..., 0::2], w_striped[..., 1::2]


def _gmm(lhs, rhs, group_sizes, out_dtype):
  """Grouped GEMM: rows of lhs in group l multiply rhs[l]."""
  return jax.lax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type=jnp.float32).astype(out_dtype)


def _tgmm(lhs, rhs, group_sizes, num_groups):
  """Per-group lhs^T @ rhs over the ragged row dimension -> [L, lhs_cols, rhs_cols] in float32."""
  dims = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(([0], [0]), ([], [])),
      lhs_ragged_dimensions=[0],
      rhs_group_dimensions=[],
  )
  out = jax.lax.ragged_dot_general(lhs, rhs, group_sizes, dims, preferred_element_type=jnp.float32)
  assert out.shape[0] == num_groups
  return out


def striped_swiglu(h: jax.Array) -> jax.Array:
  gate = h[..., 0::2].astype(jnp.float32)
  up = h[..., 1::2].astype(jnp.float32)
  return (jax.nn.silu(gate) * up).astype(h.dtype)


def striped_swiglu_bwd(h: jax.Array, d_act: jax.Array) -> jax.Array:
  """Gradient of `striped_swiglu` w.r.t. its striped input h."""
  gate = h[..., 0::2].astype(jnp.float32)
  up = h[..., 1::2].astype(jnp.float32)
  d_act = d_act.astype(jnp.float32)
  sig = jax.nn.sigmoid(gate)
  silu = gate * sig
  d_gate = d_act * up * (sig + silu * (1.0 - sig))
  d_up = d_act * silu
  return jnp.stack([d_gate, d_up], axis=-1).reshape(h.shape).astype(h.dtype)


class ForwardResiduals(NamedTuple):
  h: jax.Array  # [S, 2F] striped pre-activation
  act: jax.Array  # [S, F]


def expert_forward(x, w_up, w_down, group_sizes):
  """x: [S, D] expert-major slice -> y: [S, D]."""
  h = _gmm(x, w_up, group_sizes, x.dtype)
  act = striped_swiglu(h)
  y = _gmm(act, w_down, group_sizes, x.dtype)
  return y, ForwardResiduals(h, act)


class ExpertGrads(NamedTuple):
  dx: jax.Array  # [S, D]
  dw_up: jax.Array  # [L, D, 2F] float32
  dw_down: jax.Array  # [L, F, D] float32


def expert_backward(x, w_up, w_down, group_sizes, res: ForwardResiduals, dy) -> ExpertGrads:
  num_groups = w_up.shape[0]
  d_act = _gmm(dy, jnp.swapaxes(w_down, 1, 2), group_sizes, x.dtype)
  dw_down = _tgmm(res.act, dy, group_sizes, num_groups)
  dh = striped_swiglu_bwd(res.h, d_act)
  dx = _gmm(dh, jnp.swapaxes(w_up, 1, 2), group_sizes, x.dtype)
  dw_up = _tgmm(x, dh, group_sizes, num_groups)
  return ExpertGrads(dx, dw_up, dw_down)
