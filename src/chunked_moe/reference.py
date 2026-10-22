"""Dense single-device MoE used as the numerical oracle in tests."""

import jax
import jax.numpy as jnp

from chunked_moe import expert


def reference_moe(x, expert_ids, weights, w_up, w_down):
  """x: [N, D], expert_ids/weights: [N, K], w_up: [E, D, 2F] striped, w_down: [E, F, D]."""
  xf = x.astype(jnp.float32)
  h = jnp.einsum("nd,nkdf->nkf", xf, w_up[expert_ids].astype(jnp.float32))
  act = expert.striped_swiglu(h)
  y = jnp.einsum("nkf,nkfd->nkd", act, w_down[expert_ids].astype(jnp.float32))
  return jnp.einsum("nkd,nk->nd", y, weights.astype(jnp.float32)).astype(x.dtype)
