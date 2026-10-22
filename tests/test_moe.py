"""Chunked EP MoE vs. the dense reference, on simulated CPU devices."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from chunked_moe import ChunkedMoEConfig, chunked_moe, reference_moe

jax.config.update("jax_enable_x64", False)


def _routing(key, num_tokens, num_experts, top_k, skew):
  """Top-k expert ids; skew="all_one" sends every token's first choice to expert 0."""
  logits = jax.random.normal(key, (num_tokens, num_experts))
  if skew == "all_one":
    logits = logits.at[:, 0].add(100.0)
  elif skew == "rank0":
    logits = logits.at[:, : num_experts // 4].add(100.0)
  weights, ids = jax.lax.top_k(jax.nn.softmax(logits), top_k)
  return ids.astype(jnp.int32), weights


def _setup(ep, tokens_per_rank, num_experts, top_k, hidden, ffn, skew, seed=0):
  keys = jax.random.split(jax.random.key(seed), 5)
  n = ep * tokens_per_rank
  x = jax.random.normal(keys[0], (n, hidden), jnp.float32)
  ids, weights = _routing(keys[1], n, num_experts, top_k, skew)
  w_up = jax.random.normal(keys[2], (num_experts, hidden, 2 * ffn), jnp.float32) / np.sqrt(hidden)
  w_down = jax.random.normal(keys[3], (num_experts, ffn, hidden), jnp.float32) / np.sqrt(ffn)
  g = jax.random.normal(keys[4], (n, hidden), jnp.float32)
  return x, ids, weights, w_up, w_down, g


def _chunked(mesh, cfg):
  spec = P("ep")
  f = functools.partial(chunked_moe, cfg=cfg, axis_name="ep")
  return jax.shard_map(
      lambda x, i, w, u, d: f(x, i, w, u, d),
      mesh=mesh,
      in_specs=(spec, spec, spec, spec, spec),
      out_specs=spec,
  )


@pytest.mark.parametrize("skew", ["uniform", "rank0", "all_one"])
@pytest.mark.parametrize("slice_size", [3, 8, 64])
@pytest.mark.parametrize("stripe", [1, 4])
def test_forward_and_grads_match_reference(skew, slice_size, stripe):
  _check(skew, slice_size, stripe, comm_dtype=None, tol=1e-4)


@pytest.mark.parametrize("skew", ["uniform", "all_one"])
def test_fp8_wire_is_close(skew):
  # Rows travel as float8_e4m3fn with per-row scales (about 2 significant digits).
  _check(skew, slice_size=8, stripe=4, comm_dtype="float8_e4m3fn", tol=None, max_rel_err=0.08)


def _rel_err(a, b):
  return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def _check(skew, slice_size, stripe, comm_dtype, tol, max_rel_err=None):
  ep, tokens, experts, top_k, hidden, ffn = 4, 16, 8, 2, 8, 12
  mesh = jax.make_mesh((ep,), ("ep",))
  cfg = ChunkedMoEConfig(
      num_experts=experts,
      top_k=top_k,
      slice_size=slice_size,
      transport="emulated",
      comm_dtype=comm_dtype,
      stripe_width=stripe,
  )
  x, ids, weights, w_up, w_down, g = _setup(ep, tokens, experts, top_k, hidden, ffn, skew)
  put = lambda a: jax.device_put(a, NamedSharding(mesh, P("ep")))

  chunked = _chunked(mesh, cfg)

  def chunked_loss(x, w, u, d):
    return jnp.sum(chunked(x, put(ids), w, u, d) * g)

  ref = functools.partial(reference_moe, stripe=stripe)

  def ref_loss(x, w, u, d):
    return jnp.sum(ref(x, ids, w, u, d) * g)

  args = (x, weights, w_up, w_down)
  out = jax.jit(chunked)(*map(put, (x, ids, weights, w_up, w_down)))
  got = jax.jit(jax.grad(chunked_loss, argnums=(0, 1, 2, 3)))(*map(put, args))
  want = jax.grad(ref_loss, argnums=(0, 1, 2, 3))(*args)
  pairs = [("out", out, ref(x, ids, weights, w_up, w_down))]
  pairs += list(zip(["dx", "dweights", "dw_up", "dw_down"], got, want))
  for name, a, b in pairs:
    if max_rel_err is None:
      np.testing.assert_allclose(a, b, rtol=tol, atol=tol, err_msg=name)
    else:
      assert _rel_err(a, b) < max_rel_err, f"{name}: relative error {_rel_err(a, b):.4f}"
