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
@pytest.mark.parametrize("comm_dtype", [None])
def test_forward_and_grads_match_reference(skew, slice_size, comm_dtype):
  ep, tokens, experts, top_k, hidden, ffn = 4, 16, 8, 2, 8, 12
  mesh = jax.make_mesh((ep,), ("ep",))
  cfg = ChunkedMoEConfig(num_experts=experts, top_k=top_k, slice_size=slice_size, transport="emulated", comm_dtype=comm_dtype)
  x, ids, weights, w_up, w_down, g = _setup(ep, tokens, experts, top_k, hidden, ffn, skew)
  put = lambda a: jax.device_put(a, NamedSharding(mesh, P("ep")))

  chunked = _chunked(mesh, cfg)

  def chunked_loss(x, w, u, d):
    return jnp.sum(chunked(x, put(ids), w, u, d) * g)

  def ref_loss(x, w, u, d):
    return jnp.sum(reference_moe(x, ids, w, u, d) * g)

  args = (x, weights, w_up, w_down)
  out = jax.jit(chunked)(*map(put, (x, ids, weights, w_up, w_down)))
  np.testing.assert_allclose(out, reference_moe(x, ids, weights, w_up, w_down), rtol=1e-5, atol=1e-5)

  got = jax.jit(jax.grad(chunked_loss, argnums=(0, 1, 2, 3)))(*map(put, args))
  want = jax.grad(ref_loss, argnums=(0, 1, 2, 3))(*args)
  for name, a, b in zip(["dx", "dweights", "dw_up", "dw_down"], got, want):
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=name)
