"""Properties of the slice plan, checked directly rather than through numerics."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from chunked_moe import plan as plan_lib

EP, TOKENS, EXPERTS, TOP_K = 4, 32, 8, 2


def _ids(skew, seed=0):
  logits = jax.random.normal(jax.random.key(seed), (EP * TOKENS, EXPERTS))
  if skew:
    logits = logits.at[:, 0].add(100.0)
  return jax.lax.top_k(logits, TOP_K)[1].astype(jnp.int32)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("slice_size", [1, 5, 16, 1000])
def test_slices_cover_streams_exactly(skew, slice_size):
  mesh = jax.make_mesh((EP,), ("ep",))
  max_slices = EP * TOKENS * TOP_K // slice_size + 2  # static upper bound for the test

  @functools.partial(jax.shard_map, mesh=mesh, in_specs=P("ep"), out_specs=P("ep"))
  def per_rank(ids):
    plan = plan_lib.make_plan(ids, EXPERTS, slice_size, "ep")

    def one(j):
      d = plan_lib.dispatch_args(plan, j, slice_size, "ep")
      c = plan_lib.combine_args(plan, j, slice_size, "ep")
      g = plan_lib.slice_group_sizes(plan, j, slice_size, "ep")
      return d.send_sizes, d.recv_sizes, c.send_sizes, c.recv_sizes, g

    per_slice = jax.vmap(one)(jnp.arange(max_slices))
    me = jax.lax.axis_index("ep")
    stream_rows = plan.expert_rows[me].sum()
    return tuple(a[None] for a in per_slice) + (plan.num_slices[None], stream_rows[None], plan.counts[None])

  ids = jax.device_put(_ids(skew), NamedSharding(mesh, P("ep")))
  send, recv, c_send, c_recv, groups, num_slices, stream_rows, counts = map(np.asarray, jax.jit(per_rank)(ids))

  counts = counts[0]  # identical on every rank: [source, receiver, local expert]
  local = EXPERTS // EP
  assert len(set(num_slices.tolist())) == 1, "all ranks must loop the same number of times"
  n = int(num_slices[0])
  assert n == -(-int(stream_rows.max()) // slice_size)

  for r in range(EP):
    # No padding is ever communicated: received rows over all slices == real stream rows.
    assert recv[r].sum() == stream_rows[r] == counts[:, r, :].sum()
    # Each slice fits in S rows, and only the last active slice may be partial.
    per_slice_rows = recv[r].sum(axis=1)
    assert (per_slice_rows <= slice_size).all()
    assert (per_slice_rows[n:] == 0).all()
    active = per_slice_rows[: -(-int(stream_rows[r]) // slice_size)]
    assert (active[:-1] == slice_size).all()
    # Group sizes in each slice add up to its rows and follow the expert order.
    np.testing.assert_array_equal(groups[r].sum(axis=1), per_slice_rows)
    # Every (source, local expert) segment is delivered exactly once, and returned once.
    np.testing.assert_array_equal(recv[r].sum(axis=0).reshape(EP, local), counts[:, r, :])
    np.testing.assert_array_equal(c_send[r].sum(axis=0).reshape(EP, local), counts[:, r, :])
  for s in range(EP):
    np.testing.assert_array_equal(send[s].sum(axis=0).reshape(EP, local), counts[s])
    np.testing.assert_array_equal(c_recv[s], send[s])  # results come back in the shape they left
