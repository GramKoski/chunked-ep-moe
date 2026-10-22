"""Dropless, chunked expert-parallel MoE layer.

Call `chunked_moe` inside `shard_map` over the expert-parallel axis. Per rank:

  forward:   sort token copies by expert -> plan slices -> loop over slices:
               dispatch slice j+1 | expert MLP on slice j | return slice j
             -> weighted sum of each token's K expert outputs
  backward:  one loop over slices that re-dispatches inputs and output
             gradients, recomputes the slice's forward, and runs its backward
             right away. Only the routing plan is saved between the passes.

Slices bound every per-iteration buffer to S rows no matter how imbalanced the
routing is, which is what makes guaranteed-dropless affordable: no buffer is
ever sized for the worst case. Issuing slice j+1's dispatch before slice j's
compute gives the scheduler (or the NVSHMEM kernel) the overlap.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp

from chunked_moe import expert
from chunked_moe import plan as plan_lib
from chunked_moe import util
from chunked_moe.config import ChunkedMoEConfig
from chunked_moe.transport import ragged


def _exchange_fn(cfg: ChunkedMoEConfig):
  if cfg.transport == "ragged_a2a":
    return ragged.ragged_exchange
  if cfg.transport == "emulated":
    return ragged.emulated_exchange
  if cfg.transport == "nvshmem":
    from chunked_moe.transport import nvshmem  # pylint: disable=import-outside-toplevel

    return functools.partial(nvshmem.nvshmem_exchange, num_sms=cfg.num_sms)
  raise ValueError(f"Unknown transport {cfg.transport!r}")


def _expert_fns(cfg: ChunkedMoEConfig):
  if cfg.expert_compute == "reference":
    module = expert
  elif cfg.expert_compute == "fused":
    from chunked_moe.kernels import fused_expert as module  # pylint: disable=import-outside-toplevel
  else:
    raise ValueError(f"Unknown expert_compute {cfg.expert_compute!r}")
  kw = dict(stripe=cfg.stripe_width)
  if cfg.expert_compute == "fused":
    kw["num_sms"] = cfg.num_sms
  return functools.partial(module.expert_forward, **kw), functools.partial(module.expert_backward, **kw)


def _quantize_rows(x, comm_dtype):
  """Per-row absmax quantization for the wire. Returns (values, scales[n, 1])."""
  if comm_dtype is None:
    return x, None
  dtype = jnp.dtype(comm_dtype)
  amax = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=-1, keepdims=True)
  scale = jnp.maximum(amax, 1e-12) / float(jnp.finfo(dtype).max)
  return (x.astype(jnp.float32) / scale).astype(dtype), scale


def _dequantize_rows(q, scale, dtype):
  if scale is None:
    return q
  return (q.astype(jnp.float32) * scale).astype(dtype)


class _Dispatcher:
  """Moves rows from this rank's send order into slice buffers, optionally quantized on the wire."""

  def __init__(self, cfg, plan, axis_name, rows, compute_dtype):
    self.cfg, self.plan, self.axis_name = cfg, plan, axis_name
    self.exchange = _exchange_fn(cfg)
    self.compute_dtype = compute_dtype
    self.values, self.scales = _quantize_rows(rows, cfg.comm_dtype)

  def __call__(self, j):
    s = self.cfg.slice_size
    args = plan_lib.dispatch_args(self.plan, j, s, self.axis_name)
    values = self.exchange(self.values, util.zeros((s,) + self.values.shape[1:], self.values.dtype, self.axis_name), args, self.axis_name)
    scales = None
    if self.scales is not None:
      scales = self.exchange(self.scales, util.zeros((s, 1), self.scales.dtype, self.axis_name), args, self.axis_name)
    return _dequantize_rows(values, scales, self.compute_dtype)


def _send_order_rows(plan, num_experts_per_tok, *arrays):
  token_of_row = plan.order // num_experts_per_tok
  return token_of_row, [a[token_of_row] for a in arrays]


def _forward(x, expert_ids, weights, w_up, w_down, cfg: ChunkedMoEConfig, axis_name):
  num_tokens, hidden = x.shape
  k = cfg.top_k
  s = cfg.slice_size
  exchange = _exchange_fn(cfg)
  fwd, _ = _expert_fns(cfg)

  plan = plan_lib.make_plan(expert_ids, cfg.num_experts, s, axis_name)
  token_of_row, (send_x,) = _send_order_rows(plan, k, x)
  dispatch = _Dispatcher(cfg, plan, axis_name, send_x, x.dtype)

  def body(carry):
    j, cur, ret = carry
    nxt = dispatch(j + 1)  # independent of this slice's compute, so it can overlap it
    y, _ = fwd(cur, w_up, w_down, plan_lib.slice_group_sizes(plan, j, s, axis_name), need_residuals=False)
    ret = exchange(y, ret, plan_lib.combine_args(plan, j, s, axis_name), axis_name)
    return j + 1, nxt, ret

  init = (util.varying(jnp.int32(0), axis_name), dispatch(0), util.zeros((num_tokens * k, hidden), x.dtype, axis_name))
  _, _, ret = jax.lax.while_loop(lambda c: c[0] < plan.num_slices, body, init)

  row_weights = weights.reshape(-1)[plan.order].astype(jnp.float32)
  out = jnp.zeros((num_tokens, hidden), jnp.float32).at[token_of_row].add(ret.astype(jnp.float32) * row_weights[:, None])
  return out.astype(x.dtype), plan


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def chunked_moe(x, expert_ids, weights, w_up, w_down, cfg: ChunkedMoEConfig, axis_name):
  """Dropless chunked MoE over the expert-parallel axis `axis_name`.

  Args:
    x: [T, D] this rank's tokens.
    expert_ids: [T, K] int32 global expert ids from the router.
    weights: [T, K] combine weights from the router (differentiable).
    w_up: [L, D, 2F] this rank's experts' striped gate/up weights.
    w_down: [L, F, D] this rank's experts' down weights.
    cfg: Static configuration.
    axis_name: Expert-parallel mesh axis (size P, with E = P * L).

  Returns:
    [T, D] MoE output for this rank's tokens.
  """
  out, _ = _forward(x, expert_ids, weights, w_up, w_down, cfg, axis_name)
  return out


def _chunked_moe_fwd(x, expert_ids, weights, w_up, w_down, cfg, axis_name):
  out, plan = _forward(x, expert_ids, weights, w_up, w_down, cfg, axis_name)
  # Full remat: only the inputs and the integer routing plan are kept.
  return out, (x, weights, w_up, w_down, plan)


def _chunked_moe_bwd(cfg, axis_name, residuals, g_out):
  x, weights, w_up, w_down, plan = residuals
  num_tokens, hidden = x.shape
  k = cfg.top_k
  s = cfg.slice_size
  exchange = _exchange_fn(cfg)
  fwd, bwd = _expert_fns(cfg)
  accum = jnp.dtype(cfg.accum_dtype)

  token_of_row, (send_x, send_g) = _send_order_rows(plan, k, x, g_out.astype(x.dtype))
  row_weights = weights.reshape(-1)[plan.order].astype(jnp.float32)[:, None]
  dispatch_x = _Dispatcher(cfg, plan, axis_name, send_x, x.dtype)
  dispatch_g = _Dispatcher(cfg, plan, axis_name, send_g, x.dtype)
  # Router weights travel unquantized: one float per row.
  dispatch_w = _Dispatcher(dataclasses.replace(cfg, comm_dtype=None), plan, axis_name, row_weights, jnp.float32)

  def body(carry):
    j, cur, dx_rows, dw_rows, dw_up, dw_down = carry
    nxt = (dispatch_x(j + 1), dispatch_g(j + 1), dispatch_w(j + 1))
    cur_x, cur_g, cur_w = cur
    group_sizes = plan_lib.slice_group_sizes(plan, j, s, axis_name)

    # Recompute this slice's forward, then run its backward in the same iteration.
    y, res = fwd(cur_x, w_up, w_down, group_sizes)
    dy = (cur_w * cur_g.astype(jnp.float32)).astype(x.dtype)
    d_weight = jnp.sum(cur_g.astype(jnp.float32) * y.astype(jnp.float32), axis=-1, keepdims=True)
    grads = bwd(cur_x, w_up, w_down, group_sizes, res, dy)

    back = plan_lib.combine_args(plan, j, s, axis_name)
    dx_rows = exchange(grads.dx, dx_rows, back, axis_name)
    dw_rows = exchange(d_weight, dw_rows, back, axis_name)
    return (
        j + 1,
        nxt,
        dx_rows,
        dw_rows,
        dw_up + grads.dw_up.astype(accum),
        dw_down + grads.dw_down.astype(accum),
    )

  init = (
      util.varying(jnp.int32(0), axis_name),
      (dispatch_x(0), dispatch_g(0), dispatch_w(0)),
      util.zeros((num_tokens * k, hidden), x.dtype, axis_name),
      util.zeros((num_tokens * k, 1), jnp.float32, axis_name),
      util.zeros(w_up.shape, accum, axis_name),
      util.zeros(w_down.shape, accum, axis_name),
  )
  _, _, dx_rows, dw_rows, dw_up, dw_down = jax.lax.while_loop(lambda c: c[0] < plan.num_slices, body, init)

  dx = jnp.zeros((num_tokens, hidden), jnp.float32).at[token_of_row].add(dx_rows.astype(jnp.float32))
  d_weights = jnp.zeros((num_tokens * k,), jnp.float32).at[plan.order].set(dw_rows[:, 0]).reshape(weights.shape)
  return (
      dx.astype(x.dtype),
      None,
      d_weights.astype(weights.dtype),
      dw_up.astype(w_up.dtype),
      dw_down.astype(w_down.dtype),
  )


chunked_moe.defvjp(_chunked_moe_fwd, _chunked_moe_bwd)
