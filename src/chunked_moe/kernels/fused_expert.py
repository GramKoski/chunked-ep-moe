"""Fused Mosaic GPU (Hopper wgmma) kernels for the per-slice expert MLP.

Forward:   act = SwiGLU(x @ w_up)       one kernel, SwiGLU in the epilogue
           y   = act @ w_down           grouped GEMM
Backward:  dh  = SwiGLU'(dy @ w_down^T) one kernel, the SwiGLU gradient in the
                                        epilogue (d_act never reaches HBM)
           dx  = dh @ w_up^T            grouped GEMM
           dW  = x^T dh, act^T dy       transposed grouped GEMMs (weight grads)

The slice is expert-major (rows of each local expert are contiguous), which is
what every kernel here assumes. The grouped GEMMs follow
jax.experimental.pallas.ops.gpu.ragged_dot_mgpu and transposed_ragged_dot_mgpu,
with the SM count passed in (deviceless compiles have no device to ask), a
float32 output for weight gradients, and the two fused epilogues added.

Constraints: the slice rows S and every K dimension are multiples of the block
sizes below, and the FFN width is a multiple of the stripe width.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from chunked_moe.expert import ExpertGrads, ForwardResiduals

BLOCK_M = 64
BLOCK_K = 64
MAX_CONCURRENT_STEPS = 2
_PARAMS = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup)


@dataclasses.dataclass(frozen=True)
class _GroupInfo:
  """Which expert (group) and which rows the current M block covers."""

  group_id: jax.Array
  block: jax.Array
  block_start: jax.Array
  start_within_block: jax.Array
  actual_size: jax.Array

  @classmethod
  def create(cls, group_sizes_ref, num_groups, tile, tid):
    # A block that straddles a group boundary is visited once per group, so the
    # M grid has cdiv(m, tile) + num_groups - 1 entries.
    tile = jnp.int32(tile)
    group_end = group_start = block = group = end = jnp.int32(0)
    for i in range(num_groups):
      start = end
      end = start + group_sizes_ref[i]
      start_block = lax.div(start, tile)
      block_end = lax.div(end - 1, tile) + 1
      tid_begin = start_block + i
      tid_end = block_end + i
      mine = (tid_begin <= tid) & (tid < tid_end)
      block = lax.select(mine, tid - tid_begin + start_block, block)
      group = lax.select(mine, jnp.int32(i), group)
      group_start = lax.select(mine, start, group_start)
      group_end = lax.select(mine, end, group_end)
    block_start = block * tile
    actual_start = jnp.maximum(group_start, block_start)
    actual_end = jnp.minimum(group_end, block_start + tile)
    return cls(group, block, block_start, actual_start - block_start, actual_end - actual_start)


def _store_rows(o_smem, o_gmem_cols, info: _GroupInfo, block_m, m):
  """Copies this group's rows of the tile to GMEM. TMA boxes are static, so the
  dynamic row count is covered with a ladder of power-of-two row copies."""
  smem_start = info.start_within_block
  remaining = min(block_m, m)
  while remaining > 0:
    rows = 1 << int(math.log2(remaining))
    remaining //= 2

    @pl.when(info.actual_size & rows != 0)
    def _():
      plgpu.copy_smem_to_gmem(
          o_smem.at[pl.ds(smem_start, rows)],
          o_gmem_cols.at[pl.ds(info.block_start + smem_start, rows)],
      )

    smem_start += info.actual_size & rows


def _kernel(body, out_type, num_sms):
  return plgpu.kernel(
      body, out_type=out_type, grid=(num_sms,), grid_names=("sm",), compiler_params=_PARAMS
  )


def _check(m, k):
  if m % BLOCK_M or k % BLOCK_K:
    raise ValueError(f"rows ({m}) and contraction dim ({k}) must be multiples of {BLOCK_M} and {BLOCK_K}")


def gmm(lhs, rhs, group_sizes, *, block_n=128, transpose_rhs=False, num_sms=132):
  """Grouped GEMM. lhs [M, K]; rhs [G, K, N] (or [G, N, K] with transpose_rhs) -> [M, N]."""
  m, k = lhs.shape
  g = rhs.shape[0]
  n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
  _check(m, k)

  def body(gs_ref, lhs_ref, rhs_ref, o_ref):
    grid_m = pl.cdiv(m, BLOCK_M) + g - 1
    grid_n = pl.cdiv(n, block_n)

    @plgpu.nd_loop((grid_m * grid_n,), collective_axes="sm")
    def _tile(loop):
      mi, ni = plgpu.planar_snake(loop.index[0], (grid_m, grid_n), 1, 1)
      info = _GroupInfo.create(gs_ref, g, BLOCK_M, mi)

      def acc_scope(acc_ref):
        plgpu.emit_pipeline(
            lambda _, a, b: plgpu.wgmma(acc_ref, a, plgpu.transpose_ref(b, (1, 0)) if transpose_rhs else b),
            grid=(k // BLOCK_K,),
            in_specs=[
                plgpu.BlockSpec((BLOCK_M, BLOCK_K), lambda kk: (info.block, kk), delay_release=1),
                plgpu.BlockSpec(
                    (block_n, BLOCK_K) if transpose_rhs else (BLOCK_K, block_n),
                    (lambda kk: (ni, kk)) if transpose_rhs else (lambda kk: (kk, ni)),
                    delay_release=1,
                ),
            ],
            max_concurrent_steps=MAX_CONCURRENT_STEPS,
        )(lhs_ref, rhs_ref.at[info.group_id])
        return acc_ref[...]

      acc = pl.run_scoped(acc_scope, plgpu.ACC((BLOCK_M, block_n)))

      @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM((BLOCK_M, block_n), o_ref.dtype))
      def _store(o_smem):
        o_smem[...] = acc.astype(o_smem.dtype)
        plgpu.commit_smem()
        _store_rows(o_smem, o_ref.at[:, pl.ds(ni * block_n, block_n)], info, BLOCK_M, m)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  return _kernel(body, jax.ShapeDtypeStruct((m, n), lhs.dtype), num_sms)(group_sizes, lhs, rhs)


def gmm_swiglu(x, w_up, group_sizes, *, stripe, return_h, num_sms=132):
  """act = SwiGLU(x @ w_up) for striped w_up [G, K, 2F]. Also returns h [M, 2F] if asked.

  Each output tile is `stripe` columns of act. Its block loads gate stripe 2*ni
  and up stripe 2*ni + 1 of the same weights into two accumulators, so the
  SwiGLU happens in registers.
  """
  m, k = x.shape
  g, _, two_f = w_up.shape
  ffn = two_f // 2
  _check(m, k)
  if ffn % stripe:
    raise ValueError(f"ffn={ffn} must be a multiple of stripe={stripe}")
  grid_n = ffn // stripe

  def body(gs_ref, x_ref, w_ref, act_ref, *h_refs):
    grid_m = pl.cdiv(m, BLOCK_M) + g - 1

    @plgpu.nd_loop((grid_m * grid_n,), collective_axes="sm")
    def _tile(loop):
      mi, ni = plgpu.planar_snake(loop.index[0], (grid_m, grid_n), 1, 1)
      info = _GroupInfo.create(gs_ref, g, BLOCK_M, mi)

      def acc_scope(acc_gate, acc_up):
        def step(_, a, b_gate, b_up):
          plgpu.wgmma(acc_gate, a, b_gate)
          plgpu.wgmma(acc_up, a, b_up)

        plgpu.emit_pipeline(
            step,
            grid=(k // BLOCK_K,),
            in_specs=[
                plgpu.BlockSpec((BLOCK_M, BLOCK_K), lambda kk: (info.block, kk), delay_release=1),
                plgpu.BlockSpec((BLOCK_K, stripe), lambda kk: (kk, 2 * ni), delay_release=1),
                plgpu.BlockSpec((BLOCK_K, stripe), lambda kk: (kk, 2 * ni + 1), delay_release=1),
            ],
            max_concurrent_steps=MAX_CONCURRENT_STEPS,
        )(x_ref, w_ref.at[info.group_id], w_ref.at[info.group_id])
        return acc_gate[...], acc_up[...]

      gate, up = pl.run_scoped(acc_scope, plgpu.ACC((BLOCK_M, stripe)), plgpu.ACC((BLOCK_M, stripe)))
      act = gate * up / (1.0 + jnp.exp(-gate))  # silu(gate) * up, still in registers

      outs = [(act, act_ref.at[:, pl.ds(ni * stripe, stripe)])]
      if return_h:
        (h_ref,) = h_refs
        outs.append((gate, h_ref.at[:, pl.ds(2 * ni * stripe, stripe)]))
        outs.append((up, h_ref.at[:, pl.ds((2 * ni + 1) * stripe, stripe)]))

      @functools.partial(pl.run_scoped, smem=[plgpu.SMEM((BLOCK_M, stripe), x.dtype)] * len(outs))
      def _store(smem):
        for buf, (val, dst) in zip(smem, outs):
          buf[...] = val.astype(buf.dtype)
        plgpu.commit_smem()
        for buf, (_, dst) in zip(smem, outs):
          _store_rows(buf, dst, info, BLOCK_M, m)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out_type = [jax.ShapeDtypeStruct((m, ffn), x.dtype)]
  if return_h:
    out_type.append(jax.ShapeDtypeStruct((m, two_f), x.dtype))
  outs = _kernel(body, out_type, num_sms)(group_sizes, x, w_up)
  return (outs[0], outs[1]) if return_h else (outs[0], None)


def gmm_dswiglu(dy, w_down, h, group_sizes, *, stripe, num_sms=132):
  """dh = SwiGLU'(h) applied to (dy @ w_down^T), returned in the striped [M, 2F] layout.

  w_down is [G, F, D]. Each tile computes `stripe` columns of d_act, loads the
  matching gate and up stripes of h, and writes both stripes of dh.
  """
  m, k = dy.shape  # k = D
  g, ffn, _ = w_down.shape
  _check(m, k)
  if ffn % stripe:
    raise ValueError(f"ffn={ffn} must be a multiple of stripe={stripe}")
  grid_n = ffn // stripe

  def body(gs_ref, dy_ref, w_ref, h_ref, dh_ref):
    grid_m = pl.cdiv(m, BLOCK_M) + g - 1

    @plgpu.nd_loop((grid_m * grid_n,), collective_axes="sm")
    def _tile(loop):
      mi, ni = plgpu.planar_snake(loop.index[0], (grid_m, grid_n), 1, 1)
      info = _GroupInfo.create(gs_ref, g, BLOCK_M, mi)

      def acc_scope(acc_ref):
        plgpu.emit_pipeline(
            lambda _, a, b: plgpu.wgmma(acc_ref, a, plgpu.transpose_ref(b, (1, 0))),
            grid=(k // BLOCK_K,),
            in_specs=[
                plgpu.BlockSpec((BLOCK_M, BLOCK_K), lambda kk: (info.block, kk), delay_release=1),
                plgpu.BlockSpec((stripe, BLOCK_K), lambda kk: (ni, kk), delay_release=1),
            ],
            max_concurrent_steps=MAX_CONCURRENT_STEPS,
        )(dy_ref, w_ref.at[info.group_id])
        return acc_ref[...]

      d_act = pl.run_scoped(acc_scope, plgpu.ACC((BLOCK_M, stripe)))
      rows = pl.ds(info.block * BLOCK_M, BLOCK_M)

      @functools.partial(
          pl.run_scoped,
          h_smem=[plgpu.SMEM((BLOCK_M, stripe), h.dtype)] * 2,
          out_smem=[plgpu.SMEM((BLOCK_M, stripe), h.dtype)] * 2,
          bar=plgpu.Barrier(num_arrivals=2),
      )
      def _epilogue(h_smem, out_smem, bar):
        plgpu.copy_gmem_to_smem(h_ref.at[rows, pl.ds(2 * ni * stripe, stripe)], h_smem[0], bar)
        plgpu.copy_gmem_to_smem(h_ref.at[rows, pl.ds((2 * ni + 1) * stripe, stripe)], h_smem[1], bar)
        plgpu.barrier_wait(bar)
        gate = plgpu.layout_cast(h_smem[0][...], plgpu.Layout.WGMMA).astype(jnp.float32)
        up = plgpu.layout_cast(h_smem[1][...], plgpu.Layout.WGMMA).astype(jnp.float32)
        sig = 1.0 / (1.0 + jnp.exp(-gate))
        silu = gate * sig
        out_smem[0][...] = (d_act * up * (sig + silu * (1.0 - sig))).astype(h.dtype)
        out_smem[1][...] = (d_act * silu).astype(h.dtype)
        plgpu.commit_smem()
        _store_rows(out_smem[0], dh_ref.at[:, pl.ds(2 * ni * stripe, stripe)], info, BLOCK_M, m)
        _store_rows(out_smem[1], dh_ref.at[:, pl.ds((2 * ni + 1) * stripe, stripe)], info, BLOCK_M, m)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  return _kernel(body, jax.ShapeDtypeStruct((m, 2 * ffn), dy.dtype), num_sms)(group_sizes, dy, w_down, h)


def tgmm(lhs, rhs, group_sizes, *, block_m=64, block_n=128, num_sms=132):
  """Per-group lhs^T @ rhs over the ragged row dimension. lhs [R, M], rhs [R, N] -> float32 [G, M, N]."""
  r, m = lhs.shape
  _, n = rhs.shape
  g = group_sizes.shape[0]
  if m % block_m or n % block_n:
    raise ValueError(f"m={m} and n={n} must be multiples of {block_m} and {block_n}")
  gs = group_sizes.astype(jnp.int32)
  ends = jnp.cumsum(gs)
  starts = ends - gs
  block_starts = starts // BLOCK_K * BLOCK_K
  num_blocks = (-(ends // -BLOCK_K) * BLOCK_K - block_starts) // BLOCK_K
  swizzle = plgpu.find_swizzle(BLOCK_K * jnp.dtype(lhs.dtype).itemsize * 8)
  transforms = (plgpu.TilingTransform((8, swizzle // jnp.dtype(lhs.dtype).itemsize)), plgpu.SwizzleTransform(swizzle))

  def body(gs_ref, starts_ref, ends_ref, nblocks_ref, bstarts_ref, lhs_ref, rhs_ref, o_ref):
    grid_m, grid_n = m // block_m, n // block_n

    @plgpu.nd_loop((g, grid_m * grid_n), collective_axes="sm")
    def _tile(loop):
      gi = loop.index[0]
      mi, ni = plgpu.planar_snake(loop.index[1], (grid_m, grid_n), 1, 1)
      rows = pl.ds(pl.multiple_of(bstarts_ref[gi], BLOCK_K), r)  # may run past the end; never read there

      def acc_scope(acc_ref):
        def step(idx, a_smem, b_smem):
          (bi,) = idx
          iota_a = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (BLOCK_K, block_m), 0), plgpu.Layout.WGMMA)
          iota_b = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (BLOCK_K, block_n), 0), plgpu.Layout.WGMMA)

          # The first and last K blocks of a group also hold rows of neighbouring
          # groups or of the never-written tail of the slice, which may be
          # uninitialized (even NaN). Select them away in both operands; a
          # multiply by zero would keep NaNs.
          def keep_rows(lo, hi):
            a_smem[...] = jnp.where((iota_a >= lo) & (iota_a <= hi), a_smem[...], jnp.zeros((), a_smem.dtype))
            b_smem[...] = jnp.where((iota_b >= lo) & (iota_b <= hi), b_smem[...], jnp.zeros((), b_smem.dtype))
            plgpu.commit_smem()

          first = lax.rem(starts_ref[gi], BLOCK_K)
          last = lax.rem(ends_ref[gi] - 1, BLOCK_K)
          single = nblocks_ref[gi] == 1

          @pl.when((bi == 0) & single)
          def _():
            keep_rows(first, last)

          @pl.when((bi == 0) & ~single)
          def _():
            keep_rows(first, BLOCK_K - 1)

          @pl.when((bi == nblocks_ref[gi] - 1) & ~single)
          def _():
            keep_rows(0, last)

          plgpu.wgmma(acc_ref, plgpu.transpose_ref(a_smem, (1, 0)), b_smem)

        @pl.when(gs_ref[gi] > 0)
        def _():
          plgpu.emit_pipeline(
              step,
              grid=(nblocks_ref[gi],),
              in_specs=[
                  plgpu.BlockSpec((BLOCK_K, block_m), lambda kk: (kk, mi), delay_release=1, transforms=transforms),
                  plgpu.BlockSpec((BLOCK_K, block_n), lambda kk: (kk, ni), delay_release=1, transforms=transforms),
              ],
              max_concurrent_steps=MAX_CONCURRENT_STEPS,
          )(lhs_ref.at[rows, :], rhs_ref.at[rows, :])

        return acc_ref[...]

      acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

      @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM((block_m, block_n), jnp.float32))
      def _store(o_smem):
        o_smem[...] = acc
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_ref.at[gi, pl.ds(mi * block_m, block_m), pl.ds(ni * block_n, block_n)])
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

  out = _kernel(body, jax.ShapeDtypeStruct((g, m, n), jnp.float32), num_sms)
  return out(gs, starts, ends, num_blocks, block_starts, lhs, rhs)


def expert_forward(x, w_up, w_down, group_sizes, *, stripe, num_sms=132, need_residuals=True):
  act, h = gmm_swiglu(x, w_up, group_sizes, stripe=stripe, return_h=need_residuals, num_sms=num_sms)
  y = gmm(act, w_down, group_sizes, num_sms=num_sms)
  return y, ForwardResiduals(h, act)


def expert_backward(x, w_up, w_down, group_sizes, res: ForwardResiduals, dy, *, stripe, num_sms=132) -> ExpertGrads:
  dh = gmm_dswiglu(dy, w_down, res.h, group_sizes, stripe=stripe, num_sms=num_sms)
  dw_down = tgmm(res.act, dy, group_sizes, num_sms=num_sms)
  dx = gmm(dh, w_up, group_sizes, transpose_rhs=True, num_sms=num_sms)
  dw_up = tgmm(x, dh, group_sizes, num_sms=num_sms)
  return ExpertGrads(dx, dw_up, dw_down)
