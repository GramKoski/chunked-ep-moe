# chunked-ep-moe

A guaranteed-dropless, chunked expert-parallel MoE layer for JAX, modelled on
the design Cohere described at GTC ("Fully Dropless Sparse Model Training in
JAX"), with two transports (`ragged_all_to_all` and one-sided NVSHMEM-style
puts from a Mosaic GPU kernel), fused Hopper kernels, and a MaxText
integration (`moe_ep_impl=chunked`).

## The idea

Dropless MoE under XLA's static shapes usually means sizing receive buffers for
the worst case (every token routed to one rank), which grows with the EP size.
This layer never does that:

1. Each rank sorts its `T*K` token copies by expert. Everyone all-gathers the
   `[P, P, L]` traffic matrix (tiny), so every rank knows the full plan.
2. Each receiver lays out everything it will get as one **packed, expert-major
   stream** (no padding), and the stream is cut into fixed **slices** of `S`
   rows. All ranks loop `max_r ceil(R_r / S)` times; empty tail slices are
   skipped, so at most `S - 1` padding rows are ever moved or computed.
3. Senders write their rows straight into **expert-major positions** of the
   receiver's slice buffer (one segment per receiver x local expert), so a
   slice arrives already grouped by expert and goes straight into the grouped
   GEMM, with no local re-sort.
4. The forward loop issues slice `j+1`'s dispatch before slice `j`'s compute,
   so communication can hide behind the GEMMs.
5. The backward is **one loop**: per slice it re-dispatches the inputs and
   output gradients, recomputes the slice's forward, and runs its backward
   immediately (full remat; only the integer routing plan is saved).

Every per-iteration buffer is `S` rows regardless of how imbalanced routing is.

| Cohere talk | Here |
|---|---|
| Fully dropless, packed layout, skip padding slices | `plan.py` slice planner; empty tail slices skipped |
| Pipelining over token chunks, comm/compute overlap | dispatch `j+1` issued before compute `j` (`moe.py`) |
| Remat + backward in a single loop | `_chunked_moe_bwd` |
| NVSHMEM one-sided puts + "tokens arrived" signal | `transport/nvshmem.py` (Mosaic GPU `remote_ref` + semaphores) |
| `ragged_all_to_all` fallback | `transport/ragged.py` |
| Striped SwiGLU fused into the expansion GEMM epilogue | `kernels/fused_expert.py: gmm_swiglu` |
| Fused SwiGLU gradient | `kernels/fused_expert.py: gmm_dswiglu` |
| FP8 token comms | `comm_dtype="float8_e4m3fn"` (per-row scales) |

## Layout

```
src/chunked_moe/
  config.py            ChunkedMoEConfig
  plan.py              traffic matrix, slice plan, ragged-exchange arguments
  moe.py               chunked_moe: custom_vjp, forward loop, fused remat+backward loop
  expert.py            reference per-slice MLP (ragged_dot) with striped SwiGLU
  reference.py         dense single-device oracle
  transport/ragged.py  ragged_all_to_all, and a dense emulation for CPU tests
  transport/nvshmem.py one-sided put + signal kernel (Mosaic GPU)
  kernels/fused_expert.py  Hopper wgmma kernels (fused epilogues, weight-grad GEMMs)
scripts/compile_gpu_aot.py  compile the layer for GPUs without a GPU
tests/                      CPU tests (simulated devices, Mosaic GPU interpreter)
```

## Usage

```python
from chunked_moe import ChunkedMoEConfig, chunked_moe, stripe_weights

cfg = ChunkedMoEConfig(num_experts=256, top_k=8, slice_size=8192,
                       transport="ragged_a2a", expert_compute="fused")
# inside jax.shard_map over the expert-parallel axis "ep":
w_up = stripe_weights(w_gate, w_up_proj, cfg.stripe_width)    # [L, D, 2F]
out = chunked_moe(x, expert_ids, router_weights, w_up, w_down, cfg, "ep")
```

In MaxText (branch `chunked-ep-moe` of the maxtext checkout):

```
moe_ep_impl=chunked chunked_moe_transport=ragged_a2a chunked_moe_expert_compute=fused \
chunked_moe_slice_size=8192 [chunked_moe_comm_dtype=float8_e4m3fn]
```

MaxText's own router (top-k, routed bias, routed scaling, load-balance loss,
bias-update counts) is kept; dispatch, expert compute and combine are replaced.

## What is verified, and how

There was no GPU available, so verification is split three ways.

**Numerically tested on CPU** (`pytest`, 8 simulated devices):
- Forward and all gradients (x, router weights, w_up, w_down) of the full
  layer vs. a dense reference, across slice sizes 3/8/64, stripe widths 1/4,
  and routing from uniform to "every token's first choice is expert 0".
- FP8 wire format within the expected e4m3 error.
- Every fused Mosaic GPU kernel, run in JAX's Mosaic GPU interpreter
  (`tests/test_fused_kernels.py`), including partial slices and empty experts.
  Two interpreter gaps are filled by test-only shims (`tests/interpret_shims.py`).
- MaxText end to end: a 5-layer DeepSeek-V3-tiny trains for 4 steps with the
  same losses as MaxText's default MoE (to bf16 rounding), at EP=1 and EP=4.

**Compiled for GPUs, deviceless** (XLA:GPU from an H100 target spec):
- The layer, forward + backward, at DeepSeek-V3 MoE sizes on 8 x H100, for
  `ragged_a2a` + reference, `ragged_a2a` + fused, and the put/signal transport
  + fused (11 Mosaic GPU kernels in the optimized HLO).
- DeepSeek-V3 through MaxText's `train_compile` (see the maxtext branch).

**Not verified** (needs real GPUs):
- Run-time behaviour of the put/signal kernel: the ready/done semaphore
  protocol, memory ordering of the signal, and whether XLA keeps the aliased
  receive buffer in symmetric memory across loop iterations without copies.
- Performance of anything, including whether XLA actually overlaps the
  `ragged_all_to_all` of slice j+1 with slice j's GEMMs.

## Known limitations

- **NVSHMEM proper**: the jaxlib 0.11.1 / jax-cuda12-plugin wheels used here
  have no NVSHMEM support in XLA, so the transport kernel was compiled with
  Mosaic's other peer-memory lowering (XLA-managed peer pointers, one NVLink
  domain). The kernel source is the same for both lowerings;
  `enable_nvshmem_lowering()` selects NVSHMEM on a build that has it. The
  cu12 NVSHMEM wheel also has no sm_100 bitcode, so NVSHMEM on Blackwell needs
  the CUDA 13 packages.
- Narrow rows (the router weight and its gradient, 4 bytes/row) always use
  `ragged_all_to_all`.
- Fused kernels are Hopper (wgmma) only and need S, D multiples of 64 and the
  FFN width a multiple of the stripe width.
- No expert tensor parallelism; MaxText validation rejects TP > 1.
- MaxText stores gate and up separately, so the integration stripes them every
  step (a weight-sized copy per layer).
- Not implemented from the talk: the fused router (GEMM + sigmoid + top-k
  epilogue), MXFP8 GEMMs with the fused requantize-transpose, and kernel-level
  fusion of the communication into the GEMMs.
- Compile-only GPU devices all report `slice_index` 0, so a deviceless compile
  treats all GPUs as one ICI domain.

## Deviceless GPU compilation

```
python scripts/compile_gpu_aot.py --spec h100_sxm.txtpb --topology 1x1x8 \
    --transport nvshmem --expert-compute fused
```

Specs live in XLA at `xla/backends/gpu/target_config/specs/`. The script sets
`--xla_gpu_autotune_level=0` and `--xla_gpu_enable_cub_radix_sort=false`: GEMM
autotuning and the CUB sort rewrite both query a real device at compile time.
