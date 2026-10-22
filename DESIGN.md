# chunked-ep-moe: design, limitations, future work

This document explains how the layer works end to end, what is and is not
verified, and what is left to do. The README is the short version.

Notation: `P` expert-parallel (EP) ranks, `E` experts, `L = E / P` local
experts per rank, `K` experts per token (top-k), `T` tokens per rank, `D`
model width, `F` expert FFN width, `S` slice size in rows.

---

## 1. The problem

A MoE layer under expert parallelism has to move every token copy to the rank
that owns its expert, run that expert, and move the result back. The number of
copies each rank receives depends on the router, so it changes every step, but
XLA needs every buffer shape fixed at compile time.

The usual answers both give something up:

- **Capacity factor**: give each expert a fixed number of slots and drop the
  overflow. Tokens are lost, and the padding is computed and communicated.
- **Worst-case buffers**: size each receive buffer for "every token goes to one
  rank", which is guaranteed dropless but costs `min(P, E/K)` times the
  balanced size in memory (8x for DeepSeek-V3 at EP=8), and the factor grows
  with EP.

Cohere's design (GTC talk, "Fully Dropless Sparse Model Training in JAX") gets
guaranteed dropless without worst-case buffers by packing tokens densely and
processing them in fixed-size slices. This repo implements that design.

## 2. Algorithm

Everything below runs per rank inside `jax.shard_map` over the EP axis.

### 2.1 Routing plan (`plan.py`)

1. The caller provides `expert_ids [T, K]` and `weights [T, K]` from its router.
   This layer does not route; in MaxText the existing router is used.
2. The `T*K` token copies are sorted by global expert id (`order`). Expert ids
   are laid out rank-major (`e = r*L + l`), so after the sort all copies for
   receiver `r` are contiguous and, within that, grouped by local expert `l`.
   This is the **send order**.
3. Each rank counts its copies per expert and all-gathers them into the traffic
   matrix `counts[s, r, l]` (source, receiver, local expert). This is `P*P*L`
   integers, the only extra communication the plan needs, and it means every
   rank knows the full plan.
4. From `counts`:
   - `send_start[s, r*L + l]`: where source `s`'s copies for `(r, l)` begin in
     its send order (exclusive prefix sum).
   - The **receive stream** of rank `r` is everything it will get, packed and
     expert-major: local expert 0 from source 0, 1, ..., then local expert 1,
     and so on. `recv_start[r, l, s]` is where segment `(s, l)` starts in it.
     The stream has no padding: its length `R_r` is exactly the rows `r`
     receives.
   - `expert_start[r, l]`, `expert_rows[r, l]`: where each local expert's rows
     sit in the stream.
   - `num_slices = ceil(max_r R_r / S)`, identical on every rank.

### 2.2 Slices

Rank `r`'s stream is cut into slices of `S` rows; slice `j` covers stream rows
`[j*S, (j+1)*S)`. Every rank runs `num_slices` iterations so collectives line
up. A rank whose stream is shorter simply sends and receives zero rows in its
tail iterations. Consequences:

- **Bounded memory**: every per-iteration buffer is `S` rows, regardless of
  how imbalanced routing is. Nothing is sized for the worst case.
- **Bounded waste**: only the last active slice of each rank is partial, so at
  most `S - 1` padding rows exist per rank, and none of them are communicated
  (the ragged exchange only moves real rows).
- **Expert-major arrival**: since the stream is expert-major and each sender
  writes its rows at their stream positions, a slice arrives already grouped
  by expert. `slice_group_sizes` gives the per-expert row counts, and the
  grouped GEMM consumes the slice directly with no local re-sort (in the
  ragged path this is "scatter fused into the communication").

For slice `j`, `dispatch_args` intersects each `(r, l)` segment with
`[j*S, (j+1)*S)` and emits one ragged segment per (peer, local expert), i.e.
`P*L` segments per call: input offset in the send order, size, output offset
in the receiver's slice buffer, and the matching receive sizes.
`combine_args` is the exact reverse, sending each processed row back to its
position in the original sender's send order.

### 2.3 Forward (`moe.py: _forward`)

```
plan = make_plan(expert_ids)
send_x = x[order // K]                       # rows in send order
cur = dispatch(0)
for j in range(num_slices):                  # lax.while_loop
  nxt = dispatch(j + 1)                      # independent of this slice's compute
  y = expert_mlp(cur, group_sizes(j))        # grouped GEMMs on an S-row slice
  ret = combine(y, j, into=ret)              # back to send-order positions
  cur = nxt
out[t] = sum_k weights[t, k] * ret[pos(t, k)]
```

Dispatch of slice `j+1` is issued before compute of slice `j`, so the transfer
can overlap the GEMMs: XLA's scheduler handles that for `ragged_all_to_all`,
and the put/signal kernel does it by construction.

### 2.4 Backward: fused remat + backward (`moe.py: _chunked_moe_bwd`)

The layer is a `jax.custom_vjp`. The forward saves only `x`, the router
weights, the expert weights and the integer plan, i.e. full rematerialization.
The backward is a single loop:

```
for j in range(num_slices):
  x_j, g_j, w_j = dispatch x rows, output-gradient rows, router-weight rows
  h, act, y = expert forward on x_j            # recompute
  dy = w_j * g_j
  d_weight_row = <g_j, y>                      # gradient of the router weight
  dx_j, dW_up, dW_down = expert backward
  combine dx_j and d_weight_row back to the senders
  accumulate dW_up, dW_down (float32)
```

Recompute and backward share one iteration, so there are two loops per layer
(forward, backward) rather than three (forward, remat, backward), and no
per-slice activation is kept between the passes. The router-weight gradient is
computed on the receiver from `g` and the recomputed `y`, so only one float per
row comes back rather than the full `y`.

### 2.5 Expert MLP and striped SwiGLU (`expert.py`)

Each expert is SwiGLU: `y = (silu(x W_gate) * (x W_up)) W_down`. Gate and up are
stored together as one `[L, D, 2F]` matrix in a **striped** layout: stripes of
`stripe` columns alternate `[gate 0:s | up 0:s | gate s:2s | up s:2s | ...]`.

The point is the fused kernel: its output tile is one stripe wide, so the same
thread block loads gate stripe `2n` and up stripe `2n+1`, holds both
accumulators in registers, and applies SwiGLU there. The `[S, 2F]`
pre-activation never has to be written to HBM in the forward pass.
`stripe=1` is the per-column interleave used by gpt-oss checkpoints.
`stripe_weights` / `unstripe_weights` convert to and from separate matrices.

`expert.py` is the reference implementation on `jax.lax.ragged_dot` (grouped
GEMM) and `ragged_dot_general` with a ragged contracting dimension (weight
gradients), with the backward written out by hand so the layer controls what
is recomputed.

### 2.6 FP8 on the wire

With `comm_dtype="float8_e4m3fn"`, token rows (and output-gradient rows in the
backward) are quantized per row (absmax scale) before dispatch, the scales
travel alongside, and rows are dequantized on arrival. Compute stays in the
model dtype. This halves dispatch bytes relative to bf16.

## 3. Transports

Both implement the contract of `jax.lax.ragged_all_to_all`, with offsets laid
out as `[peer * L + segment]`, so `moe.py` does not know which one it is using.

### 3.1 `ragged_a2a` (`transport/ragged.py`)

A direct call to `jax.lax.ragged_all_to_all`. Multiple segments per peer are
supported by the op, which is what lets senders write expert-major positions.
On GPU, XLA lowers it to NCCL.

`emulated_exchange` reproduces the same semantics with a padded dense
`all_to_all` and is used only for CPU tests (XLA:CPU has no ragged-all-to-all).

### 3.2 Put/signal kernel (`transport/nvshmem.py`)

A Mosaic GPU (Pallas) kernel in the style of Cohere's NVSHMEM transport: the
sender writes rows straight into the receiver's buffer and signals it; the
receiver does not participate in the transfer.

One launch:

1. **Ready barrier.** Block 0 of every rank signals a `ready` semaphore on every
   peer, then waits for `P` signals, then releases its other blocks via a local
   `go` semaphore. Remote writes do not respect the receiver's stream order, so
   without this a fast sender could write into memory the receiver is still
   using for something else, or before the receiver's zero-fill of a fresh
   buffer.
2. **Puts.** Blocks stride over the rows of each `(peer, segment)`; each row is
   copied through registers in 128-thread vector chunks into
   `remote_ref(output, peer)`. Register copies handle arbitrary row counts
   without TMA's static box shapes.
3. **Done.** Every block signals a `done` semaphore on every receiver; block 0
   of each receiver waits for `num_blocks * P` signals, which is what makes the
   kernel's completion mean "all my incoming rows have landed".

The semaphores come from `pl.get_global` (persistent across launches). Each has
a single waiter that consumes exactly one launch's worth of signals, so a peer
that is already in its next launch only adds counts that the next launch will
consume. `output` is aliased in place (`input_output_aliases`), so the combine
can accumulate into the persistent return buffer across slices. Peers are
addressed with a dict device id (`_peer`): the EP-axis coordinates are given,
every other mesh axis defaults to the caller's own coordinate, which is
required inside MaxText's 12-axis mesh.

Mosaic lowers `remote_ref` and remote semaphores in one of two ways:

- **NVSHMEM** (multi-process, multi-node): enabled with
  `enable_nvshmem_lowering()`.
- **XLA collective metadata**: XLA allocates the buffers symmetrically and
  passes each kernel its peers' addresses. One NVLink domain only. This is the
  mode used here, because the installed XLA has no NVSHMEM (section 7).

Narrow rows (router weight and its gradient, one float per row) fall back to
`ragged_all_to_all`; they are about 4 bytes against roughly 14 KB of
activations per row.

## 4. Fused kernels (`kernels/fused_expert.py`)

Hopper (`wgmma`) kernels in Mosaic GPU, adapted from JAX's
`ragged_dot_mgpu.py` / `transposed_ragged_dot_mgpu.py` with the SM count passed
in (no device to query in deviceless compiles). Block sizes: `M=64`, `K=64`,
output tile `N = stripe` (128).

| kernel | computes | fusion |
|---|---|---|
| `gmm_swiglu` | `act = SwiGLU(x @ W_up)` | two accumulators (gate stripe, up stripe), SwiGLU in the epilogue; writes `h` only when the backward needs it |
| `gmm` | `y = act @ W_down`, `dx = dh @ W_up^T` | grouped GEMM; transposed B via `transpose_ref` |
| `gmm_dswiglu` | `dh = SwiGLU'(h) * (dy @ W_down^T)` | loads the matching `h` stripes by TMA in the epilogue; `d_act` never reaches HBM |
| `tgmm` | `dW = lhs^T @ rhs` per expert | weight gradients in float32; boundary rows of each group selected away in both operands |

Grouped GEMMs visit an M block once per group it overlaps
(`cdiv(M, 64) + G - 1` blocks) and store rows with a power-of-two ladder of TMA
copies, because TMA boxes are static but group boundaries are not.

`tgmm` selects away rows of neighbouring groups (and the never-written tail of
the slice) with `where`, not multiply-by-zero: those rows can be uninitialized
memory, and `0 * NaN = NaN`. This bug was found by the interpreter tests.

## 5. MaxText integration (maxtext branch `chunked-ep-moe`)

### 5.1 `moe_ep_impl=chunked`

- Config fields (`configs/types.py`, `configs/base.yml`): `moe_ep_impl`,
  `chunked_moe_transport`, `chunked_moe_expert_compute`,
  `chunked_moe_slice_size`, `chunked_moe_comm_dtype`,
  `chunked_moe_stripe_width`, `chunked_moe_num_sms`.
- `_validate_chunked_moe` requires `sparse_matmul=true`,
  `use_ring_of_experts=false`, tensor parallelism 1, SwiGLU, no MLP bias, not
  Llama 4.
- In `layers/moe.py`, `sparse_matmul`'s `shard_map` body (`_route_and_compute`)
  branches to `_chunked_ep_route_and_compute`, which:
  1. calls MaxText's `get_topk` (sigmoid scores, routed bias, routed scaling),
     and computes the load-balance loss and the bias-update counts exactly as
     `permute` does;
  2. stripes `w0` (gate) and `w1` (up) into `w_up`, casts weights to the compute
     dtype;
  3. calls `chunked_moe.chunked_moe` with `self._expert_parallelism_name` as
     the EP axis;
  4. returns the same tuple as the default path, with the overflow flag always
     `False`.
- The `shard_map` specs, the FSDP all-gather of expert weights at its entry,
  shared experts, attention and dense layers are unchanged.

### 5.2 Deviceless GPU compile

`compile_gpu_target_config` (an XLA `GpuTargetConfigProto`, e.g.
`h100_sxm.txtpb` from `xla/backends/gpu/target_config/specs/`) and
`compile_gpu_topology` (`<partitions>x<hosts>x<gpus>`) make `train_compile`
build compile-only CUDA devices instead of calling `jax.devices()`. The target
device count for batch sizing comes from the topology string. Compiles need
`--xla_gpu_autotune_level=0 --xla_gpu_enable_cub_radix_sort=false`, because
GEMM autotuning and the CUB sort rewrite both query a real GPU.

## 6. What is verified

| check | how | result |
|---|---|---|
| layer forward + 4 gradients vs dense reference | `test_moe.py`, 8 simulated CPU devices, emulated transport | match to 1e-4 across slice sizes 3/8/64, stripes 1/4, uniform to fully skewed routing |
| FP8 wire | `test_moe.py` | relative error < 8% (e4m3) |
| plan invariants | `test_plan.py` | every row delivered and returned once, no padding on the wire, no slice over S, same loop count on all ranks |
| each fused kernel | `test_fused_kernels.py`, Mosaic GPU interpreter | matches reference, incl. partial slices and an empty expert |
| MaxText training | DeepSeek-V3-tiny (5 layers), 4 steps on CPU | losses equal to MaxText's default MoE within bf16 rounding, at EP=1 and EP=4 |
| GPU compile, layer | `scripts/compile_gpu_aot.py`, 8 x H100 spec, DSv3 MoE sizes | compiles for ragged+reference, ragged+fused, put/signal+fused |
| GPU compile, model | MaxText `train_compile`, DeepSeek-V3 671B, 128 x H100, EP=8 x FSDP=16 | ragged+reference 79.6 GB/GPU; put/signal+fused 73.4 GB/GPU |

Not verified: anything at run time on a GPU (correctness of the put/signal
protocol and memory ordering, whether XLA keeps the aliased buffer in
symmetric memory without per-iteration copies, actual communication/compute
overlap, performance), and the layer with fused kernels inside the loop (the
interpreter hangs there; the halves are tested separately).

## 7. Limitations

**Environment-driven**

- **No NVSHMEM in this XLA.** The jaxlib 0.11.1 / jax-cuda12-plugin wheels have
  no NVSHMEM support compiled into XLA ("NVSHMEM is not supported in XLA"), so
  the transport was only compiled with the collective-metadata lowering, which
  is limited to one NVLink domain. jaxlib also does not register
  `--xla_gpu_experimental_enable_nvshmem` (only the plugin does), so neither
  `XLA_FLAGS` nor compile options can carry it. Needs a matched jaxlib/plugin
  built with NVSHMEM (e.g. NVIDIA's JAX containers).
- **No sm_100 NVSHMEM bitcode** in `nvidia-nvshmem-cu12` 3.7.2: NVSHMEM on
  Blackwell needs the CUDA 13 packages.
- **Deviceless topology has no slice structure**: compile-only GPU devices all
  report `slice_index` 0, so a deviceless compile treats all GPUs as one ICI
  domain; DCN vs NVLink placement cannot be expressed.
- **Transformer Engine not installed** (TE-JAX fails to build without NVTX
  headers), so MaxText attention was `dot_product` in the DSv3 compile.

**Design and implementation**

- **Fused kernels are Hopper-only** (`wgmma`), bf16/f16, and need `S` and `D`
  multiples of 64 and `F` a multiple of the stripe width. No Blackwell
  (`tcgen05`) variants.
- **Put/signal kernel is simple**: one row per block iteration, copies through
  registers, no TMA, no batching of small segments, every block signals every
  peer. Fine for correctness, likely not for peak NVLink bandwidth.
- **Overlap in the ragged path is left to XLA's scheduler.** Nothing forces the
  slice `j+1` dispatch to run concurrently with slice `j` compute, and the
  combine of slice `j` sits at the end of the loop body.
- **Send-order buffers are `T*K` rows.** `send_x`, `send_g` and the return
  buffers are full-size (e.g. 470 MB each for DSv3 at T=4096); only the
  receive side is bounded by `S`.
- **No expert tensor parallelism**: expert weights must be whole on each EP
  rank (MaxText's FSDP all-gather at the shard_map entry provides that).
- **MaxText weights are striped every step**: gate and up are stored
  separately, so `stripe_weights` copies a weight-sized tensor per layer in the
  forward and the backward.
- **Router weights always use `ragged_all_to_all`**, even with the put/signal
  transport.
- **FP8 is wire-only**: GEMMs run in bf16. No MXFP8 GEMMs, and the backward
  re-dispatches bf16-quantized rows rather than stashing FP8 activations.
- **MaxText validation excludes** `use_ring_of_experts`, MLP bias, Llama 4,
  non-SwiGLU activations, quantized (AQT/QWIX) expert weights; forced-routing
  padding (`-1` ids) is not handled.

## 8. Future work

Roughly in order of value.

1. **Run it on GPUs.** One H100 node is enough to validate the put/signal
   protocol (ready/done semaphores, signal release semantics), check
   for per-iteration copies into symmetric memory in the profile, and measure
   overlap. Use NVIDIA's JAX container for an XLA with NVSHMEM; then multi-node.
2. **Fused router.** Router GEMM with sigmoid/softmax, routed bias and top-k in
   the epilogue (the talk uses a bitonic top-k). Testable in the interpreter.
   Group-limited routing (DeepSeek's `n_routing_groups`) needs a second-stage
   reduction.
3. **Store weights striped in MaxText.** A config option to keep `wi` as one
   striped parameter (and convert checkpoints), removing the per-step stripe
   copy.
4. **Port the GEMM kernels to CuTe DSL** (`cutlass.jax.cutlass_call` exists in
   the installed wheel) for CUTLASS-quality Hopper and Blackwell GEMMs, warp
   specialization and TMA gather; keep the Pallas versions as the
   interpreter-tested reference.
5. **MXFP8 path.** MXFP8 grouped GEMMs, the fused dequantize-transpose-requantize
   for the weight-gradient GEMM (the talk's "MXFP8 transpose problem"), and
   stashing FP8 activations instead of recomputing from bf16.
6. **Better transport kernel.** TMA bulk copies for full row blocks, register
   copies only for tails; per-peer semaphores instead of all-to-all signalling;
   and pass a router weight column in the row payload so the narrow-row
   `ragged_all_to_all` disappears.
7. **Kernel-level communication fusion.** Let the grouped GEMM start on rows of
   a slice as each sender's `done` arrives (per-source semaphores), instead of
   after the whole slice: the talk's "fuse the communication into the kernel".
8. **Bound the send side.** Gather send rows from `x` per slice inside the
   transport kernel instead of materializing `send_x`, and accumulate weighted
   returned rows straight into `out[T, D]` instead of a `T*K` return buffer.
9. **Explicit overlap in the ragged path.** Software-pipeline the combine
   (combine `j-1` alongside compute `j`), and check the scheduler's choices in
   the dumped HLO.
10. **Expert waves / fixed-size token waves.** Cut the expert-sorted stream into
    fixed-size waves that may split an expert across waves (the idea discussed
    alongside the talk); `plan.py` already computes everything needed.
11. **Expert tensor parallelism** with a windowed grouped GEMM: a `ppermute` ring
    over the TP axis with a `ragged_dot` per arriving source block.
12. **Multi-slice deviceless compiles.** Give compile-only GPU devices a slice
    structure (NVLink domains) so DCN axes can be compiled and inspected.
13. **Debug the interpreter hang** so the layer with fused kernels can be
    numerically tested end to end on CPU.
