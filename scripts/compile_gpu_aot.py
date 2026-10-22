"""Compile the chunked MoE layer (forward + backward) for GPUs without having one.

JAX can build compile-only CUDA devices from an XLA GpuTargetConfigProto
(xla/backends/gpu/target_config/specs/*.txtpb), so the whole program, including
Mosaic GPU kernels, is lowered and compiled by XLA:GPU exactly as it would be on
the real machine; it just cannot run.

Example:
  python scripts/compile_gpu_aot.py --spec h100_sxm.txtpb --topology 1x1x8 \\
      --transport nvshmem --expert-compute fused
"""

import argparse
import functools
import os
import re
import time

import jax
import jax.numpy as jnp
from jax.experimental import topologies
from jax.sharding import NamedSharding, PartitionSpec as P

from chunked_moe import ChunkedMoEConfig, chunked_moe


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--spec", required=True, help="GpuTargetConfigProto text file, e.g. h100_sxm.txtpb")
  ap.add_argument("--topology", default="1x1x8", help="slices x hosts x devices-per-host")
  ap.add_argument("--transport", default="ragged_a2a", choices=["ragged_a2a", "nvshmem"])
  ap.add_argument("--expert-compute", default="reference", choices=["reference", "fused"])
  ap.add_argument("--tokens-per-rank", type=int, default=4096)
  ap.add_argument("--hidden", type=int, default=7168)
  ap.add_argument("--ffn", type=int, default=2048)
  ap.add_argument("--experts", type=int, default=256)
  ap.add_argument("--top-k", type=int, default=8)
  ap.add_argument("--slice-size", type=int, default=4096)
  ap.add_argument("--comm-dtype", default=None)
  ap.add_argument(
      "--peer-lowering",
      default="xla",
      choices=["xla", "nvshmem"],
      help="How Mosaic GPU lowers the transport's remote refs: XLA collective metadata "
      "(peer pointers inside one NVLink domain) or NVSHMEM (needs an XLA built with NVSHMEM)",
  )
  ap.add_argument("--dump-hlo", default=None, help="write the optimized HLO text here")
  args = ap.parse_args()

  # Deviceless compiles must not touch a GPU: GEMM autotuning benchmarks on the
  # device, and the CUB radix-sort rewrite asks the CUDA runtime for scratch
  # sizes. Neither flag is needed (or wanted) on a real machine.
  flags = ["--xla_gpu_autotune_level=0", "--xla_gpu_enable_cub_radix_sort=false"]
  os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", "")] + flags)
  compiler_options = {}

  topo = topologies.get_topology_desc(platform="cuda", target_config=open(args.spec).read(), topology=args.topology)
  if args.transport == "nvshmem" and args.peer_lowering == "nvshmem":
    from chunked_moe.transport import nvshmem  # pylint: disable=import-outside-toplevel

    compiler_options.update(nvshmem.enable_nvshmem_lowering())
  mesh = topologies.make_mesh(topo, (len(topo.devices),), ("ep",))
  ep = mesh.size
  cfg = ChunkedMoEConfig(
      num_experts=args.experts,
      top_k=args.top_k,
      slice_size=args.slice_size,
      transport=args.transport,
      expert_compute=args.expert_compute,
      comm_dtype=args.comm_dtype,
      num_sms=topo.devices[0].core_count if hasattr(topo.devices[0], "core_count") else 132,
  )
  print(f"{ep} x {topo.devices[0].device_kind}; {cfg}")

  spec = P("ep")
  layer = jax.shard_map(functools.partial(chunked_moe, cfg=cfg, axis_name="ep"), mesh=mesh, in_specs=(spec,) * 5, out_specs=spec)

  def train_step(x, ids, w, w_up, w_down, g):
    loss = lambda x, w, u, d: jnp.sum(layer(x, ids, w, u, d).astype(jnp.float32) * g)
    return jax.grad(loss, argnums=(0, 1, 2, 3))(x, w, w_up, w_down)

  sds = lambda shape, dt: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, spec))
  n, d, f, e, k = ep * args.tokens_per_rank, args.hidden, args.ffn, args.experts, args.top_k
  inputs = (
      sds((n, d), jnp.bfloat16),
      sds((n, k), jnp.int32),
      sds((n, k), jnp.float32),
      sds((e, d, 2 * f), jnp.bfloat16),
      sds((e, f, d), jnp.bfloat16),
      sds((n, d), jnp.bfloat16),
  )
  t = time.time()
  compiled = jax.jit(train_step).lower(*inputs).compile(compiler_options)
  print(f"compiled forward+backward in {time.time() - t:.1f}s")
  hlo = compiled.as_text()
  kinds = {
      "ragged-all-to-all": len(re.findall(r"= \S+ ragged-all-to-all", hlo)),
      "mosaic_gpu kernels": len(re.findall(r'custom_call_target="mosaic_gpu', hlo)),
      "while loops": len(re.findall(r" while\(", hlo)),
  }
  print("ops in optimized HLO:", kinds)
  print(compiled.memory_analysis())
  if args.dump_hlo:
    with open(args.dump_hlo, "w") as fh:
      fh.write(hlo)


if __name__ == "__main__":
  main()
