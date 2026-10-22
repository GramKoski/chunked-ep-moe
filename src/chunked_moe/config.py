"""Static configuration for the chunked expert-parallel MoE layer."""

import dataclasses
from typing import Literal

Transport = Literal["ragged_a2a", "nvshmem", "emulated"]
ExpertCompute = Literal["reference", "fused"]


@dataclasses.dataclass(frozen=True)
class ChunkedMoEConfig:
  """Hyperparameters that are fixed at trace time.

  Attributes:
    num_experts: Global number of routed experts E.
    top_k: Experts per token K.
    slice_size: Rows S in one dispatch slice. Each receiver's packed token
      stream is cut into slices of S rows; empty tail slices are skipped, so at
      most S - 1 rows of padding are ever communicated or computed.
    transport: How slice rows move between expert-parallel ranks.
      "ragged_a2a" uses `jax.lax.ragged_all_to_all`, "nvshmem" uses one-sided
      puts from a Mosaic GPU kernel, "emulated" reproduces ragged_a2a semantics
      with a dense all_to_all so the algorithm can be tested on CPU.
    expert_compute: "reference" runs the per-slice expert MLP with
      `jax.lax.ragged_dot`; "fused" uses the Mosaic GPU kernels.
    comm_dtype: Dtype of token rows on the wire. float8_e4m3fn quantizes with
      per-row scales before sending and dequantizes on arrival.
    accum_dtype: Dtype of the weight-gradient accumulators in the backward loop.
    num_sms: SM count used to size Mosaic GPU grids. Passed explicitly because
      deviceless compilation has no device to query.
  """

  num_experts: int
  top_k: int
  slice_size: int = 1024
  transport: Transport = "ragged_a2a"
  expert_compute: ExpertCompute = "reference"
  comm_dtype: str | None = None
  accum_dtype: str = "float32"
  num_sms: int = 132

  def __post_init__(self):
    if self.slice_size <= 0:
      raise ValueError(f"slice_size must be positive, got {self.slice_size}")
    if self.top_k <= 0 or self.top_k > self.num_experts:
      raise ValueError(f"top_k={self.top_k} must be in [1, num_experts={self.num_experts}]")
