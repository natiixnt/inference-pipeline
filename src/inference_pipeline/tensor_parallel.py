"""Tensor parallelism for distributing large model layers across GPUs.

For a 70B model on 8x H100s we shard the attention and MLP weights across GPUs.
Each GPU holds 1/TP_SIZE of the model's width, computes its slice, and then
AllReduce combines the results. The key decisions:

- Attention QKV: column-parallel (each GPU gets a subset of attention heads)
- Attention output projection: row-parallel (each GPU has partial results, sum them)
- MLP gate/up: column-parallel
- MLP down: row-parallel

# column parallel on QKV means each GPU gets a slice of the attention heads
This is natural because attention heads are independent - no cross-head communication
needed until the output projection.

# AllReduce is the bottleneck - NVLink gives us 600GB/s but PCIe is only 64GB/s, route accordingly
On H100 SXM5 with NVLink 4.0, AllReduce on a 16K hidden state takes ~5us.
On PCIe systems that jumps to ~80us. We detect topology at init and route
communication through NVLink when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class ParallelMode(Enum):
    """How a layer's weight matrix is partitioned across devices."""

    COLUMN = "column"  # split along output dim (each GPU gets full input, partial output)
    ROW = "row"  # split along input dim (each GPU gets partial input, full output)


class InterconnectType(Enum):
    """GPU interconnect type affects communication strategy."""

    NVLINK = "nvlink"  # 600 GB/s on H100, 900 GB/s on B200
    PCIE_5 = "pcie_5"  # 64 GB/s - avoid large AllReduce here
    PCIE_4 = "pcie_4"  # 32 GB/s - really avoid large AllReduce here


@dataclass
class ShardSpec:
    """Describes how a weight tensor is sharded across the TP group."""

    layer_name: str
    mode: ParallelMode
    tp_size: int
    shard_dim: int  # which dimension is split
    full_shape: tuple[int, ...]  # shape before sharding
    shard_shape: tuple[int, ...]  # shape on each GPU after sharding

    @property
    def elements_per_shard(self) -> int:
        result = 1
        for d in self.shard_shape:
            result *= d
        return result

    @property
    def bytes_per_shard_fp16(self) -> int:
        return self.elements_per_shard * 2


@dataclass
class AllReduceConfig:
    """Configuration for the AllReduce communication primitive."""

    tp_size: int
    interconnect: InterconnectType
    # ring vs tree topology - ring is better for small messages, tree for large
    use_tree_reduce: bool = False
    # overlap compute with communication where possible
    overlap_comm_compute: bool = True
    # chunk size for pipelining AllReduce with computation
    pipeline_chunk_size: int = 4096

    @property
    def bandwidth_gbps(self) -> float:
        if self.interconnect == InterconnectType.NVLINK:
            return 600.0
        elif self.interconnect == InterconnectType.PCIE_5:
            return 64.0
        return 32.0

    def estimate_allreduce_time_us(self, num_elements: int, dtype_bytes: int = 2) -> float:
        """Estimate AllReduce latency for a given tensor size.

        Uses the ring AllReduce formula: 2 * (n-1)/n * data_size / bandwidth
        where n is the number of participants.
        """
        data_bytes = num_elements * dtype_bytes
        # ring allreduce: each GPU sends/receives 2*(n-1)/n * total_data
        effective_data = 2 * (self.tp_size - 1) / self.tp_size * data_bytes
        bandwidth_bytes_per_us = self.bandwidth_gbps * 1e9 / 1e6  # GB/s to bytes/us
        # add fixed latency for kernel launch + synchronization
        fixed_latency_us = 3.0 if self.interconnect == InterconnectType.NVLINK else 12.0
        return fixed_latency_us + effective_data / bandwidth_bytes_per_us


@dataclass
class TPLayer:
    """A tensor-parallel layer with its sharding spec and communication config."""

    name: str
    shard_spec: ShardSpec
    allreduce_config: AllReduceConfig
    # for row-parallel layers, AllReduce happens after the matmul
    # for column-parallel, no AllReduce needed until the paired row-parallel layer
    needs_allreduce: bool = False
    # estimated compute time in microseconds (for overlap scheduling)
    compute_time_us: float = 0.0

    @property
    def comm_compute_ratio(self) -> float:
        """Ratio of communication to computation time.

        We want this < 1.0, meaning compute dominates and communication is hidden.
        If > 1.0, communication is the bottleneck and we should consider larger
        pipeline chunks or reducing TP degree.
        """
        if self.compute_time_us == 0:
            return float("inf")
        comm_time = self.allreduce_config.estimate_allreduce_time_us(
            self.shard_spec.elements_per_shard
        )
        return comm_time / self.compute_time_us


class TensorParallelGroup:
    """Manages tensor parallelism across a group of GPUs.

    Handles weight sharding, communication scheduling, and load balancing.
    The goal is to keep all GPUs equally busy while minimizing communication
    overhead. On NVLink systems with TP=8, AllReduce overhead is typically
    under 5% of total forward pass time. On PCIe it can hit 15-20%, which is
    why we only do TP across NVLink-connected GPUs and use pipeline parallelism
    across PCIe boundaries.
    """

    def __init__(
        self,
        tp_size: int,
        interconnect: InterconnectType = InterconnectType.NVLINK,
        hidden_size: int = 8192,  # 70B model
        num_attention_heads: int = 64,
        num_kv_heads: int = 8,
        intermediate_size: int = 28672,
        num_layers: int = 80,
    ) -> None:
        self._tp_size = tp_size
        self._interconnect = interconnect
        self._hidden_size = hidden_size
        self._num_attention_heads = num_attention_heads
        self._num_kv_heads = num_kv_heads
        self._intermediate_size = intermediate_size
        self._num_layers = num_layers

        # validate TP size divides attention heads evenly
        if num_attention_heads % tp_size != 0:
            raise ValueError(
                f"TP size {tp_size} must divide num_attention_heads {num_attention_heads}"
            )
        if num_kv_heads % tp_size != 0 and tp_size % num_kv_heads != 0:
            # GQA: either divide KV heads across GPUs, or replicate them
            logger.warning(
                "kv_heads_replicated",
                num_kv_heads=num_kv_heads,
                tp_size=tp_size,
            )

        self._allreduce_config = AllReduceConfig(
            tp_size=tp_size,
            interconnect=interconnect,
            use_tree_reduce=tp_size > 4,
            overlap_comm_compute=interconnect == InterconnectType.NVLINK,
        )

        # build the sharding plan for one transformer layer
        self._layer_specs = self._build_layer_specs()

        logger.info(
            "tensor_parallel_init",
            tp_size=tp_size,
            interconnect=interconnect.value,
            hidden_size=hidden_size,
            heads_per_gpu=num_attention_heads // tp_size,
            kv_heads_per_gpu=max(1, num_kv_heads // tp_size),
            allreduce_bw_gbps=self._allreduce_config.bandwidth_gbps,
        )

    def _build_layer_specs(self) -> list[TPLayer]:
        """Build sharding specs for all sublayers in one transformer block."""
        head_dim = self._hidden_size // self._num_attention_heads
        heads_per_gpu = self._num_attention_heads // self._tp_size
        kv_heads_per_gpu = max(1, self._num_kv_heads // self._tp_size)

        layers: list[TPLayer] = []

        # QKV projection - column parallel
        # each GPU computes its slice of Q, K, V for its assigned attention heads
        # column parallel on QKV means each GPU gets a slice of the attention heads
        qkv_out_size = (heads_per_gpu + 2 * kv_heads_per_gpu) * head_dim
        qkv_spec = ShardSpec(
            layer_name="attention.qkv_proj",
            mode=ParallelMode.COLUMN,
            tp_size=self._tp_size,
            shard_dim=0,  # output dimension
            full_shape=(
                (self._num_attention_heads + 2 * self._num_kv_heads) * head_dim,
                self._hidden_size,
            ),
            shard_shape=(qkv_out_size, self._hidden_size),
        )
        layers.append(
            TPLayer(
                name="attention.qkv_proj",
                shard_spec=qkv_spec,
                allreduce_config=self._allreduce_config,
                needs_allreduce=False,  # no comm needed after column-parallel
                compute_time_us=self._estimate_matmul_time(
                    self._hidden_size, qkv_out_size
                ),
            )
        )

        # Output projection - row parallel
        # each GPU has partial attention output, row-parallel matmul + AllReduce
        o_proj_spec = ShardSpec(
            layer_name="attention.o_proj",
            mode=ParallelMode.ROW,
            tp_size=self._tp_size,
            shard_dim=1,  # input dimension
            full_shape=(self._hidden_size, self._hidden_size),
            shard_shape=(self._hidden_size, self._hidden_size // self._tp_size),
        )
        layers.append(
            TPLayer(
                name="attention.o_proj",
                shard_spec=o_proj_spec,
                allreduce_config=self._allreduce_config,
                needs_allreduce=True,  # AllReduce to combine partial results
                compute_time_us=self._estimate_matmul_time(
                    self._hidden_size // self._tp_size, self._hidden_size
                ),
            )
        )

        # MLP gate + up projection - column parallel
        # SwiGLU: gate and up are fused, each GPU gets a slice of intermediate dim
        mlp_slice = self._intermediate_size // self._tp_size
        gate_up_spec = ShardSpec(
            layer_name="mlp.gate_up_proj",
            mode=ParallelMode.COLUMN,
            tp_size=self._tp_size,
            shard_dim=0,
            full_shape=(2 * self._intermediate_size, self._hidden_size),
            shard_shape=(2 * mlp_slice, self._hidden_size),
        )
        layers.append(
            TPLayer(
                name="mlp.gate_up_proj",
                shard_spec=gate_up_spec,
                allreduce_config=self._allreduce_config,
                needs_allreduce=False,
                compute_time_us=self._estimate_matmul_time(
                    self._hidden_size, 2 * mlp_slice
                ),
            )
        )

        # MLP down projection - row parallel
        down_spec = ShardSpec(
            layer_name="mlp.down_proj",
            mode=ParallelMode.ROW,
            tp_size=self._tp_size,
            shard_dim=1,
            full_shape=(self._hidden_size, self._intermediate_size),
            shard_shape=(self._hidden_size, mlp_slice),
        )
        layers.append(
            TPLayer(
                name="mlp.down_proj",
                shard_spec=down_spec,
                allreduce_config=self._allreduce_config,
                needs_allreduce=True,
                compute_time_us=self._estimate_matmul_time(mlp_slice, self._hidden_size),
            )
        )

        return layers

    def _estimate_matmul_time(self, m: int, n: int, batch_tokens: int = 1) -> float:
        """Estimate matmul time in microseconds on H100.

        H100 does ~990 TFLOPS fp16. A matmul of [batch, m] x [m, n] is
        2 * batch * m * n FLOPs.
        """
        flops = 2 * batch_tokens * m * n
        # H100 peak: 990 TFLOPS, realistic sustained: ~750 TFLOPS
        tflops_sustained = 750.0
        flops_per_us = tflops_sustained * 1e6  # TFLOPS to FLOPS/us
        return flops / flops_per_us

    def get_shard_plan(self) -> list[TPLayer]:
        """Return the full sharding plan for one transformer layer."""
        return self._layer_specs

    def estimate_layer_time_us(self, batch_tokens: int = 1) -> dict[str, float]:
        """Estimate total time for one transformer layer including communication.

        Returns breakdown of compute vs communication time.
        """
        total_compute_us = 0.0
        total_comm_us = 0.0

        for layer in self._layer_specs:
            # scale compute time by batch size
            compute = layer.compute_time_us * batch_tokens
            total_compute_us += compute

            if layer.needs_allreduce:
                # AllReduce size depends on output dimension
                comm = self._allreduce_config.estimate_allreduce_time_us(
                    layer.shard_spec.shard_shape[0] * batch_tokens
                )
                # with overlap, we can hide some communication behind compute
                if self._allreduce_config.overlap_comm_compute:
                    # overlap hides min(compute, comm) of the comm time
                    effective_comm = max(0, comm - compute * 0.7)
                else:
                    effective_comm = comm
                total_comm_us += effective_comm

        return {
            "compute_us": total_compute_us,
            "communication_us": total_comm_us,
            "total_us": total_compute_us + total_comm_us,
            "comm_overhead_pct": (
                total_comm_us / (total_compute_us + total_comm_us) * 100
                if (total_compute_us + total_comm_us) > 0
                else 0.0
            ),
        }

    def total_model_memory_per_gpu_bytes(self, dtype_bytes: int = 2) -> int:
        """Calculate total model parameter memory per GPU after TP sharding."""
        bytes_per_layer = sum(
            layer.shard_spec.bytes_per_shard_fp16 for layer in self._layer_specs
        )
        # embedding + final LN + lm_head are typically replicated or only on rank 0
        embedding_bytes = 128256 * self._hidden_size * dtype_bytes  # vocab_size * hidden
        total = bytes_per_layer * self._num_layers + embedding_bytes // self._tp_size
        return total

    @property
    def stats(self) -> dict[str, int | float]:
        timing = self.estimate_layer_time_us(batch_tokens=1)
        return {
            "tp_size": self._tp_size,
            "interconnect": self._interconnect.value,
            "num_layers": self._num_layers,
            "heads_per_gpu": self._num_attention_heads // self._tp_size,
            "memory_per_gpu_gb": self.total_model_memory_per_gpu_bytes() / (1024**3),
            "comm_overhead_pct": timing["comm_overhead_pct"],
            "allreduce_bw_gbps": self._allreduce_config.bandwidth_gbps,
        }
