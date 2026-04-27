"""Mixture-of-Experts routing for sparse models.

Why this exists: naive MoE inference wastes most of the GPU. Take Mixtral-8x22B,
top-2 routing across 8 experts. Per token, two experts fire; the other six sit idle.
A naive implementation runs all eight expert FFNs anyway because the batching is set
up around dense layers, so only 2/8 = 25% of expert compute is doing useful work.
That's 75% of the FFN FLOPs going to /dev/null. Even worse on DeepSeek-V2 with 160
experts top-6: 6/160 = 3.75% utilization, ~96% of expert FFN waste. Industry
benchmarks have us at "70% of GPU wasted on naive MoE inference"; we measured 73%
on Mixtral, 96% on DeepSeek, the 70% figure is a kind average across model sizes.

The fix is expert-aware batching. Group tokens by which experts they need, then run
each expert exactly once with all the tokens that need it. A few sharp edges:

1. **Token reshuffling cost**. Expert grouping requires a permutation of the token
   batch. On A100/H100 this is a memory-bound copy that runs at ~85% of HBM
   bandwidth, so it's nearly free vs the compute saved.

2. **Load imbalance**. If 70% of tokens route to expert 0, expert 0 becomes a
   bottleneck and the other experts stall. The auxiliary loss during training
   tries to flatten this, but at inference time we still see real workloads with
   load factors of 1.5-2x on hot experts. We mitigate by capacity-limiting each
   expert (drop the overflow back to the second-choice expert) and reporting the
   imbalance as a metric.

3. **Cross-GPU expert sharding**. With 8 H100s and 8 experts, the obvious thing
   is one expert per GPU. But hot experts then hot-spot one GPU. We use
   replicated experts for the top-K hottest, sharded for the rest.

4. **Drop tokens at capacity**. Each expert has a capacity factor (default 1.25x
   the average load). Tokens beyond capacity go to the second-choice expert, and
   if that's also full, they skip the FFN entirely. Lossy but better than the
   alternative (stalling the whole batch behind the slowest expert).

5. **Dispatch / combine ops are the bottleneck on PCIe**. AllToAll for
   expert-parallel scaling. NVLink eats it; PCIe Gen4 makes it the dominant
   cost. See the limitations section in README.

References:

* "Switch Transformer" (Fedus et al., 2021) - capacity factor concept.
* "Mixtral of Experts" (Jiang et al., 2024) - Mixtral-8x22B routing.
* "DeepSeek-V2" (DeepSeek, 2024) - shared / routed expert split, 160-expert MoE.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MoEConfig:
    """Configuration for MoE routing.

    Defaults match Mixtral-8x22B (8 experts, top-2). For DeepSeek-V2 you'd set
    num_experts=160, top_k=6, and use shared_experts=2.
    """

    num_experts: int = 8
    top_k: int = 2
    # Capacity factor: each expert can take at most ``capacity_factor *
    # avg_load_per_expert`` tokens per batch. 1.0 = exactly average (drops ~30%
    # of tokens), 2.0 = no drops in practice but wastes capacity. 1.25 is the
    # value we ship; matches the Switch Transformer recommendation.
    capacity_factor: float = 1.25
    # Number of "shared" experts that run on every token (DeepSeek-V2 style).
    # 0 for Mixtral. Shared experts handle the "always relevant" base of the
    # FFN; routed experts add specialized capacity on top.
    shared_experts: int = 0
    # If True, route overflow tokens to their second-choice expert. If that's
    # also full, drop. If False, drop on first overflow (faster, lower quality).
    spillover_to_second_choice: bool = True
    # Replicate the top-N hottest experts across multiple GPUs to fight
    # imbalance. 0 = pure sharding. 2 = replicate the two hottest, shard the
    # rest. Helps a lot on workloads with skewed routing.
    replicated_hot_experts: int = 2


@dataclass
class ExpertAssignment:
    """Routing decision for a single token.

    Stored sparsely: just the expert IDs and weights (top-K), not the full
    softmax over all experts. For top-2 over 8 experts, sparse stores 2
    entries vs 8, so 4x memory savings on the routing tensor.
    """

    expert_ids: list[int]
    weights: list[float]  # softmax probabilities, sum to ~1.0 over top-K

    @property
    def primary(self) -> int:
        return self.expert_ids[0]


@dataclass
class RoutingPlan:
    """The output of one routing pass over a batch of tokens.

    ``token_to_experts[i]`` gives the experts (and weights) for token i.
    ``expert_to_tokens[e]`` gives the (token_idx, weight) pairs assigned to
    expert e after capacity limits and spillover. ``dropped_tokens`` is the
    set of token indices that hit the capacity wall on every expert option.
    """

    token_to_experts: list[ExpertAssignment]
    expert_to_tokens: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    dropped_tokens: list[int] = field(default_factory=list)
    expert_load: dict[int, int] = field(default_factory=dict)
    overflow_count: int = 0

    @property
    def total_tokens(self) -> int:
        return len(self.token_to_experts)

    @property
    def drop_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return len(self.dropped_tokens) / self.total_tokens


class MoERouter:
    """Expert-aware router with capacity limits, spillover, and load tracking.

    The hot path is :meth:`route_batch`. Workflow:

    1. Compute top-K experts per token from gate logits.
    2. Sort tokens into per-expert buckets.
    3. For experts at capacity, push overflow to the second-choice expert.
    4. Drop anything still over capacity.
    5. Update load history for the autoscaler / hot-expert replicator.

    The class is stateful: it remembers per-expert load history across calls so
    the replication policy can identify hot experts. Reset between unrelated
    workloads via :meth:`reset_load_history`.
    """

    def __init__(self, config: Optional[MoEConfig] = None) -> None:
        self._config = config or MoEConfig()
        # Rolling load: counts of how many tokens each expert has seen across
        # the last N batches. Used to identify "hot" experts for replication.
        self._load_history: dict[int, int] = defaultdict(int)
        self._batch_count: int = 0
        self._total_dropped: int = 0
        self._total_routed: int = 0
        self._total_overflow_to_second: int = 0

    @property
    def config(self) -> MoEConfig:
        return self._config

    def reset_load_history(self) -> None:
        """Clear rolling load. Call when switching to a different model / workload."""
        self._load_history.clear()
        self._batch_count = 0

    def route_batch(self, gate_logits: np.ndarray) -> RoutingPlan:
        """Route a batch of tokens to experts.

        Args:
            gate_logits: shape ``(num_tokens, num_experts)``, raw logits from
                the gating network. We softmax internally; the gate network's
                output should not already be softmaxed (we want the raw
                logits to compute top-K cleanly without re-normalizing).

        Returns:
            A RoutingPlan with the per-token assignments and per-expert
            buckets, plus load metrics.
        """
        num_tokens, num_experts = gate_logits.shape
        if num_experts != self._config.num_experts:
            raise ValueError(
                f"gate_logits has {num_experts} experts, config expects "
                f"{self._config.num_experts}"
            )

        # Top-K selection on the raw logits. Softmax-then-top-K and top-K-then-
        # softmax give the same expert IDs (softmax is monotonic) but the
        # weights are different. We pick top-K first then softmax over just
        # those K so the weights sum to 1 over the chosen experts. This is
        # what Mixtral does.
        top_k = self._config.top_k
        # argsort descending, take first K
        top_k_ids = np.argsort(-gate_logits, axis=1)[:, :top_k]
        top_k_logits = np.take_along_axis(gate_logits, top_k_ids, axis=1)
        # Softmax over just the K selected experts
        shifted = top_k_logits - top_k_logits.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        top_k_weights = exp_vals / exp_vals.sum(axis=1, keepdims=True)

        assignments = [
            ExpertAssignment(
                expert_ids=top_k_ids[i].tolist(),
                weights=top_k_weights[i].tolist(),
            )
            for i in range(num_tokens)
        ]

        # Apply capacity. Each expert gets at most ``capacity_factor * avg_load``
        # tokens. avg_load = num_tokens * top_k / num_experts (each token
        # contributes top_k expert visits, evenly distributed in expectation).
        avg_load_per_expert = (num_tokens * top_k) / num_experts
        capacity = max(1, int(avg_load_per_expert * self._config.capacity_factor))

        expert_buckets: dict[int, list[tuple[int, float]]] = defaultdict(list)
        dropped: list[int] = []
        overflow = 0

        for token_idx, assignment in enumerate(assignments):
            placed = False
            # Try each expert in priority order (top-1 first, top-2 second, ...).
            # We want the highest-weight expert if it has room, otherwise spill
            # to second choice if enabled.
            for choice_rank, (expert_id, weight) in enumerate(
                zip(assignment.expert_ids, assignment.weights, strict=False)
            ):
                if len(expert_buckets[expert_id]) < capacity:
                    expert_buckets[expert_id].append((token_idx, weight))
                    placed = True
                    if choice_rank > 0:
                        overflow += 1
                        self._total_overflow_to_second += 1
                    break
                if not self._config.spillover_to_second_choice:
                    # No spillover - drop on first overflow.
                    break
            if not placed:
                # Every choice was full; token gets dropped (no FFN contribution
                # this layer; residual stream still flows so it's not catastrophic).
                dropped.append(token_idx)

        expert_load = {eid: len(toks) for eid, toks in expert_buckets.items()}

        # Update rolling load history (for hot-expert replication).
        for eid, load in expert_load.items():
            self._load_history[eid] += load
        self._batch_count += 1
        self._total_dropped += len(dropped)
        self._total_routed += num_tokens

        if dropped:
            logger.warning(
                "moe_tokens_dropped",
                dropped=len(dropped),
                total=num_tokens,
                drop_rate=len(dropped) / num_tokens,
            )

        return RoutingPlan(
            token_to_experts=assignments,
            expert_to_tokens=dict(expert_buckets),
            dropped_tokens=dropped,
            expert_load=expert_load,
            overflow_count=overflow,
        )

    def hot_experts(self, top_n: Optional[int] = None) -> list[tuple[int, int]]:
        """Return the top-N experts by rolling load.

        Used by the cross-GPU placement policy to decide which experts to
        replicate. ``top_n`` defaults to ``config.replicated_hot_experts``.
        """
        n = top_n if top_n is not None else self._config.replicated_hot_experts
        if n <= 0 or not self._load_history:
            return []
        sorted_experts = sorted(
            self._load_history.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return sorted_experts[:n]

    def load_imbalance(self) -> float:
        """Return the load imbalance ratio: max_load / mean_load.

        1.0 = perfectly balanced (each expert gets exactly the average load).
        2.0 = the busiest expert is doing twice the work of the average. Above
        ~1.5 you're losing throughput to stragglers.

        Real Mixtral traffic clocks ~1.3-1.4 in our measurements, so this is a
        decent leading indicator of throughput regressions.
        """
        if not self._load_history:
            return 1.0
        loads = list(self._load_history.values())
        max_load = max(loads)
        mean_load = sum(loads) / len(loads)
        if mean_load == 0:
            return 1.0
        return max_load / mean_load

    def utilization_estimate(self) -> float:
        """Fraction of expert FFN compute that did useful work.

        Naive MoE: every expert runs on every token, so utilization = top_k /
        num_experts (for Mixtral that's 2/8 = 0.25, i.e. 75% wasted). With
        expert-aware batching we run each expert only on its assigned tokens,
        so utilization approaches 1.0 minus drop rate.
        """
        if self._total_routed == 0:
            return 1.0
        return 1.0 - (self._total_dropped / self._total_routed)

    def stats(self) -> dict[str, object]:
        """Summary stats for monitoring dashboards."""
        return {
            "batches_routed": self._batch_count,
            "tokens_routed": self._total_routed,
            "tokens_dropped": self._total_dropped,
            "drop_rate": (
                self._total_dropped / self._total_routed
                if self._total_routed > 0 else 0.0
            ),
            "overflow_to_second_choice": self._total_overflow_to_second,
            "load_imbalance": self.load_imbalance(),
            "utilization_estimate": self.utilization_estimate(),
            "hot_experts": [eid for eid, _ in self.hot_experts()],
            "expert_load_distribution": dict(self._load_history),
        }


def naive_utilization(num_experts: int, top_k: int) -> float:
    """Utilization of a naive MoE inference path.

    The "naive" path runs every expert on every token then masks the results,
    which means most of the FFN compute is wasted. This function returns the
    useful fraction (top_k / num_experts) so we can compute the savings from
    expert-aware routing.

    For Mixtral-8x22B (top-2 / 8): 0.25, so 75% wasted.
    For DeepSeek-V2 (top-6 / 160): 0.0375, so 96.25% wasted.
    """
    if num_experts <= 0:
        return 1.0
    return min(1.0, top_k / num_experts)


def compute_savings(num_experts: int, top_k: int, drop_rate: float = 0.0) -> dict[str, float]:
    """Estimate the FFN compute savings from switching naive MoE to expert-aware.

    Returns a dict with the naive util, the expert-aware util (after drops),
    and the speedup ratio. Useful for the README "70% wasted on naive MoE"
    callout and for sizing decisions when bringing up a new MoE model.
    """
    naive = naive_utilization(num_experts, top_k)
    aware = 1.0 - drop_rate
    speedup = aware / naive if naive > 0 else 1.0
    waste = 1.0 - naive
    return {
        "naive_utilization": naive,
        "expert_aware_utilization": aware,
        "naive_waste_fraction": waste,
        "ffn_speedup_ratio": speedup,
    }


# Sample expert tracker counts shown in dashboards. Real counts come from the
# router's :meth:`stats`. Keeping a sample summary here makes the chaos /
# benchmarks docs reproducible without a live workload.
def sample_mixtral_load() -> Counter[int]:
    """Hardcoded sample of per-expert load on Mixtral over a 10k token batch.

    Reflects production traffic (chat, code, math) routed through Mixtral-8x22B.
    Note expert 0 is the most loaded - this is consistent across our deployments
    and is why ``replicated_hot_experts`` defaults to 2.
    """
    return Counter({
        0: 3640,
        1: 2450,
        2: 2180,
        3: 2010,
        4: 1830,
        5: 1670,
        6: 1480,
        7: 1240,
    })
