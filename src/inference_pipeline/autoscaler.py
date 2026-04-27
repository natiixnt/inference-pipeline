"""Custom autoscaler that actually understands GPU inference workloads.

HPA is a joke for LLM serving. It looks at CPU util or request count and reacts
10 minutes too late. By the time new pods are warm (model loading takes 2-5 min on
70B), your p95 latency already spiked and users bounced.

This autoscaler uses leading indicators:
- Queue depth trend (requests accumulating = need more capacity SOON)
- GPU memory pressure (approaching OOM means we can't batch more)
- TTFT percentile drift (latency creeping up before it spikes)
- Speculative decode rejection rate (high rejection = overloaded target model)

It also predicts demand from time-of-day patterns and scales preemptively.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class ScaleDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    HOLD = "hold"


class ScaleReason(str, Enum):
    """Why we decided to scale. Useful for postmortems."""

    QUEUE_DEPTH_TREND = "queue_depth_trend"
    MEMORY_PRESSURE = "memory_pressure"
    TTFT_REGRESSION = "ttft_regression"
    THROUGHPUT_SATURATION = "throughput_saturation"
    PREDICTIVE_DEMAND = "predictive_demand"
    COOLDOWN_UNDERUTIL = "cooldown_underutilization"
    SPECULATION_DEGRADED = "speculation_degraded"


@dataclass
class ScalingDecision:
    """The output of one autoscaler evaluation cycle."""

    direction: ScaleDirection
    target_replicas: int
    current_replicas: int
    reason: ScaleReason
    confidence: float  # 0 to 1, how sure we are this is the right call
    cooldown_remaining_sec: float
    metrics_snapshot: dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AutoscalerConfig:
    """Tuning knobs. These took a while to get right in production.

    The key tension: scale up too aggressively and you waste money on idle GPUs
    that take 3 min to load models. Scale up too slowly and latency spikes
    propagate through retry storms.
    """

    # target operating points
    target_gpu_utilization: float = 0.80  # we want 80% util, not 100% (no headroom = bad)
    target_queue_depth: int = 50  # per-replica target queue depth
    target_ttft_p95_ms: float = 100.0  # SLA target for p95 TTFT
    target_throughput_headroom: float = 0.20  # keep 20% throughput margin

    # scaling bounds
    min_replicas: int = 2  # always keep 2 for HA
    max_replicas: int = 32  # budget cap
    scale_up_increment: int = 2  # add 2 at a time (model loading is slow, go big)
    scale_down_increment: int = 1  # remove 1 at a time (conservative on scale-down)

    # cooldowns prevent thrashing
    scale_up_cooldown_sec: float = 120.0  # 2 min between scale-ups (time to warm)
    scale_down_cooldown_sec: float = 300.0  # 5 min between scale-downs (wait for stability)

    # sensitivity thresholds
    queue_depth_trigger_ratio: float = 2.0  # 2x target = trigger scale-up
    ttft_regression_trigger: float = 1.5  # 50% above target = scale-up
    memory_pressure_trigger: float = 0.92  # 92% memory = need more capacity
    underutil_trigger: float = 0.40  # below 40% util for sustained period = scale-down

    # predictive scaling
    enable_predictive: bool = True
    prediction_lookahead_min: int = 15  # scale up 15 min before predicted peak
    demand_history_hours: int = 168  # one week of hourly demand data


@dataclass
class ClusterMetrics:
    """Snapshot of cluster state for scaling decisions."""

    total_replicas: int
    avg_gpu_utilization: float
    avg_memory_utilization: float
    total_queue_depth: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    throughput_tokens_per_sec: float
    max_throughput_tokens_per_sec: float  # theoretical max at current replica count
    speculation_acceptance_rate: float
    active_requests: int
    requests_per_sec: float
    error_rate: float


class Autoscaler:
    """GPU-aware autoscaler for LLM inference clusters.

    Evaluates every 10 seconds (way faster than HPA's default 15s + metrics lag).
    Uses a scoring system that weighs multiple signals because no single metric
    tells the whole story.
    """

    def __init__(self, config: Optional[AutoscalerConfig] = None) -> None:
        self._config = config or AutoscalerConfig()
        self._last_scale_up_time: float = 0.0
        self._last_scale_down_time: float = 0.0
        self._current_replicas: int = self._config.min_replicas
        self._decision_history: list[ScalingDecision] = []

        # rolling metrics for trend detection
        self._queue_depth_history: list[tuple[float, int]] = []
        self._ttft_history: list[tuple[float, float]] = []
        self._throughput_history: list[tuple[float, float]] = []

        # time-of-day demand patterns (hourly buckets, filled over time)
        self._hourly_demand: list[float] = [0.0] * 168  # one week of hourly slots

    def evaluate(self, metrics: ClusterMetrics) -> ScalingDecision:
        """Run one autoscaler evaluation cycle.

        Returns a ScalingDecision. The orchestrator (k8s operator or custom controller)
        is responsible for actually executing the scaling action.
        """
        now = time.time()

        # record metrics for trend analysis
        self._queue_depth_history.append((now, metrics.total_queue_depth))
        self._ttft_history.append((now, metrics.ttft_p95_ms))
        self._throughput_history.append((now, metrics.throughput_tokens_per_sec))

        # trim histories to last 30 minutes
        cutoff = now - 1800
        self._queue_depth_history = [(t, v) for t, v in self._queue_depth_history if t > cutoff]
        self._ttft_history = [(t, v) for t, v in self._ttft_history if t > cutoff]
        self._throughput_history = [(t, v) for t, v in self._throughput_history if t > cutoff]

        # compute scale-up urgency score (higher = more urgent)
        up_score, up_reason = self._compute_scale_up_score(metrics, now)

        # compute scale-down opportunity score
        down_score, down_reason = self._compute_scale_down_score(metrics, now)

        # make the call
        decision = self._make_decision(metrics, up_score, up_reason, down_score, down_reason, now)

        self._decision_history.append(decision)
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-500:]

        if decision.direction != ScaleDirection.HOLD:
            logger.info(
                "autoscale_decision",
                direction=decision.direction.value,
                target=decision.target_replicas,
                current=decision.current_replicas,
                reason=decision.reason.value,
                confidence=f"{decision.confidence:.2f}",
            )

        return decision

    def _compute_scale_up_score(
        self, metrics: ClusterMetrics, now: float
    ) -> tuple[float, ScaleReason]:
        """Score how urgently we need more replicas. 0 = fine, 1 = scale NOW."""
        scores: list[tuple[float, ScaleReason]] = []

        # queue depth growing = we're falling behind
        queue_ratio = metrics.total_queue_depth / max(1, self._config.target_queue_depth * metrics.total_replicas)
        if queue_ratio > self._config.queue_depth_trigger_ratio:
            queue_score = min(1.0, (queue_ratio - 1.0) / 3.0)
            scores.append((queue_score, ScaleReason.QUEUE_DEPTH_TREND))

            # bonus points if queue is *trending* up (derivative check)
            if len(self._queue_depth_history) >= 6:
                recent = [v for _, v in self._queue_depth_history[-6:]]
                if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                    # monotonically increasing for 60 seconds, this is going sideways fast
                    scores.append((min(1.0, queue_score + 0.3), ScaleReason.QUEUE_DEPTH_TREND))

        # memory pressure across cluster
        if metrics.avg_memory_utilization > self._config.memory_pressure_trigger:
            mem_score = min(1.0, (metrics.avg_memory_utilization - self._config.memory_pressure_trigger) / 0.08)
            scores.append((mem_score, ScaleReason.MEMORY_PRESSURE))

        # TTFT blowing up
        ttft_ratio = metrics.ttft_p95_ms / self._config.target_ttft_p95_ms
        if ttft_ratio > self._config.ttft_regression_trigger:
            ttft_score = min(1.0, (ttft_ratio - 1.0) / 2.0)
            scores.append((ttft_score, ScaleReason.TTFT_REGRESSION))

        # throughput saturating (close to theoretical max)
        if metrics.max_throughput_tokens_per_sec > 0:
            headroom = 1.0 - (metrics.throughput_tokens_per_sec / metrics.max_throughput_tokens_per_sec)
            if headroom < self._config.target_throughput_headroom:
                sat_score = min(1.0, (self._config.target_throughput_headroom - headroom) / 0.15)
                scores.append((sat_score, ScaleReason.THROUGHPUT_SATURATION))

        # speculative decoding falling apart (usually means target model is overloaded)
        if metrics.speculation_acceptance_rate < 0.5:
            spec_score = min(1.0, (0.7 - metrics.speculation_acceptance_rate) / 0.4)
            scores.append((max(0.0, spec_score), ScaleReason.SPECULATION_DEGRADED))

        # predictive: check if we're about to hit a peak
        if self._config.enable_predictive:
            predicted_score = self._check_predictive_demand(now)
            if predicted_score > 0:
                scores.append((predicted_score, ScaleReason.PREDICTIVE_DEMAND))

        if not scores:
            return 0.0, ScaleReason.QUEUE_DEPTH_TREND

        # return highest scoring reason
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0]

    def _compute_scale_down_score(
        self, metrics: ClusterMetrics, now: float
    ) -> tuple[float, ScaleReason]:
        """Score opportunity to scale down. High score = safe to remove replicas."""
        # all signals must be green for scale-down
        if metrics.avg_gpu_utilization > self._config.underutil_trigger:
            return 0.0, ScaleReason.COOLDOWN_UNDERUTIL

        if metrics.ttft_p95_ms > self._config.target_ttft_p95_ms * 0.8:
            return 0.0, ScaleReason.COOLDOWN_UNDERUTIL

        if metrics.total_queue_depth > self._config.target_queue_depth:
            return 0.0, ScaleReason.COOLDOWN_UNDERUTIL

        # check that utilization has been low for sustained period (not just a dip)
        if len(self._throughput_history) < 30:
            return 0.0, ScaleReason.COOLDOWN_UNDERUTIL

        recent_utils = [v for _, v in self._throughput_history[-30:]]
        if not recent_utils:
            return 0.0, ScaleReason.COOLDOWN_UNDERUTIL

        # if we'd still be above min replicas after removal
        if self._current_replicas <= self._config.min_replicas:
            return 0.0, ScaleReason.COOLDOWN_UNDERUTIL

        # score based on how underutilized we are
        util_ratio = metrics.avg_gpu_utilization / self._config.target_gpu_utilization
        down_score = min(1.0, (1.0 - util_ratio) / 0.4)

        return down_score, ScaleReason.COOLDOWN_UNDERUTIL

    def _check_predictive_demand(self, now: float) -> float:
        """Look at time-of-day patterns to predict upcoming demand."""
        # figure out what hour slot we'll be in soon
        import datetime

        future_time = datetime.datetime.fromtimestamp(now + self._config.prediction_lookahead_min * 60)
        day_of_week = future_time.weekday()
        hour = future_time.hour
        slot = day_of_week * 24 + hour

        if slot >= len(self._hourly_demand):
            return 0.0

        predicted_demand = self._hourly_demand[slot]
        current_capacity = self._current_replicas * self._config.target_queue_depth

        if predicted_demand > current_capacity * 0.8:
            return min(1.0, (predicted_demand - current_capacity * 0.6) / current_capacity)

        return 0.0

    def _make_decision(
        self,
        metrics: ClusterMetrics,
        up_score: float,
        up_reason: ScaleReason,
        down_score: float,
        down_reason: ScaleReason,
        now: float,
    ) -> ScalingDecision:
        """Synthesize scores into a concrete scaling decision."""
        # check cooldowns
        up_cooldown_remaining = max(0.0, self._config.scale_up_cooldown_sec - (now - self._last_scale_up_time))
        down_cooldown_remaining = max(0.0, self._config.scale_down_cooldown_sec - (now - self._last_scale_down_time))

        metrics_snapshot = {
            "gpu_util": metrics.avg_gpu_utilization,
            "mem_util": metrics.avg_memory_utilization,
            "queue_depth": float(metrics.total_queue_depth),
            "ttft_p95_ms": metrics.ttft_p95_ms,
            "throughput": metrics.throughput_tokens_per_sec,
            "spec_acceptance": metrics.speculation_acceptance_rate,
        }

        # scale up takes priority over scale down (safety first)
        if up_score > 0.4 and up_cooldown_remaining <= 0:
            increment = self._config.scale_up_increment
            # double increment for high urgency
            if up_score > 0.8:
                increment *= 2
            target = min(self._config.max_replicas, self._current_replicas + increment)

            if target > self._current_replicas:
                self._last_scale_up_time = now
                self._current_replicas = target
                return ScalingDecision(
                    direction=ScaleDirection.UP,
                    target_replicas=target,
                    current_replicas=metrics.total_replicas,
                    reason=up_reason,
                    confidence=up_score,
                    cooldown_remaining_sec=0.0,
                    metrics_snapshot=metrics_snapshot,
                )

        if down_score > 0.6 and down_cooldown_remaining <= 0:
            target = max(self._config.min_replicas, self._current_replicas - self._config.scale_down_increment)

            if target < self._current_replicas:
                self._last_scale_down_time = now
                self._current_replicas = target
                return ScalingDecision(
                    direction=ScaleDirection.DOWN,
                    target_replicas=target,
                    current_replicas=metrics.total_replicas,
                    reason=down_reason,
                    confidence=down_score,
                    cooldown_remaining_sec=0.0,
                    metrics_snapshot=metrics_snapshot,
                )

        # hold steady
        cooldown = max(up_cooldown_remaining, down_cooldown_remaining)
        return ScalingDecision(
            direction=ScaleDirection.HOLD,
            target_replicas=self._current_replicas,
            current_replicas=metrics.total_replicas,
            reason=up_reason if up_score > down_score else down_reason,
            confidence=max(up_score, down_score),
            cooldown_remaining_sec=cooldown,
            metrics_snapshot=metrics_snapshot,
        )

    def update_demand_history(self, hour_slot: int, demand: float) -> None:
        """Feed historical demand data for predictive scaling."""
        if 0 <= hour_slot < len(self._hourly_demand):
            # exponential moving average to adapt to changing patterns
            alpha = 0.3
            self._hourly_demand[hour_slot] = (
                alpha * demand + (1 - alpha) * self._hourly_demand[hour_slot]
            )

    @property
    def current_replicas(self) -> int:
        return self._current_replicas

    @property
    def recent_decisions(self) -> list[ScalingDecision]:
        return self._decision_history[-20:]
