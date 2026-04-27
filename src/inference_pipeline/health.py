"""Health check endpoint with GPU memory status and degradation detection.

Not your typical /healthz that returns 200 and calls it a day. This one actually
knows if the system is about to fall over before it does. Checks GPU memory pressure,
model loading state, and detects gradual degradation (the silent killer of serving systems).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """System health states. DEGRADED is the interesting one."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # still serving but something's off
    UNHEALTHY = "unhealthy"  # stop sending traffic NOW
    STARTING = "starting"  # models loading, not ready yet


class DegradationSignal(str, Enum):
    """What's going wrong. Multiple can fire simultaneously."""

    MEMORY_PRESSURE = "memory_pressure"
    TTFT_REGRESSION = "ttft_regression"
    THROUGHPUT_DROP = "throughput_drop"
    HIGH_QUEUE_DEPTH = "high_queue_depth"
    MODEL_NOT_LOADED = "model_not_loaded"
    CACHE_THRASHING = "cache_thrashing"
    GPU_THERMAL_THROTTLE = "gpu_thermal_throttle"


@dataclass
class GPUHealthState:
    """Per-device health snapshot. Polled at ~1Hz from NVML/DCGM."""

    device_id: str
    memory_total_bytes: int
    memory_used_bytes: int
    memory_free_bytes: int
    gpu_utilization_pct: float
    temperature_celsius: float
    power_draw_watts: float
    is_healthy: bool = True
    throttle_reason: Optional[str] = None

    @property
    def memory_utilization(self) -> float:
        if self.memory_total_bytes == 0:
            return 0.0
        return self.memory_used_bytes / self.memory_total_bytes

    @property
    def is_memory_critical(self) -> bool:
        # 95% is where fragmentation starts eating you alive
        return self.memory_utilization > 0.95

    @property
    def is_thermal_throttling(self) -> bool:
        # H100 starts throttling around 83C, A100 at 80C
        return self.temperature_celsius > 82


@dataclass
class ModelReadinessState:
    """Tracks whether each model shard is loaded and ready to serve."""

    model_id: str
    is_loaded: bool = False
    load_progress_pct: float = 0.0
    load_start_time: Optional[float] = None
    last_inference_time: Optional[float] = None
    consecutive_errors: int = 0
    warmup_complete: bool = False

    @property
    def is_ready(self) -> bool:
        # loaded + warmed up + not erroring out
        return self.is_loaded and self.warmup_complete and self.consecutive_errors < 3

    @property
    def seconds_since_last_inference(self) -> Optional[float]:
        if self.last_inference_time is None:
            return None
        return time.monotonic() - self.last_inference_time


@dataclass
class DegradationEvent:
    """A detected degradation with context for the on-call."""

    signal: DegradationSignal
    severity: float  # 0.0 to 1.0, how bad is it
    message: str
    device_id: Optional[str] = None
    started_at: float = field(default_factory=time.monotonic)
    resolved: bool = False


@dataclass
class HealthReport:
    """Full health check response. This is what the LB and k8s probes consume."""

    status: HealthStatus
    uptime_seconds: float
    gpu_states: list[GPUHealthState]
    model_states: list[ModelReadinessState]
    active_degradations: list[DegradationEvent]
    queue_depth: int
    active_requests: int
    ttft_p95_ms: float
    throughput_tokens_per_sec: float


class HealthChecker:
    """Continuous health monitoring with degradation detection.

    Runs as a background coroutine, polling GPU state and computing rolling
    metrics. The key insight: don't wait for things to break. Detect the *trend*
    toward breakage and signal DEGRADED before you hit UNHEALTHY.

    The LB drains DEGRADED nodes gradually instead of hard-cutting them,
    which avoids the thundering herd problem of all traffic shifting at once.
    """

    def __init__(
        self,
        memory_pressure_threshold: float = 0.90,  # start worrying at 90%
        ttft_regression_factor: float = 2.0,  # 2x baseline = degraded
        throughput_drop_factor: float = 0.6,  # 40% drop = degraded
        queue_depth_threshold: int = 500,  # more than this and we're backed up
        cache_eviction_rate_threshold: float = 0.1,  # 10% eviction/sec = thrashing
        thermal_threshold_celsius: float = 82.0,
    ) -> None:
        self._memory_threshold = memory_pressure_threshold
        self._ttft_regression_factor = ttft_regression_factor
        self._throughput_drop_factor = throughput_drop_factor
        self._queue_depth_threshold = queue_depth_threshold
        self._cache_eviction_threshold = cache_eviction_rate_threshold
        self._thermal_threshold = thermal_threshold_celsius

        self._start_time = time.monotonic()
        self._gpu_states: dict[str, GPUHealthState] = {}
        self._model_states: dict[str, ModelReadinessState] = {}
        self._active_degradations: list[DegradationEvent] = []

        # rolling metrics for trend detection
        self._ttft_baseline_ms: Optional[float] = None
        self._throughput_baseline: Optional[float] = None
        self._ttft_history: list[float] = []
        self._throughput_history: list[float] = []
        self._cache_eviction_history: list[float] = []

    def update_gpu_state(self, state: GPUHealthState) -> None:
        """Called by the NVML poller every ~1s."""
        self._gpu_states[state.device_id] = state

    def update_model_state(self, state: ModelReadinessState) -> None:
        """Called when model loading state changes."""
        self._model_states[state.model_id] = state

    def record_ttft(self, ttft_ms: float) -> None:
        """Record a TTFT observation for trend detection."""
        self._ttft_history.append(ttft_ms)
        # keep last 1000 observations, ~10 minutes at moderate load
        if len(self._ttft_history) > 1000:
            self._ttft_history = self._ttft_history[-1000:]

        # establish baseline from first 100 observations (warmup period)
        if self._ttft_baseline_ms is None and len(self._ttft_history) >= 100:
            sorted_hist = sorted(self._ttft_history[:100])
            self._ttft_baseline_ms = sorted_hist[94]  # p95 of warmup period

    def record_throughput(self, tokens_per_sec: float) -> None:
        """Record throughput observation."""
        self._throughput_history.append(tokens_per_sec)
        if len(self._throughput_history) > 100:
            self._throughput_history = self._throughput_history[-100:]

        if self._throughput_baseline is None and len(self._throughput_history) >= 20:
            self._throughput_baseline = sum(self._throughput_history[:20]) / 20

    def record_cache_evictions(self, evictions_per_sec: float) -> None:
        """Track KV-cache eviction rate for thrashing detection."""
        self._cache_eviction_history.append(evictions_per_sec)
        if len(self._cache_eviction_history) > 60:
            self._cache_eviction_history = self._cache_eviction_history[-60:]

    def check_health(self, queue_depth: int, active_requests: int) -> HealthReport:
        """Run all degradation checks and produce a health report.

        This is the main entry point, called by the /health endpoint handler.
        """
        self._active_degradations = []  # reset, re-detect each time

        # check all the ways things can go sideways
        self._check_memory_pressure()
        self._check_ttft_regression()
        self._check_throughput_drop()
        self._check_queue_depth(queue_depth)
        self._check_model_readiness()
        self._check_cache_thrashing()
        self._check_thermal_throttling()

        # determine overall status
        status = self._compute_status()

        # compute current p95 TTFT from recent observations
        ttft_p95 = 0.0
        if self._ttft_history:
            sorted_ttft = sorted(self._ttft_history[-100:])
            idx = min(len(sorted_ttft) - 1, int(len(sorted_ttft) * 0.95))
            ttft_p95 = sorted_ttft[idx]

        # current throughput from recent window
        throughput = 0.0
        if self._throughput_history:
            throughput = sum(self._throughput_history[-10:]) / min(10, len(self._throughput_history))

        return HealthReport(
            status=status,
            uptime_seconds=time.monotonic() - self._start_time,
            gpu_states=list(self._gpu_states.values()),
            model_states=list(self._model_states.values()),
            active_degradations=list(self._active_degradations),
            queue_depth=queue_depth,
            active_requests=active_requests,
            ttft_p95_ms=ttft_p95,
            throughput_tokens_per_sec=throughput,
        )

    def _check_memory_pressure(self) -> None:
        """Flag GPUs approaching OOM territory."""
        for device_id, state in self._gpu_states.items():
            if state.memory_utilization > self._memory_threshold:
                severity = min(1.0, (state.memory_utilization - self._memory_threshold) / 0.10)
                self._active_degradations.append(
                    DegradationEvent(
                        signal=DegradationSignal.MEMORY_PRESSURE,
                        severity=severity,
                        message=(
                            f"{device_id} at {state.memory_utilization:.1%} memory, "
                            f"free: {state.memory_free_bytes / (1024**3):.1f}GB"
                        ),
                        device_id=device_id,
                    )
                )

    def _check_ttft_regression(self) -> None:
        """Detect TTFT creeping up vs baseline. Classic sign of memory pressure or scheduling issues."""
        if self._ttft_baseline_ms is None or len(self._ttft_history) < 20:
            return

        recent_sorted = sorted(self._ttft_history[-50:])
        idx = min(len(recent_sorted) - 1, int(len(recent_sorted) * 0.95))
        current_p95 = recent_sorted[idx]

        ratio = current_p95 / self._ttft_baseline_ms
        if ratio > self._ttft_regression_factor:
            severity = min(1.0, (ratio - self._ttft_regression_factor) / 2.0)
            self._active_degradations.append(
                DegradationEvent(
                    signal=DegradationSignal.TTFT_REGRESSION,
                    severity=severity,
                    message=(
                        f"TTFT p95 regressed {ratio:.1f}x vs baseline "
                        f"({current_p95:.0f}ms vs {self._ttft_baseline_ms:.0f}ms)"
                    ),
                )
            )

    def _check_throughput_drop(self) -> None:
        """Detect throughput falling off a cliff."""
        if self._throughput_baseline is None or len(self._throughput_history) < 10:
            return

        current = sum(self._throughput_history[-10:]) / 10
        ratio = current / self._throughput_baseline

        if ratio < self._throughput_drop_factor:
            severity = min(1.0, (self._throughput_drop_factor - ratio) / 0.3)
            self._active_degradations.append(
                DegradationEvent(
                    signal=DegradationSignal.THROUGHPUT_DROP,
                    severity=severity,
                    message=(
                        f"Throughput dropped to {ratio:.0%} of baseline "
                        f"({current:.0f} vs {self._throughput_baseline:.0f} tok/s)"
                    ),
                )
            )

    def _check_queue_depth(self, queue_depth: int) -> None:
        """Requests piling up means we can't keep up."""
        if queue_depth > self._queue_depth_threshold:
            severity = min(1.0, queue_depth / (self._queue_depth_threshold * 3))
            self._active_degradations.append(
                DegradationEvent(
                    signal=DegradationSignal.HIGH_QUEUE_DEPTH,
                    severity=severity,
                    message=f"Queue depth at {queue_depth}, threshold is {self._queue_depth_threshold}",
                )
            )

    def _check_model_readiness(self) -> None:
        """Catch models that crashed or never finished loading."""
        for model_id, state in self._model_states.items():
            if not state.is_ready:
                self._active_degradations.append(
                    DegradationEvent(
                        signal=DegradationSignal.MODEL_NOT_LOADED,
                        severity=1.0 if not state.is_loaded else 0.5,
                        message=(
                            f"{model_id}: loaded={state.is_loaded}, "
                            f"warmup={state.warmup_complete}, "
                            f"errors={state.consecutive_errors}"
                        ),
                    )
                )

    def _check_cache_thrashing(self) -> None:
        """KV-cache evicting faster than we can fill = thrashing. Bad news."""
        if len(self._cache_eviction_history) < 10:
            return

        recent_rate = sum(self._cache_eviction_history[-10:]) / 10
        if recent_rate > self._cache_eviction_threshold:
            severity = min(1.0, recent_rate / (self._cache_eviction_threshold * 3))
            self._active_degradations.append(
                DegradationEvent(
                    signal=DegradationSignal.CACHE_THRASHING,
                    severity=severity,
                    message=f"Cache eviction rate {recent_rate:.3f}/s, thrashing threshold {self._cache_eviction_threshold}",
                )
            )

    def _check_thermal_throttling(self) -> None:
        """GPUs cooking themselves. Performance drops 10-20% when this kicks in."""
        for device_id, state in self._gpu_states.items():
            if state.is_thermal_throttling:
                severity = min(1.0, (state.temperature_celsius - self._thermal_threshold) / 10.0)
                self._active_degradations.append(
                    DegradationEvent(
                        signal=DegradationSignal.GPU_THERMAL_THROTTLE,
                        severity=severity,
                        message=f"{device_id} at {state.temperature_celsius}C, throttling likely",
                        device_id=device_id,
                    )
                )

    def _compute_status(self) -> HealthStatus:
        """Roll up degradation signals into overall health status."""
        # if any model isn't ready, we're still starting
        if any(not s.is_loaded for s in self._model_states.values()):
            return HealthStatus.STARTING

        if not self._active_degradations:
            return HealthStatus.HEALTHY

        max_severity = max(d.severity for d in self._active_degradations)
        num_signals = len(self._active_degradations)

        # multiple concurrent degradations compound
        if max_severity > 0.8 or num_signals >= 3:
            return HealthStatus.UNHEALTHY
        if max_severity > 0.3 or num_signals >= 2:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY
