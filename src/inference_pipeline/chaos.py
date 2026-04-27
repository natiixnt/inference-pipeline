"""Chaos engineering harness for the inference pipeline.

Production inference does not get to assume happy-path hardware. GPUs throw XID
errors, NVLink fabrics flap, NCCL hangs on AllReduce, and OOM kills processes the
moment a request comes in 5% over its memory estimate. The harness here
deterministically injects these failures so we can measure recovery, not assume
it.

Failure modes covered:

1. **GPU device failure**. nvidia-smi reports the device but it stops responding
   to memcpy or kernel launches. We model this by marking the device unhealthy
   in the scheduler. Expected behaviour: in-flight requests on that device are
   rescheduled to peer devices within the failover SLA (target <2s).

2. **Network partition**. A node loses connectivity to the rest of the cluster
   while the GPUs are still alive. From the scheduler's perspective the device
   list shrinks; the partitioned node stops sending heartbeats. Expected:
   requests pinned to partitioned devices are reassigned and the node is
   removed from the placement pool until heartbeats resume.

3. **OOM event**. CUDA OOM during a forward pass. The KV cache eviction path
   should drop the lowest-priority sequences (with their KV blocks) until the
   active set fits in memory, then resume. Expected: no full process restart.

4. **Cascading failures**. Two GPUs fail in quick succession (fabric event).
   Tests that the failover path doesn't stampede the remaining devices and
   trigger their own OOMs.

The harness is deliberately scheduler-level. We do not fault-inject inside CUDA
because reproducing real CUDA failures requires a debugger in the kernel and is
not portable. Instead we simulate the symptoms (device drops out of placement,
in-flight requests fail, memory pressure spikes) and verify the recovery path
end-to-end.

Results from running this harness against v2.0 are in
``benchmarks/chaos_results.md``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class FailureMode(Enum):
    """The failure modes the harness can inject."""

    GPU_FAILURE = "gpu_failure"
    NETWORK_PARTITION = "network_partition"
    OOM = "oom"
    NCCL_HANG = "nccl_hang"
    CASCADING = "cascading"


@dataclass
class FailureEvent:
    """Recorded injection event."""

    timestamp: float
    mode: FailureMode
    affected_devices: list[str]
    description: str = ""


@dataclass
class RecoveryEvent:
    """Recorded recovery event."""

    timestamp: float
    mode: FailureMode
    recovery_seconds: float
    rescheduled_requests: int = 0
    lost_requests: int = 0
    affected_devices: list[str] = field(default_factory=list)

    @property
    def met_sla(self) -> bool:
        """The 2-second failover SLA from the README."""
        return self.recovery_seconds < 2.0

    @property
    def loss_rate(self) -> float:
        total = self.rescheduled_requests + self.lost_requests
        if total == 0:
            return 0.0
        return self.lost_requests / total


@dataclass
class ChaosConfig:
    """Configuration for the chaos harness.

    Defaults are tuned for unit-test runs, where you want the run to finish in
    a few seconds. For production-grade soak tests, push ``max_failure_rate``
    way up and run for hours.
    """

    seed: int = 42
    failover_sla_seconds: float = 2.0
    # Probability per tick of injecting a failure when chaos is active.
    # 0.05 = ~one event per 20 ticks. Pair with a 100ms tick for ~2s MTBI.
    failure_probability: float = 0.05
    # Whether to allow cascading failures (multiple GPUs in <5 seconds).
    allow_cascading: bool = True
    # Maximum simultaneous failed devices. Above this we stop the chaos run -
    # the cluster is too degraded to learn anything new.
    max_simultaneous_failures: int = 3


class ChaosHarness:
    """Coordinates failure injection and recovery measurement.

    Hookup:

    1. Construct with a list of device IDs and callbacks for the actions
       (``mark_device_unhealthy``, ``restore_device``, ``count_in_flight``,
       ``count_active_requests``).
    2. Call :meth:`inject` to fire a specific failure mode now.
    3. Call :meth:`tick` in a loop to randomly inject failures.
    4. Read :meth:`results` for the post-run summary.

    The harness is single-threaded by design. Multi-process chaos goes through
    the scheduler's normal mechanisms; we don't need locks here.
    """

    def __init__(
        self,
        devices: list[str],
        config: Optional[ChaosConfig] = None,
        on_device_fail: Optional[Callable[[str], None]] = None,
        on_device_restore: Optional[Callable[[str], None]] = None,
        in_flight_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        self._devices = list(devices)
        self._config = config or ChaosConfig()
        self._rng = random.Random(self._config.seed)
        self._on_fail = on_device_fail or (lambda _d: None)
        self._on_restore = on_device_restore or (lambda _d: None)
        self._in_flight = in_flight_fn or (lambda _d: 0)
        self._failed_devices: set[str] = set()
        self._failures: list[FailureEvent] = []
        self._recoveries: list[RecoveryEvent] = []

    # ------------------------------------------------------------------
    # Injection paths
    # ------------------------------------------------------------------
    def inject_gpu_failure(self, device_id: Optional[str] = None) -> FailureEvent:
        """Drop a single GPU from the placement pool.

        Picks a random healthy device if not specified. Records the in-flight
        count so the recovery measurement can compare ``rescheduled`` vs ``lost``.
        """
        device = device_id or self._pick_random_healthy()
        if device is None:
            raise RuntimeError("no healthy devices left to fail")
        affected_in_flight = self._in_flight(device)
        self._failed_devices.add(device)
        self._on_fail(device)
        event = FailureEvent(
            timestamp=time.monotonic(),
            mode=FailureMode.GPU_FAILURE,
            affected_devices=[device],
            description=f"GPU {device} simulated XID, {affected_in_flight} in-flight requests",
        )
        self._failures.append(event)
        logger.warning("chaos_gpu_failure", device=device, in_flight=affected_in_flight)
        return event

    def inject_network_partition(self, devices: Optional[list[str]] = None) -> FailureEvent:
        """Partition off a subset of devices from the rest of the cluster.

        Models a top-of-rack switch flap. From the scheduler's POV the
        partitioned devices stop heartbeating. We mark them unhealthy and
        track them for restore.
        """
        if devices is None:
            # Partition off ~25% of healthy devices, picked at random
            healthy = [d for d in self._devices if d not in self._failed_devices]
            count = max(1, len(healthy) // 4)
            devices = self._rng.sample(healthy, k=count)
        for d in devices:
            self._failed_devices.add(d)
            self._on_fail(d)
        event = FailureEvent(
            timestamp=time.monotonic(),
            mode=FailureMode.NETWORK_PARTITION,
            affected_devices=list(devices),
            description=f"Network partition isolating {devices}",
        )
        self._failures.append(event)
        logger.warning("chaos_network_partition", devices=devices)
        return event

    def inject_oom(self, device_id: Optional[str] = None) -> FailureEvent:
        """Trigger an OOM on the chosen device.

        Unlike GPU_FAILURE this is recoverable: we expect the KV evictor to
        free space and the device to come back online without manual
        intervention. We mark unhealthy briefly and clear in :meth:`recover`.
        """
        device = device_id or self._pick_random_healthy()
        if device is None:
            raise RuntimeError("no healthy devices left to OOM")
        self._failed_devices.add(device)
        self._on_fail(device)
        event = FailureEvent(
            timestamp=time.monotonic(),
            mode=FailureMode.OOM,
            affected_devices=[device],
            description=f"CUDA OOM on {device}, KV evictor should reclaim",
        )
        self._failures.append(event)
        logger.warning("chaos_oom", device=device)
        return event

    def inject_cascading(self) -> FailureEvent:
        """Two GPUs fail within 100ms. Models a fabric / PSU event.

        Tests the scheduler's ability to handle multiple concurrent failovers
        without stampeding the survivors. We've seen real-world cases where
        the second failover triggers OOMs on the survivors because the first
        failover packed them too tight.
        """
        if not self._config.allow_cascading:
            raise RuntimeError("cascading failures disabled in config")
        first = self._pick_random_healthy()
        if first is None:
            raise RuntimeError("no healthy devices left for cascade")
        self._failed_devices.add(first)
        self._on_fail(first)
        # Tiny gap to model the realistic "two events 100ms apart" pattern.
        time.sleep(0.001)
        second = self._pick_random_healthy()
        if second is None:
            # Only one device available; treat as a single failure.
            event = FailureEvent(
                timestamp=time.monotonic(),
                mode=FailureMode.CASCADING,
                affected_devices=[first],
                description="cascade with only one device available, single failure recorded",
            )
            self._failures.append(event)
            return event
        self._failed_devices.add(second)
        self._on_fail(second)
        event = FailureEvent(
            timestamp=time.monotonic(),
            mode=FailureMode.CASCADING,
            affected_devices=[first, second],
            description="Two GPUs failed within 100ms (fabric event)",
        )
        self._failures.append(event)
        logger.error("chaos_cascading", devices=[first, second])
        return event

    def inject(self, mode: FailureMode) -> FailureEvent:
        """Dispatch by mode."""
        if mode == FailureMode.GPU_FAILURE:
            return self.inject_gpu_failure()
        if mode == FailureMode.NETWORK_PARTITION:
            return self.inject_network_partition()
        if mode == FailureMode.OOM:
            return self.inject_oom()
        if mode == FailureMode.NCCL_HANG:
            # NCCL hang is similar to GPU failure from the scheduler's POV:
            # the device stops responding. We model it as such, and the only
            # real difference is the logged description.
            event = self.inject_gpu_failure()
            event.mode = FailureMode.NCCL_HANG
            event.description = "NCCL collective hang - watchdog timeout"
            return event
        if mode == FailureMode.CASCADING:
            return self.inject_cascading()
        raise ValueError(f"unknown failure mode {mode}")

    # ------------------------------------------------------------------
    # Recovery and ticking
    # ------------------------------------------------------------------
    def recover(self, device_id: str, *, rescheduled: int = 0, lost: int = 0) -> RecoveryEvent:
        """Mark a device healthy again and record the recovery time.

        ``rescheduled`` and ``lost`` come from the scheduler: how many of the
        in-flight requests on the failed device made it onto a peer, vs how
        many returned an error to the user. The README claims ``< 2s`` failover
        with near-zero loss; this is what we measure to verify.
        """
        if device_id not in self._failed_devices:
            raise RuntimeError(f"device {device_id} is not in the failed set")
        # Find the most recent failure event for this device
        last_failure: Optional[FailureEvent] = None
        for ev in reversed(self._failures):
            if device_id in ev.affected_devices:
                last_failure = ev
                break
        if last_failure is None:
            raise RuntimeError(f"no failure event for {device_id}")

        recovery_seconds = time.monotonic() - last_failure.timestamp
        self._failed_devices.discard(device_id)
        self._on_restore(device_id)

        event = RecoveryEvent(
            timestamp=time.monotonic(),
            mode=last_failure.mode,
            recovery_seconds=recovery_seconds,
            rescheduled_requests=rescheduled,
            lost_requests=lost,
            affected_devices=[device_id],
        )
        self._recoveries.append(event)
        if event.met_sla:
            logger.info(
                "chaos_recovery_sla_met",
                device=device_id,
                seconds=recovery_seconds,
                rescheduled=rescheduled,
                lost=lost,
            )
        else:
            logger.error(
                "chaos_recovery_sla_violated",
                device=device_id,
                seconds=recovery_seconds,
                sla=self._config.failover_sla_seconds,
            )
        return event

    def tick(self) -> Optional[FailureEvent]:
        """Probabilistically inject a failure. Returns the event if any."""
        if len(self._failed_devices) >= self._config.max_simultaneous_failures:
            return None
        if self._rng.random() >= self._config.failure_probability:
            return None
        # Pick a mode weighted by what we see in production. GPU failures and
        # OOM dominate; network partitions are rarer; cascading rarest.
        mode = self._rng.choices(
            [FailureMode.GPU_FAILURE, FailureMode.OOM, FailureMode.NETWORK_PARTITION,
             FailureMode.NCCL_HANG, FailureMode.CASCADING],
            weights=[0.40, 0.30, 0.15, 0.10, 0.05],
            k=1,
        )[0]
        try:
            return self.inject(mode)
        except RuntimeError as exc:
            logger.debug("chaos_inject_skipped", mode=mode.value, reason=str(exc))
            return None

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def _pick_random_healthy(self) -> Optional[str]:
        healthy = [d for d in self._devices if d not in self._failed_devices]
        if not healthy:
            return None
        return self._rng.choice(healthy)

    @property
    def failed_devices(self) -> set[str]:
        return set(self._failed_devices)

    @property
    def failures(self) -> list[FailureEvent]:
        return list(self._failures)

    @property
    def recoveries(self) -> list[RecoveryEvent]:
        return list(self._recoveries)

    def results(self) -> dict[str, object]:
        """Roll-up of all the runs against this harness."""
        if not self._recoveries:
            return {
                "failures_injected": len(self._failures),
                "recoveries_completed": 0,
                "average_recovery_seconds": 0.0,
                "p95_recovery_seconds": 0.0,
                "sla_violations": 0,
                "total_lost_requests": 0,
                "total_rescheduled": 0,
            }
        seconds = sorted(r.recovery_seconds for r in self._recoveries)
        p95_idx = max(0, int(len(seconds) * 0.95) - 1)
        return {
            "failures_injected": len(self._failures),
            "recoveries_completed": len(self._recoveries),
            "average_recovery_seconds": sum(seconds) / len(seconds),
            "p95_recovery_seconds": seconds[p95_idx],
            "sla_violations": sum(1 for r in self._recoveries if not r.met_sla),
            "total_lost_requests": sum(r.lost_requests for r in self._recoveries),
            "total_rescheduled": sum(r.rescheduled_requests for r in self._recoveries),
            "loss_rate": (
                sum(r.lost_requests for r in self._recoveries)
                / max(1, sum(r.lost_requests + r.rescheduled_requests for r in self._recoveries))
            ),
        }
