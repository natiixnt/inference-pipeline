"""Model-aware request routing with load balancing and affinity.

Routes incoming inference requests to the appropriate model replica based on
model availability, current load, SLA requirements, and KV-cache affinity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies."""

    LEAST_LOADED = auto()
    AFFINITY_FIRST = auto()
    LATENCY_OPTIMIZED = auto()
    COST_OPTIMIZED = auto()
    ROUND_ROBIN = auto()


class ReplicaHealth(Enum):
    """Health status of a model replica."""

    HEALTHY = auto()
    DEGRADED = auto()
    DRAINING = auto()
    UNHEALTHY = auto()


@dataclass
class ModelReplica:
    """A single replica (instance) of a deployed model."""

    replica_id: str
    model_id: str
    device_id: str
    gpu_type: str
    max_batch_size: int
    current_batch_size: int = 0
    current_queue_depth: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    health: ReplicaHealth = ReplicaHealth.HEALTHY
    last_heartbeat: float = field(default_factory=time.monotonic)
    total_requests_served: int = 0
    error_rate_1m: float = 0.0

    @property
    def load_factor(self) -> float:
        """Current load as fraction of capacity."""
        if self.max_batch_size == 0:
            return 1.0
        return (self.current_batch_size + self.current_queue_depth) / self.max_batch_size

    @property
    def is_available(self) -> bool:
        return self.health in (ReplicaHealth.HEALTHY, ReplicaHealth.DEGRADED)

    @property
    def available_capacity(self) -> int:
        return max(0, self.max_batch_size - self.current_batch_size - self.current_queue_depth)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    replica_id: str
    model_id: str
    device_id: str
    estimated_wait_ms: float
    strategy_used: RoutingStrategy
    reason: str


@dataclass
class ModelDeployment:
    """Deployment configuration for a model."""

    model_id: str
    min_replicas: int = 1
    max_replicas: int = 8
    target_latency_ms: float = 200.0
    preferred_gpu_types: list[str] = field(default_factory=lambda: ["H100", "A100"])
    can_use_speculative: bool = True


class RequestRouter:
    """Routes inference requests to model replicas.

    Supports multiple routing strategies and handles:
    - Load-aware routing to prevent hotspots
    - Affinity-based routing for KV-cache reuse (session stickiness)
    - Health-aware routing that avoids degraded replicas
    - Graceful draining during model updates
    - Overflow routing to lower-tier GPUs under pressure
    """

    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.AFFINITY_FIRST,
        heartbeat_timeout_ms: float = 5000.0,
        max_queue_depth: int = 100,
    ) -> None:
        self._default_strategy = default_strategy
        self._heartbeat_timeout_ms = heartbeat_timeout_ms
        self._max_queue_depth = max_queue_depth

        self._replicas: dict[str, ModelReplica] = {}
        self._model_replicas: dict[str, list[str]] = {}  # model_id -> replica_ids
        self._deployments: dict[str, ModelDeployment] = {}
        self._session_affinity: dict[str, str] = {}  # session_id -> replica_id
        self._round_robin_index: dict[str, int] = {}

    def register_replica(self, replica: ModelReplica) -> None:
        """Register a new model replica."""
        self._replicas[replica.replica_id] = replica
        if replica.model_id not in self._model_replicas:
            self._model_replicas[replica.model_id] = []
        self._model_replicas[replica.model_id].append(replica.replica_id)
        logger.info(
            "replica_registered",
            replica_id=replica.replica_id,
            model_id=replica.model_id,
        )

    def deregister_replica(self, replica_id: str) -> None:
        """Remove a replica from the routing table."""
        if replica_id not in self._replicas:
            return
        replica = self._replicas.pop(replica_id)
        if replica.model_id in self._model_replicas:
            self._model_replicas[replica.model_id] = [
                r for r in self._model_replicas[replica.model_id] if r != replica_id
            ]
        # Clear affinity entries pointing to this replica
        self._session_affinity = {
            k: v for k, v in self._session_affinity.items() if v != replica_id
        }

    def register_deployment(self, deployment: ModelDeployment) -> None:
        """Register a model deployment configuration."""
        self._deployments[deployment.model_id] = deployment

    def route(
        self,
        model_id: str,
        session_id: Optional[str] = None,
        strategy: Optional[RoutingStrategy] = None,
        required_capacity: int = 1,
    ) -> Optional[RoutingDecision]:
        """Route a request to the best available replica.

        Args:
            model_id: Target model identifier.
            session_id: Optional session for affinity routing.
            strategy: Override routing strategy for this request.
            required_capacity: Minimum available batch slots needed.

        Returns:
            RoutingDecision or None if no replica is available.
        """
        strategy = strategy or self._default_strategy
        candidates = self._get_healthy_replicas(model_id, required_capacity)

        if not candidates:
            logger.warning("no_available_replicas", model_id=model_id)
            return None

        # Try affinity first if session exists
        if session_id and session_id in self._session_affinity:
            affinity_id = self._session_affinity[session_id]
            if affinity_id in self._replicas:
                replica = self._replicas[affinity_id]
                if replica.is_available and replica.available_capacity >= required_capacity:
                    return RoutingDecision(
                        replica_id=affinity_id,
                        model_id=model_id,
                        device_id=replica.device_id,
                        estimated_wait_ms=self._estimate_wait(replica),
                        strategy_used=RoutingStrategy.AFFINITY_FIRST,
                        reason="session_affinity_hit",
                    )

        # Apply selected strategy
        if strategy == RoutingStrategy.LEAST_LOADED:
            selected = self._route_least_loaded(candidates)
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            selected = self._route_latency_optimized(candidates)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            selected = self._route_cost_optimized(candidates)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            selected = self._route_round_robin(model_id, candidates)
        else:
            # AFFINITY_FIRST falls through to least-loaded if no affinity hit
            selected = self._route_least_loaded(candidates)

        if selected is None:
            return None

        # Update session affinity
        if session_id:
            self._session_affinity[session_id] = selected.replica_id

        return selected

    def drain_replica(self, replica_id: str) -> None:
        """Mark a replica as draining (no new requests, finish in-flight)."""
        if replica_id in self._replicas:
            self._replicas[replica_id].health = ReplicaHealth.DRAINING
            logger.info("replica_draining", replica_id=replica_id)

    def update_replica_metrics(
        self,
        replica_id: str,
        current_batch_size: int,
        queue_depth: int,
        avg_latency_ms: float,
        p95_latency_ms: float,
        error_rate: float,
    ) -> None:
        """Update runtime metrics for a replica."""
        if replica_id not in self._replicas:
            return
        replica = self._replicas[replica_id]
        replica.current_batch_size = current_batch_size
        replica.current_queue_depth = queue_depth
        replica.avg_latency_ms = avg_latency_ms
        replica.p95_latency_ms = p95_latency_ms
        replica.error_rate_1m = error_rate
        replica.last_heartbeat = time.monotonic()

        # Auto-degrade replicas with high error rates
        if error_rate > 0.1 and replica.health == ReplicaHealth.HEALTHY:
            replica.health = ReplicaHealth.DEGRADED
            logger.warning("replica_degraded", replica_id=replica_id, error_rate=error_rate)

    def _get_healthy_replicas(
        self, model_id: str, required_capacity: int
    ) -> list[ModelReplica]:
        """Get available replicas for a model with sufficient capacity."""
        now = time.monotonic()
        replica_ids = self._model_replicas.get(model_id, [])
        candidates: list[ModelReplica] = []

        for rid in replica_ids:
            replica = self._replicas[rid]
            # Skip unhealthy or draining replicas
            if not replica.is_available:
                continue
            # Skip replicas with stale heartbeats
            if (now - replica.last_heartbeat) * 1000 > self._heartbeat_timeout_ms:
                replica.health = ReplicaHealth.UNHEALTHY
                continue
            # Skip overloaded replicas
            if replica.current_queue_depth >= self._max_queue_depth:
                continue
            if replica.available_capacity >= required_capacity:
                candidates.append(replica)

        return candidates

    def _route_least_loaded(self, candidates: list[ModelReplica]) -> Optional[RoutingDecision]:
        """Select the replica with the lowest load factor."""
        if not candidates:
            return None
        selected = min(candidates, key=lambda r: r.load_factor)
        return RoutingDecision(
            replica_id=selected.replica_id,
            model_id=selected.model_id,
            device_id=selected.device_id,
            estimated_wait_ms=self._estimate_wait(selected),
            strategy_used=RoutingStrategy.LEAST_LOADED,
            reason=f"load_factor={selected.load_factor:.2f}",
        )

    def _route_latency_optimized(
        self, candidates: list[ModelReplica]
    ) -> Optional[RoutingDecision]:
        """Select the replica with lowest observed latency, weighted by load."""
        if not candidates:
            return None

        def score(r: ModelReplica) -> float:
            # Combine observed latency with load penalty
            load_penalty = r.load_factor * 50.0
            return r.p95_latency_ms + load_penalty

        selected = min(candidates, key=score)
        return RoutingDecision(
            replica_id=selected.replica_id,
            model_id=selected.model_id,
            device_id=selected.device_id,
            estimated_wait_ms=self._estimate_wait(selected),
            strategy_used=RoutingStrategy.LATENCY_OPTIMIZED,
            reason=f"p95={selected.p95_latency_ms:.1f}ms",
        )

    def _route_cost_optimized(self, candidates: list[ModelReplica]) -> Optional[RoutingDecision]:
        """Prefer lower-cost GPUs when latency SLA allows."""
        if not candidates:
            return None

        # Cost ranking: L40S < A100 < H100
        cost_order = {"L40S": 0, "A100": 1, "H100": 2}

        def score(r: ModelReplica) -> tuple[int, float]:
            cost = cost_order.get(r.gpu_type, 99)
            return (cost, r.load_factor)

        selected = min(candidates, key=score)
        return RoutingDecision(
            replica_id=selected.replica_id,
            model_id=selected.model_id,
            device_id=selected.device_id,
            estimated_wait_ms=self._estimate_wait(selected),
            strategy_used=RoutingStrategy.COST_OPTIMIZED,
            reason=f"gpu_type={selected.gpu_type}",
        )

    def _route_round_robin(
        self, model_id: str, candidates: list[ModelReplica]
    ) -> Optional[RoutingDecision]:
        """Simple round-robin across available replicas."""
        if not candidates:
            return None
        idx = self._round_robin_index.get(model_id, 0)
        selected = candidates[idx % len(candidates)]
        self._round_robin_index[model_id] = idx + 1
        return RoutingDecision(
            replica_id=selected.replica_id,
            model_id=selected.model_id,
            device_id=selected.device_id,
            estimated_wait_ms=self._estimate_wait(selected),
            strategy_used=RoutingStrategy.ROUND_ROBIN,
            reason=f"index={idx}",
        )

    def _estimate_wait(self, replica: ModelReplica) -> float:
        """Estimate queue wait time in ms based on current load and throughput."""
        if replica.current_queue_depth == 0:
            return 0.0
        # Rough estimate: avg_latency * queue_depth / batch_size
        batch_parallelism = max(1, replica.max_batch_size - replica.current_batch_size)
        return (replica.avg_latency_ms * replica.current_queue_depth) / batch_parallelism

    def get_routing_table(self) -> dict[str, list[dict]]:
        """Get current routing table for observability."""
        table: dict[str, list[dict]] = {}
        for model_id, replica_ids in self._model_replicas.items():
            table[model_id] = [
                {
                    "replica_id": rid,
                    "health": self._replicas[rid].health.name,
                    "load_factor": self._replicas[rid].load_factor,
                    "available_capacity": self._replicas[rid].available_capacity,
                }
                for rid in replica_ids
                if rid in self._replicas
            ]
        return table
