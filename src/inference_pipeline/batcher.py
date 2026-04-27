"""Continuous batching with dynamic batch sizing.

Implements iteration-level scheduling where new requests can join an in-flight
batch between decode steps. Batch size adapts based on GPU memory pressure,
queue depth, and per-request generation progress.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RequestPhase(Enum):
    """Lifecycle phase of an inference request within the batcher."""

    PREFILL = auto()      # Processing input prompt (compute-bound)
    DECODE = auto()       # Generating tokens one-by-one (memory-bound)
    COMPLETE = auto()     # Finished generation
    EVICTED = auto()      # Preempted, waiting to resume


@dataclass
class BatchSlot:
    """A slot in the continuous batch representing one active request."""

    request_id: str
    model_id: str
    phase: RequestPhase = RequestPhase.PREFILL
    input_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    max_new_tokens: int = 2048
    kv_cache_block_ids: list[int] = field(default_factory=list)
    prefill_start_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None
    priority: int = 3

    @property
    def total_tokens(self) -> int:
        return len(self.input_token_ids) + len(self.generated_token_ids)

    @property
    def is_done(self) -> bool:
        return (
            self.phase == RequestPhase.COMPLETE
            or len(self.generated_token_ids) >= self.max_new_tokens
        )

    @property
    def time_to_first_token_ms(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.prefill_start_time) * 1000


@dataclass
class BatchConfig:
    """Configuration for the continuous batcher."""

    max_batch_size: int = 64
    max_total_tokens: int = 131072  # Total tokens across all slots
    prefill_chunk_size: int = 512  # Max tokens to prefill per iteration
    min_batch_efficiency: float = 0.4  # Minimum GPU utilization target
    memory_pressure_threshold: float = 0.85  # Start shedding at this utilization
    batch_timeout_ms: float = 2.0  # Max wait for batch formation
    priority_aging_ms: float = 100.0  # Priority boost per 100ms wait


class ContinuousBatcher:
    """Iteration-level batching engine for LLM inference.

    On each iteration (decode step), the batcher:
    1. Admits new requests from the waiting queue into free slots
    2. Runs chunked prefill for requests in PREFILL phase
    3. Runs a single decode step for all DECODE-phase requests
    4. Evicts completed requests and reclaims their resources
    5. Adjusts batch size based on memory pressure signals
    """

    def __init__(self, config: Optional[BatchConfig] = None) -> None:
        self._config = config or BatchConfig()
        self._active_slots: dict[str, BatchSlot] = {}
        self._waiting_queue: list[BatchSlot] = []
        self._total_tokens_in_batch: int = 0
        self._iteration_count: int = 0
        self._dynamic_max_batch: int = self._config.max_batch_size

        # Metrics
        self._total_admitted: int = 0
        self._total_completed: int = 0
        self._total_preempted: int = 0
        self._ttft_samples: list[float] = []

    def add_request(
        self,
        request_id: str,
        model_id: str,
        input_token_ids: list[int],
        max_new_tokens: int = 2048,
        priority: int = 3,
    ) -> None:
        """Add a new request to the waiting queue."""
        slot = BatchSlot(
            request_id=request_id,
            model_id=model_id,
            input_token_ids=input_token_ids,
            max_new_tokens=max_new_tokens,
            priority=priority,
        )
        self._waiting_queue.append(slot)
        self._waiting_queue.sort(key=lambda s: s.priority)

    def iteration(self, memory_utilization: float = 0.0) -> BatchIterationResult:
        """Execute one iteration of the continuous batching loop.

        Args:
            memory_utilization: Current GPU memory utilization [0, 1].

        Returns:
            BatchIterationResult with lists of requests to prefill, decode, and complete.
        """
        self._iteration_count += 1

        # Adjust dynamic batch size based on memory pressure
        self._adjust_batch_size(memory_utilization)

        # Evict completed requests
        completed = self._evict_completed()

        # Handle memory pressure by evicting lowest-priority decode requests
        evicted: list[str] = []
        if memory_utilization > self._config.memory_pressure_threshold:
            evicted = self._shed_load(memory_utilization)

        # Admit new requests from waiting queue
        admitted = self._admit_requests()

        # Partition active slots by phase
        prefill_slots: list[BatchSlot] = []
        decode_slots: list[BatchSlot] = []

        for slot in self._active_slots.values():
            if slot.phase == RequestPhase.PREFILL:
                prefill_slots.append(slot)
            elif slot.phase == RequestPhase.DECODE:
                decode_slots.append(slot)

        # Compute prefill budget: limit prefill tokens to maintain decode latency
        prefill_budget = self._compute_prefill_budget(len(decode_slots))

        return BatchIterationResult(
            iteration=self._iteration_count,
            prefill_requests=prefill_slots,
            decode_requests=decode_slots,
            completed_ids=completed,
            admitted_ids=admitted,
            evicted_ids=evicted,
            prefill_token_budget=prefill_budget,
            batch_size=len(self._active_slots),
            total_tokens=self._total_tokens_in_batch,
        )

    def mark_prefill_done(self, request_id: str) -> None:
        """Transition a request from PREFILL to DECODE phase."""
        if request_id not in self._active_slots:
            return
        slot = self._active_slots[request_id]
        slot.phase = RequestPhase.DECODE
        slot.first_token_time = time.monotonic()

        ttft = slot.time_to_first_token_ms
        if ttft is not None:
            self._ttft_samples.append(ttft)

    def mark_token_generated(self, request_id: str, token_id: int) -> None:
        """Record a newly generated token for a request."""
        if request_id not in self._active_slots:
            return
        slot = self._active_slots[request_id]
        slot.generated_token_ids.append(token_id)
        self._total_tokens_in_batch += 1

        if slot.is_done:
            slot.phase = RequestPhase.COMPLETE

    def _admit_requests(self) -> list[str]:
        """Admit waiting requests into available batch slots."""
        admitted: list[str] = []
        available_slots = self._dynamic_max_batch - len(self._active_slots)
        available_tokens = self._config.max_total_tokens - self._total_tokens_in_batch

        while self._waiting_queue and available_slots > 0:
            slot = self._waiting_queue[0]
            token_count = len(slot.input_token_ids)

            if token_count > available_tokens:
                break

            self._waiting_queue.pop(0)
            self._active_slots[slot.request_id] = slot
            self._total_tokens_in_batch += token_count
            available_slots -= 1
            available_tokens -= token_count
            admitted.append(slot.request_id)
            self._total_admitted += 1

        return admitted

    def _evict_completed(self) -> list[str]:
        """Remove completed requests and free their resources."""
        completed: list[str] = []
        for req_id, slot in list(self._active_slots.items()):
            if slot.is_done:
                self._total_tokens_in_batch -= slot.total_tokens
                del self._active_slots[req_id]
                completed.append(req_id)
                self._total_completed += 1
        return completed

    def _shed_load(self, memory_utilization: float) -> list[str]:
        """Evict lowest-priority requests to reduce memory pressure."""
        evicted: list[str] = []
        target_util = self._config.memory_pressure_threshold - 0.1

        # Sort active decode requests by priority (highest value = lowest priority)
        decode_requests = sorted(
            [
                (slot, req_id)
                for req_id, slot in self._active_slots.items()
                if slot.phase == RequestPhase.DECODE
            ],
            key=lambda x: -x[0].priority,
        )

        estimated_util = memory_utilization
        for slot, req_id in decode_requests:
            if estimated_util <= target_util:
                break

            # Estimate memory freed by evicting this request
            token_fraction = slot.total_tokens / max(1, self._total_tokens_in_batch)
            estimated_util -= token_fraction * memory_utilization * 0.5

            slot.phase = RequestPhase.EVICTED
            self._total_tokens_in_batch -= slot.total_tokens
            del self._active_slots[req_id]
            evicted.append(req_id)
            self._total_preempted += 1

            # Re-queue for later processing
            self._waiting_queue.append(slot)

        if evicted:
            logger.warning("load_shed", evicted_count=len(evicted), memory_util=memory_utilization)

        return evicted

    def _adjust_batch_size(self, memory_utilization: float) -> None:
        """Dynamically adjust max batch size based on memory pressure."""
        if memory_utilization > 0.9:
            # Aggressive reduction
            self._dynamic_max_batch = max(1, int(self._dynamic_max_batch * 0.7))
        elif memory_utilization > self._config.memory_pressure_threshold:
            # Gentle reduction
            self._dynamic_max_batch = max(1, self._dynamic_max_batch - 2)
        elif memory_utilization < 0.6 and self._dynamic_max_batch < self._config.max_batch_size:
            # Grow batch size when memory is available
            self._dynamic_max_batch = min(
                self._config.max_batch_size, self._dynamic_max_batch + 1
            )

    def _compute_prefill_budget(self, decode_count: int) -> int:
        """Compute token budget for prefill to avoid starving decode requests.

        We limit prefill work proportional to decode batch size to maintain
        consistent inter-token latency for in-flight generations.
        """
        if decode_count == 0:
            return self._config.prefill_chunk_size * 4

        # As decode batch grows, reduce prefill budget to maintain ITL
        budget_fraction = max(0.1, 1.0 - (decode_count / self._dynamic_max_batch))
        return int(self._config.prefill_chunk_size * budget_fraction)

    @property
    def ttft_p95_ms(self) -> Optional[float]:
        """Get p95 time-to-first-token in milliseconds."""
        if len(self._ttft_samples) < 10:
            return None
        return float(np.percentile(self._ttft_samples[-1000:], 95))

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "active_slots": len(self._active_slots),
            "waiting_queue": len(self._waiting_queue),
            "total_tokens": self._total_tokens_in_batch,
            "dynamic_max_batch": self._dynamic_max_batch,
            "total_admitted": self._total_admitted,
            "total_completed": self._total_completed,
            "total_preempted": self._total_preempted,
            "iterations": self._iteration_count,
        }


@dataclass
class BatchIterationResult:
    """Result of a single batching iteration."""

    iteration: int
    prefill_requests: list[BatchSlot]
    decode_requests: list[BatchSlot]
    completed_ids: list[str]
    admitted_ids: list[str]
    evicted_ids: list[str]
    prefill_token_budget: int
    batch_size: int
    total_tokens: int
