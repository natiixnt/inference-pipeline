"""Prometheus metrics collection for the inference pipeline.

Exposes key performance indicators including throughput, latency percentiles,
GPU utilization, cache hit rates, and scheduling efficiency.
"""

from __future__ import annotations

import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)
import structlog

logger = structlog.get_logger(__name__)

# Custom registry to avoid polluting the default
REGISTRY = CollectorRegistry()

# Request metrics
REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total inference requests received",
    ["model_id", "status"],
    registry=REGISTRY,
)

REQUESTS_IN_FLIGHT = Gauge(
    "inference_requests_in_flight",
    "Currently processing requests",
    ["model_id"],
    registry=REGISTRY,
)

# Latency metrics
TTFT_HISTOGRAM = Histogram(
    "inference_time_to_first_token_ms",
    "Time to first token in milliseconds",
    ["model_id"],
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 500, 1000],
    registry=REGISTRY,
)

TOTAL_LATENCY_HISTOGRAM = Histogram(
    "inference_total_latency_ms",
    "Total request latency in milliseconds",
    ["model_id"],
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000],
    registry=REGISTRY,
)

INTER_TOKEN_LATENCY = Histogram(
    "inference_inter_token_latency_ms",
    "Inter-token latency in milliseconds",
    ["model_id"],
    buckets=[5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
    registry=REGISTRY,
)

# Throughput metrics
TOKENS_GENERATED = Counter(
    "inference_tokens_generated_total",
    "Total tokens generated",
    ["model_id"],
    registry=REGISTRY,
)

TOKENS_PER_SECOND = Gauge(
    "inference_tokens_per_second",
    "Current token generation throughput",
    ["model_id"],
    registry=REGISTRY,
)

# Batch metrics
BATCH_SIZE_HISTOGRAM = Histogram(
    "inference_batch_size",
    "Batch size per iteration",
    ["model_id"],
    buckets=[1, 2, 4, 8, 16, 32, 48, 64, 96, 128],
    registry=REGISTRY,
)

BATCH_TOKENS_HISTOGRAM = Histogram(
    "inference_batch_total_tokens",
    "Total tokens in batch per iteration",
    ["model_id"],
    buckets=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    registry=REGISTRY,
)

# GPU metrics
GPU_MEMORY_UTILIZATION = Gauge(
    "inference_gpu_memory_utilization",
    "GPU memory utilization ratio",
    ["device_id", "gpu_type"],
    registry=REGISTRY,
)

GPU_COMPUTE_UTILIZATION = Gauge(
    "inference_gpu_compute_utilization",
    "GPU compute utilization ratio",
    ["device_id", "gpu_type"],
    registry=REGISTRY,
)

# KV-cache metrics
KV_CACHE_UTILIZATION = Gauge(
    "inference_kv_cache_utilization",
    "KV-cache memory utilization ratio",
    ["device_id"],
    registry=REGISTRY,
)

KV_CACHE_PREFIX_HIT_RATE = Gauge(
    "inference_kv_cache_prefix_hit_rate",
    "Prefix sharing hit rate",
    registry=REGISTRY,
)

KV_CACHE_EVICTIONS = Counter(
    "inference_kv_cache_evictions_total",
    "Total KV-cache block evictions",
    ["device_id"],
    registry=REGISTRY,
)

# Scheduler metrics
SCHEDULER_QUEUE_DEPTH = Gauge(
    "inference_scheduler_queue_depth",
    "Number of requests in scheduler queue",
    ["priority"],
    registry=REGISTRY,
)

SCHEDULER_PREEMPTIONS = Counter(
    "inference_scheduler_preemptions_total",
    "Total request preemptions",
    ["reason"],
    registry=REGISTRY,
)

SCHEDULER_PLACEMENT_LATENCY = Histogram(
    "inference_scheduler_placement_latency_ms",
    "Time to place a request on a GPU",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50],
    registry=REGISTRY,
)

# Speculative decoding metrics
SPECULATIVE_ACCEPTANCE_RATE = Gauge(
    "inference_speculative_acceptance_rate",
    "Token acceptance rate for speculative decoding",
    ["model_id"],
    registry=REGISTRY,
)

SPECULATIVE_TOKENS_SAVED = Counter(
    "inference_speculative_tokens_saved_total",
    "Tokens that avoided individual decode steps",
    ["model_id"],
    registry=REGISTRY,
)

# Model update metrics
MODEL_UPDATE_DURATION = Histogram(
    "inference_model_update_duration_s",
    "Duration of model update rollouts",
    ["model_id"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)

MODEL_ROLLBACKS = Counter(
    "inference_model_rollbacks_total",
    "Total model update rollbacks",
    ["model_id", "reason"],
    registry=REGISTRY,
)

# System info
SYSTEM_INFO = Info(
    "inference_pipeline",
    "Inference pipeline version and configuration",
    registry=REGISTRY,
)


class MetricsCollector:
    """Convenience wrapper for recording metrics from pipeline components."""

    def __init__(self, serve_metrics: bool = False, port: int = 9090) -> None:
        self._start_time = time.monotonic()
        if serve_metrics:
            start_http_server(port, registry=REGISTRY)
            logger.info("metrics_server_started", port=port)

    def record_request_received(self, model_id: str) -> None:
        REQUESTS_TOTAL.labels(model_id=model_id, status="received").inc()
        REQUESTS_IN_FLIGHT.labels(model_id=model_id).inc()

    def record_request_completed(
        self,
        model_id: str,
        ttft_ms: float,
        total_latency_ms: float,
        tokens_generated: int,
    ) -> None:
        REQUESTS_TOTAL.labels(model_id=model_id, status="completed").inc()
        REQUESTS_IN_FLIGHT.labels(model_id=model_id).dec()
        TTFT_HISTOGRAM.labels(model_id=model_id).observe(ttft_ms)
        TOTAL_LATENCY_HISTOGRAM.labels(model_id=model_id).observe(total_latency_ms)
        TOKENS_GENERATED.labels(model_id=model_id).inc(tokens_generated)

        if total_latency_ms > 0:
            tps = tokens_generated / (total_latency_ms / 1000)
            TOKENS_PER_SECOND.labels(model_id=model_id).set(tps)

    def record_request_rejected(self, model_id: str, reason: str) -> None:
        REQUESTS_TOTAL.labels(model_id=model_id, status="rejected").inc()
        REQUESTS_IN_FLIGHT.labels(model_id=model_id).dec()

    def record_batch_iteration(
        self, model_id: str, batch_size: int, total_tokens: int
    ) -> None:
        BATCH_SIZE_HISTOGRAM.labels(model_id=model_id).observe(batch_size)
        BATCH_TOKENS_HISTOGRAM.labels(model_id=model_id).observe(total_tokens)

    def record_gpu_state(
        self,
        device_id: str,
        gpu_type: str,
        memory_util: float,
        compute_util: float,
    ) -> None:
        GPU_MEMORY_UTILIZATION.labels(device_id=device_id, gpu_type=gpu_type).set(memory_util)
        GPU_COMPUTE_UTILIZATION.labels(device_id=device_id, gpu_type=gpu_type).set(compute_util)

    def record_kv_cache_stats(
        self,
        device_id: str,
        utilization: float,
        prefix_hit_rate: float,
    ) -> None:
        KV_CACHE_UTILIZATION.labels(device_id=device_id).set(utilization)
        KV_CACHE_PREFIX_HIT_RATE.set(prefix_hit_rate)

    def record_preemption(self, reason: str) -> None:
        SCHEDULER_PREEMPTIONS.labels(reason=reason).inc()

    def record_speculative_step(
        self, model_id: str, acceptance_rate: float, tokens_saved: int
    ) -> None:
        SPECULATIVE_ACCEPTANCE_RATE.labels(model_id=model_id).set(acceptance_rate)
        SPECULATIVE_TOKENS_SAVED.labels(model_id=model_id).inc(tokens_saved)

    def set_system_info(self, version: str, num_models: int, num_gpus: int) -> None:
        SYSTEM_INFO.info({
            "version": version,
            "num_models": str(num_models),
            "num_gpus": str(num_gpus),
        })
