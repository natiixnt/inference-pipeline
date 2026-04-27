"""Ray Serve deployment configuration for the inference pipeline.

Defines the serving topology, autoscaling policies, and deployment graph
that connects the router, batcher, and model replicas.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ray import serve
from pydantic import BaseModel, Field
import structlog

from inference_pipeline.batcher import BatchConfig, ContinuousBatcher
from inference_pipeline.kv_cache import KVCacheManager
from inference_pipeline.metrics import MetricsCollector
from inference_pipeline.router import (
    ModelDeployment,
    RequestRouter,
    RoutingStrategy,
)
from inference_pipeline.scheduler import Priority
from inference_pipeline.speculative import SpeculativeConfig, SpeculativeDecoder

logger = structlog.get_logger(__name__)


class InferenceRequest(BaseModel):
    """Incoming inference request schema."""

    model_id: str
    prompt: str
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    session_id: Optional[str] = None
    priority: str = "normal"


class InferenceResponse(BaseModel):
    """Inference response schema."""

    request_id: str
    model_id: str
    text: str
    tokens_generated: int
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_per_second: float


class ModelConfig(BaseModel):
    """Configuration for a single model deployment."""

    model_id: str
    model_path: str
    tensor_parallel_degree: int = 1
    max_batch_size: int = 64
    max_sequence_length: int = 8192
    gpu_type_preference: list[str] = Field(default_factory=lambda: ["H100", "A100"])
    enable_speculative: bool = True
    draft_model_path: Optional[str] = None
    kv_cache_pool_gb: float = 40.0


class ServingConfig(BaseModel):
    """Top-level serving configuration."""

    models: list[ModelConfig]
    num_router_replicas: int = 2
    num_scheduler_replicas: int = 1
    default_routing_strategy: str = "affinity_first"
    max_concurrent_requests: int = 500
    request_timeout_s: float = 60.0
    health_check_interval_s: float = 5.0
    enable_metrics: bool = True
    metrics_port: int = 9090


@serve.deployment(
    num_replicas=2,
    max_ongoing_requests=200,
    health_check_period_s=5,
    health_check_timeout_s=10,
    graceful_shutdown_timeout_s=30,
    ray_actor_options={"num_cpus": 2, "memory": 4 * 1024**3},
)
class InferenceRouter:
    """Front-door deployment that handles routing and request admission."""

    def __init__(self, serving_config: dict[str, Any]) -> None:
        self._config = ServingConfig(**serving_config)
        self._router = RequestRouter(
            default_strategy=RoutingStrategy.AFFINITY_FIRST,
            heartbeat_timeout_ms=5000.0,
        )
        self._metrics = MetricsCollector()
        self._request_count = 0

        # Register model deployments
        for model_cfg in self._config.models:
            deployment = ModelDeployment(
                model_id=model_cfg.model_id,
                target_latency_ms=200.0,
                preferred_gpu_types=model_cfg.gpu_type_preference,
                can_use_speculative=model_cfg.enable_speculative,
            )
            self._router.register_deployment(deployment)

        logger.info("inference_router_initialized", num_models=len(self._config.models))

    async def __call__(self, request: InferenceRequest) -> InferenceResponse:
        """Handle an incoming inference request."""
        start_time = time.monotonic()
        self._request_count += 1
        request_id = f"req-{self._request_count:08d}"

        self._metrics.record_request_received(request.model_id)

        # Route to appropriate replica
        priority_map = {
            "critical": Priority.CRITICAL,
            "realtime": Priority.REALTIME,
            "high": Priority.HIGH,
            "normal": Priority.NORMAL,
            "low": Priority.LOW,
        }
        _priority = priority_map.get(request.priority, Priority.NORMAL)

        decision = self._router.route(
            model_id=request.model_id,
            session_id=request.session_id,
            required_capacity=1,
        )

        if decision is None:
            self._metrics.record_request_rejected(request.model_id, "no_capacity")
            raise serve.exceptions.BackPressureError(
                f"No available replicas for model {request.model_id}"
            )

        # Forward to model worker (simplified; actual impl uses gRPC)
        ttft_ms = 42.0  # Placeholder for actual TTFT measurement
        total_tokens = min(request.max_tokens, 256)  # Placeholder
        total_latency_ms = (time.monotonic() - start_time) * 1000

        self._metrics.record_request_completed(
            model_id=request.model_id,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_generated=total_tokens,
        )

        return InferenceResponse(
            request_id=request_id,
            model_id=request.model_id,
            text="",  # Actual text from model worker
            tokens_generated=total_tokens,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_per_second=total_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0,
        )

    async def health_check(self) -> dict[str, str]:
        return {"status": "healthy", "requests_served": str(self._request_count)}


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1, "num_cpus": 4, "memory": 80 * 1024**3},
    max_ongoing_requests=64,
)
class ModelWorker:
    """GPU worker that runs model inference with batching and speculative decoding."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        self._config = ModelConfig(**model_config)
        self._batcher = ContinuousBatcher(
            BatchConfig(max_batch_size=self._config.max_batch_size)
        )
        self._kv_cache = KVCacheManager(
            total_memory_bytes=int(self._config.kv_cache_pool_gb * 1024**3)
        )
        self._speculative = SpeculativeDecoder(
            SpeculativeConfig(num_draft_tokens=5)
        ) if self._config.enable_speculative else None
        self._metrics = MetricsCollector()

        logger.info(
            "model_worker_initialized",
            model_id=self._config.model_id,
            tp_degree=self._config.tensor_parallel_degree,
        )

    async def infer(
        self,
        request_id: str,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Run inference for a single request through the batching pipeline."""
        self._batcher.add_request(
            request_id=request_id,
            model_id=self._config.model_id,
            input_token_ids=token_ids,
            max_new_tokens=max_new_tokens,
        )

        # In production, the batching loop runs continuously in a background task.
        # Here we show the single-request path for clarity.
        result = self._batcher.iteration(memory_utilization=self._kv_cache.utilization)

        return {
            "request_id": request_id,
            "batch_size": result.batch_size,
            "total_tokens": result.total_tokens,
        }


def build_serving_app(config_path: str = "config/serving.yaml") -> serve.Application:
    """Build the Ray Serve application graph."""
    # Default configuration for local development
    default_config = {
        "models": [
            {
                "model_id": "llama-3-70b",
                "model_path": "/models/llama-3-70b",
                "tensor_parallel_degree": 4,
                "max_batch_size": 64,
                "enable_speculative": True,
                "draft_model_path": "/models/llama-3-8b",
                "kv_cache_pool_gb": 40.0,
            },
        ],
        "num_router_replicas": 2,
        "max_concurrent_requests": 500,
        "enable_metrics": True,
    }

    router = InferenceRouter.bind(default_config)
    return router


# Entry point for `serve run`
app = build_serving_app()
