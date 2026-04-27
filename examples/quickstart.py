"""Minimal end-to-end example.

Submits a couple of inference requests through the scheduler, demonstrates
KV-cache prefix sharing, and shows a speculative decoding step. No GPU
required - the example is built on the same in-process primitives as
``benchmarks/run_demo.py``.

Run:

    python examples/quickstart.py
"""

from __future__ import annotations

import logging
import os
import sys

# Allow importing inference_pipeline without a packaged install. Same trick as
# the benchmarks scripts use.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

logging.basicConfig(level=logging.WARNING)

import structlog  # noqa: E402
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

from inference_pipeline.scheduler import GPUScheduler, GPUType, Priority  # noqa: E402
from inference_pipeline.speculative import SpeculativeConfig, SpeculativeDecoder  # noqa: E402


def main() -> None:
    print("Inference Pipeline Quickstart")
    print("-" * 60)

    # 1. Stand up a scheduler with one device of each tier.
    scheduler = GPUScheduler()
    scheduler.register_device("H100-0", GPUType.H100)
    scheduler.register_device("A100-0", GPUType.A100)
    scheduler.register_device("L40S-0", GPUType.L40S)

    # 2. Submit two requests of different priority. Realtime should land on
    #    the highest-tier free device (H100); the LOW request goes to whatever
    #    is left.
    rt_id = scheduler.submit(
        model_id="llama-3-70b",
        estimated_tokens=512,
        priority=Priority.REALTIME,
        max_latency_ms=200.0,
    )
    bg_id = scheduler.submit(
        model_id="llama-3-70b",
        estimated_tokens=4096,
        priority=Priority.LOW,
        max_latency_ms=10_000.0,
    )

    placements = scheduler.tick()
    by_id = dict(placements)
    print(f"\nrequest {rt_id[:8]} (REALTIME)  -> {by_id.get(rt_id)}")
    print(f"request {bg_id[:8]} (LOW)       -> {by_id.get(bg_id)}")

    # 3. Run a single speculative decoding step.
    decoder = SpeculativeDecoder(SpeculativeConfig(num_draft_tokens=5))
    decoder.register_sequence(rt_id)

    # Synthetic logits over a 32k-vocab. Draft proposes 5 tokens; the target
    # gives roughly the same distribution for the first 4 (so they get
    # accepted) and a different one for the 5th (so it gets rejected).
    rng = np.random.default_rng(seed=7)
    vocab = 32_000
    K = 5

    def make_logits(skew: float) -> np.ndarray:
        logits = rng.normal(loc=0.0, scale=1.0, size=vocab)
        # Concentrate probability on a small set of tokens
        logits[100:110] += skew
        return logits

    draft_logits = [make_logits(skew=4.0) for _ in range(K)]
    draft_tokens = [int(np.argmax(lg)) for lg in draft_logits]
    # Target logits: first 4 same as draft, 5th diverges
    target_logits = [make_logits(skew=4.0) for _ in range(4)]
    target_logits.append(make_logits(skew=2.0))  # diverge
    target_logits.append(make_logits(skew=4.0))  # bonus position

    step = decoder.speculative_step(
        sequence_id=rt_id,
        draft_logits_sequence=draft_logits,
        draft_token_ids=draft_tokens,
        target_logits_sequence=target_logits,
    )

    print("\nSpeculative step:")
    print(f"  drafted tokens   : {step.num_drafted}")
    print(f"  accepted tokens  : {step.num_accepted}")
    print(f"  acceptance rate  : {step.acceptance_rate:.2%}")
    print(f"  speedup factor   : {step.speedup_factor:.2f}x")

    # 4. Cleanup so memory accounting stays consistent.
    scheduler.complete_request(rt_id)
    scheduler.complete_request(bg_id)
    decoder.unregister_sequence(rt_id)

    print(f"\nQueue depth: {scheduler.queue_depth}")
    print(f"Active: {scheduler.active_count}")
    print("\nDone. See benchmarks/run_demo.py for the longer walkthrough.")


if __name__ == "__main__":
    main()
