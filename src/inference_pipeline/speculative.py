"""Speculative decoding implementation.

Uses a small draft model to generate candidate token sequences, then verifies
them in parallel with the target model. Accepted tokens skip individual decode
steps, improving throughput by 2-3x for autoregressive generation.

Based on: "Fast Inference from Transformers via Speculative Decoding"
(Leviathan et al., 2023)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TokenSampler(Protocol):
    """Protocol for sampling tokens from a probability distribution."""

    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int: ...
    def log_probs(self, logits: np.ndarray) -> np.ndarray: ...


class ModelForward(Protocol):
    """Protocol for model forward pass."""

    def forward(
        self, token_ids: list[int], past_key_values: object | None = None
    ) -> tuple[np.ndarray, object]:
        """Run forward pass, return (logits, new_past_key_values)."""
        ...


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # 5 draft tokens gives ~2.8x speedup on our workload. going higher tanks
    # acceptance rate because the draft model diverges from target on longer sequences
    num_draft_tokens: int = 5
    temperature: float = 0.8
    top_p: float = 0.95
    # if we get 10 rejections in a row, the draft model is clearly confused about
    # this particular continuation - fall back to standard decode
    max_rejections_before_fallback: int = 10
    acceptance_threshold: float = 0.0
    # let the system learn optimal draft length per-sequence based on accept rates
    adaptive_draft_length: bool = True


@dataclass
class SpeculativeStep:
    """Result of one speculative decoding step."""

    draft_tokens: list[int]
    accepted_tokens: list[int]
    num_drafted: int
    num_accepted: int
    acceptance_rate: float
    wall_time_ms: float
    speedup_factor: float  # Effective tokens per forward pass


@dataclass
class SpeculativeState:
    """Per-sequence state for speculative decoding."""

    sequence_id: str
    total_drafted: int = 0
    total_accepted: int = 0
    consecutive_rejections: int = 0
    current_draft_length: int = 5
    use_speculation: bool = True
    history: list[float] = field(default_factory=list)  # Recent acceptance rates

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def effective_speedup(self) -> float:
        """Average tokens generated per verification step."""
        if self.total_drafted == 0:
            return 1.0
        # Each step does 1 draft forward + 1 verify forward = 2 forwards
        # for (accepted + 1) tokens
        avg_accepted = self.total_accepted / max(1, self.total_drafted // self.current_draft_length)
        return (avg_accepted + 1) / 2.0


class SpeculativeDecoder:
    """Speculative decoding engine.

    Workflow per step:
    1. Draft model generates K candidate tokens autoregressively
    2. Target model verifies all K tokens in a single forward pass
    3. Accept tokens where target probability >= draft probability (with correction)
    4. Sample a bonus token from the adjusted distribution at the rejection point
    """

    def __init__(self, config: SpeculativeConfig | None = None) -> None:
        self._config = config or SpeculativeConfig()
        self._states: dict[str, SpeculativeState] = {}
        self._total_steps: int = 0
        self._total_tokens_saved: int = 0  # Tokens that didn't need individual decode

    def register_sequence(self, sequence_id: str) -> None:
        """Initialize speculative state for a new sequence."""
        self._states[sequence_id] = SpeculativeState(
            sequence_id=sequence_id,
            current_draft_length=self._config.num_draft_tokens,
        )

    def unregister_sequence(self, sequence_id: str) -> None:
        """Clean up state for a completed sequence."""
        self._states.pop(sequence_id, None)

    def should_speculate(self, sequence_id: str) -> bool:
        """Determine if speculation should be attempted for this sequence."""
        state = self._states.get(sequence_id)
        if state is None:
            return False
        return state.use_speculation

    def speculative_step(
        self,
        sequence_id: str,
        draft_logits_sequence: list[np.ndarray],
        draft_token_ids: list[int],
        target_logits_sequence: list[np.ndarray],
        temperature: float | None = None,
    ) -> SpeculativeStep:
        """Execute one speculative decoding verification step.

        Args:
            sequence_id: The sequence being decoded.
            draft_logits_sequence: Logits from draft model for each drafted position.
            draft_token_ids: Token IDs sampled from draft model.
            target_logits_sequence: Logits from target model for all positions (batch verify).
            temperature: Sampling temperature override.

        Returns:
            SpeculativeStep with accepted tokens and metrics.
        """
        start_time = time.monotonic()
        temp = temperature or self._config.temperature
        state = self._states.get(sequence_id)
        if state is None:
            state = SpeculativeState(sequence_id=sequence_id)
            self._states[sequence_id] = state

        num_draft = len(draft_token_ids)
        accepted_tokens: list[int] = []

        # Verify each drafted token using rejection sampling
        for i in range(num_draft):
            draft_probs = self._softmax(draft_logits_sequence[i] / temp)
            target_probs = self._softmax(target_logits_sequence[i] / temp)

            draft_token = draft_token_ids[i]
            p_target = target_probs[draft_token]
            p_draft = draft_probs[draft_token]

            # Accept if target probability is at least as high as draft probability
            if p_draft <= 0:
                # Draft assigned zero probability, always reject
                break

            # core insight from Leviathan et al.: accept with probability min(1, p_t/p_d)
            # this preserves the target distribution exactly, no approximation
            acceptance_ratio = min(1.0, p_target / p_draft)

            # stochastic acceptance - this is what makes speculative decoding lossless
            if np.random.random() < acceptance_ratio:
                accepted_tokens.append(draft_token)
            else:
                # Rejection: sample from residual distribution
                residual = np.maximum(target_probs - draft_probs, 0.0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    residual /= residual_sum
                    bonus_token = int(np.random.choice(len(residual), p=residual))
                    accepted_tokens.append(bonus_token)
                else:
                    # Fall back to target distribution
                    bonus_token = int(np.random.choice(len(target_probs), p=target_probs))
                    accepted_tokens.append(bonus_token)
                break

        # If all draft tokens accepted, sample one more from target at position K+1
        if len(accepted_tokens) == num_draft and len(target_logits_sequence) > num_draft:
            final_probs = self._softmax(target_logits_sequence[num_draft] / temp)
            bonus = int(np.random.choice(len(final_probs), p=final_probs))
            accepted_tokens.append(bonus)

        # Update state
        num_accepted = len(accepted_tokens)
        state.total_drafted += num_draft
        state.total_accepted += num_accepted
        self._total_steps += 1
        self._total_tokens_saved += max(0, num_accepted - 1)

        # Track consecutive rejections for fallback
        acceptance_rate = num_accepted / max(1, num_draft)
        state.history.append(acceptance_rate)
        if len(state.history) > 20:
            state.history = state.history[-20:]

        if num_accepted == 0:
            state.consecutive_rejections += 1
        else:
            state.consecutive_rejections = 0

        # Adaptive draft length
        if self._config.adaptive_draft_length:
            self._adapt_draft_length(state)

        # Disable speculation if consistently poor
        if state.consecutive_rejections >= self._config.max_rejections_before_fallback:
            state.use_speculation = False
            logger.info(
                "speculation_disabled",
                sequence_id=sequence_id,
                acceptance_rate=state.acceptance_rate,
            )

        wall_time_ms = (time.monotonic() - start_time) * 1000
        # Speedup: tokens generated / forward passes used (1 draft + 1 verify = 2)
        speedup = num_accepted / 2.0 if num_accepted > 0 else 0.5

        return SpeculativeStep(
            draft_tokens=draft_token_ids,
            accepted_tokens=accepted_tokens,
            num_drafted=num_draft,
            num_accepted=num_accepted,
            acceptance_rate=acceptance_rate,
            wall_time_ms=wall_time_ms,
            speedup_factor=speedup,
        )

    def _adapt_draft_length(self, state: SpeculativeState) -> None:
        """Dynamically adjust draft length based on recent acceptance rates."""
        if len(state.history) < 5:
            return

        recent_rate = np.mean(state.history[-5:])

        if recent_rate > 0.8:
            # High acceptance, try drafting more tokens
            state.current_draft_length = min(
                self._config.num_draft_tokens * 2, state.current_draft_length + 1
            )
        elif recent_rate < 0.3:
            # Low acceptance, reduce draft length
            state.current_draft_length = max(1, state.current_draft_length - 1)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum()

    def get_draft_length(self, sequence_id: str) -> int:
        """Get the current draft length for a sequence."""
        state = self._states.get(sequence_id)
        if state is None:
            return self._config.num_draft_tokens
        return state.current_draft_length

    @property
    def stats(self) -> dict[str, int | float]:
        total_drafted = sum(s.total_drafted for s in self._states.values())
        total_accepted = sum(s.total_accepted for s in self._states.values())
        return {
            "active_sequences": len(self._states),
            "total_steps": self._total_steps,
            "total_tokens_saved": self._total_tokens_saved,
            "global_acceptance_rate": total_accepted / max(1, total_drafted),
            "avg_speedup": self._total_tokens_saved / max(1, self._total_steps) + 1.0,
        }
