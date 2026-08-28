"""Context-aware trust and transferability scoring.

The baseline scorer is deliberately model-agnostic. It exposes a stable
contract that can later be replaced or augmented by a learned critic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import MemoryRecord


@dataclass(frozen=True, slots=True)
class TrustScore:
    """Decomposed trust estimate for one memory candidate."""

    similarity: float
    historical_success: float
    transferability: float
    freshness: float
    confidence: float


class MemoryTrustScorer:
    """Estimate whether a memory deserves to influence current reasoning."""

    def __init__(
        self,
        similarity_weight: float = 0.35,
        success_weight: float = 0.30,
        transfer_weight: float = 0.20,
        freshness_weight: float = 0.15,
    ) -> None:
        weights = (
            similarity_weight,
            success_weight,
            transfer_weight,
            freshness_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Trust weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("At least one trust weight must be positive")
        total_weight = sum(weights)
        self._weights = tuple(weight / total_weight for weight in weights)

    def score(
        self,
        memory: MemoryRecord,
        *,
        similarity: float,
        transferability: float | None = None,
        freshness: float = 1.0,
    ) -> TrustScore:
        """Return a deterministic trust decomposition for ``memory``."""
        similarity_score = _unit(similarity)
        historical_success = _unit(memory.empirical_success_rate)
        transferability_score = _unit(
            memory.transferability if transferability is None else transferability
        )
        freshness_score = _unit(freshness)
        component_scores = (
            similarity_score,
            historical_success,
            transferability_score,
            freshness_score,
        )
        confidence = sum(
            weight * value for weight, value in zip(self._weights, component_scores)
        )
        return TrustScore(
            similarity=similarity_score,
            historical_success=historical_success,
            transferability=transferability_score,
            freshness=freshness_score,
            confidence=confidence,
        )


def _unit(value: float) -> float:
    """Clamp an arbitrary scalar to the scoring domain [0, 1]."""
    return max(0.0, min(1.0, float(value)))
