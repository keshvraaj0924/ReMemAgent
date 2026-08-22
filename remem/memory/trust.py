"""Context-aware trust and transferability scoring.

This first implementation is intentionally model-agnostic. It provides a
stable scoring contract that can later be replaced by a learned critic.
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
    """Score whether a retrieved experience deserves to influence reasoning."""

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
        total = sum(weights)
        self._weights = tuple(weight / total for weight in weights)

    def score(
        self,
        memory: MemoryRecord,
        *,
        similarity: float,
        transferability: float | None = None,
        freshness: float = 1.0,
    ) -> TrustScore:
        similarity = _unit(similarity)
        success = _unit(memory.empirical_success_rate)
        transfer = _unit(
            transferability if transferability is not None else memory.metadata.get("transferability", 0.5)
        )
        freshness = _unit(freshness)
        confidence = sum(
            weight * value
            for weight, value in zip(self._weights, (similarity, success, transfer, freshness))
        )
        return TrustScore(similarity, success, transfer, freshness, confidence)


def _unit(value: float) -> float:
    """Clamp an arbitrary scalar to the scoring domain [0, 1]."""

    return max(0.0, min(1.0, float(value)))
