"""Route between memory-guided, hybrid, and self-reasoning paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..memory.types import MemoryRecord


class Route(str, Enum):
    """Available reasoning routes for an adaptive memory decision."""

    MEMORY = "memory"
    SELF = "self"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    """Scores and route selected by the heuristic counterfactual router."""

    route: Route
    with_memory: float
    without_memory: float
    expected_delta: float
    confidence: float


class CounterfactualRouter:
    """Heuristically route requests using bounded memory-quality evidence."""

    def __init__(self, min_benefit: float = 0.05, hybrid_margin: float = 0.08) -> None:
        if min_benefit < 0 or hybrid_margin < 0:
            raise ValueError("routing thresholds must be non-negative")
        self.min_benefit = min_benefit
        self.hybrid_margin = hybrid_margin

    def decide(
        self,
        memory: MemoryRecord,
        context_alignment: float,
        reconstructed_quality: float,
        self_score: float,
    ) -> CounterfactualResult:
        """Estimate memory utility and choose memory, hybrid, or self reasoning."""

        alignment = _bounded_score(context_alignment)
        quality = _bounded_score(reconstructed_quality)
        base = _bounded_score(self_score)
        memory_score = (
            0.35 * alignment
            + 0.30 * quality
            + 0.20 * memory.empirical_success_rate
            + 0.15 * memory.transferability
        )
        delta = memory_score - base
        confidence = min(1.0, 0.5 * abs(delta) + 0.5 * memory.confidence)

        if delta < self.min_benefit:
            route = Route.SELF
        elif delta < self.hybrid_margin:
            route = Route.HYBRID
        else:
            route = Route.MEMORY

        return CounterfactualResult(route, memory_score, base, delta, confidence)


def _bounded_score(value: float) -> float:
    """Clamp a heuristic score to the unit interval."""

    return max(0.0, min(1.0, value))
