"""Route between memory-guided, hybrid, and self-reasoning paths."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from ..memory.types import MemoryRecord

class Route(str, Enum):
    MEMORY = "memory"
    SELF = "self"
    HYBRID = "hybrid"

@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    route: Route
    with_memory: float
    without_memory: float
    expected_delta: float
    confidence: float

class CounterfactualRouter:
    def __init__(self, min_benefit: float = 0.05, hybrid_margin: float = 0.08) -> None:
        if min_benefit < 0 or hybrid_margin < 0:
            raise ValueError("routing thresholds must be non-negative")
        self.min_benefit = min_benefit
        self.hybrid_margin = hybrid_margin

    def decide(self, memory: MemoryRecord, context_alignment: float, reconstructed_quality: float, self_score: float) -> CounterfactualResult:
        alignment = max(0.0, min(1.0, context_alignment))
        quality = max(0.0, min(1.0, reconstructed_quality))
        base = max(0.0, min(1.0, self_score))
        memory_score = 0.35 * alignment + 0.30 * quality + 0.20 * memory.empirical_success + 0.15 * memory.transferability
        delta = memory_score - base
        confidence = min(1.0, 0.5 * abs(delta) + 0.5 * memory.confidence)
        if delta < self.min_benefit:
            route = Route.SELF
        elif delta < self.hybrid_margin:
            route = Route.HYBRID
        else:
            route = Route.MEMORY
        return CounterfactualResult(route, memory_score, base, delta, confidence)
