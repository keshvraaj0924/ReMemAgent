"""Memory-guided policy composition primitives.

The policy adapter keeps action generation separate from retrieval and
reconstruction. A learned or hand-written action policy receives the current
state together with deterministic memory guidance selected by the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from .pipeline import MemoryCandidate, MemoryGuidancePipeline
from .store import MemoryStore

GuidedActionPolicy = Callable[[str, str], str]
QueryBuilder = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class MemoryGuidanceDecision:
    """Traceable memory-selection result supplied to an action policy."""

    memory_id: str | None
    guidance: str
    similarity: float
    trust_confidence: float


class MemoryGuidedPolicy:
    """Compose memory guidance with an environment-facing action policy."""

    def __init__(
        self,
        store: MemoryStore,
        action_policy: GuidedActionPolicy,
        *,
        pipeline: MemoryGuidancePipeline | None = None,
        query_builder: QueryBuilder | None = None,
        minimum_trust: float = 0.0,
    ) -> None:
        """Create a policy with validated research and learned-component boundaries."""

        if not isinstance(store, MemoryStore):
            raise TypeError("store must be a MemoryStore")
        if not callable(action_policy):
            raise TypeError("action_policy must be callable")
        if pipeline is not None and not isinstance(pipeline, MemoryGuidancePipeline):
            raise TypeError("pipeline must be a MemoryGuidancePipeline when provided")
        if query_builder is not None and not callable(query_builder):
            raise TypeError("query_builder must be callable when provided")
        if isinstance(minimum_trust, bool) or not isinstance(minimum_trust, (int, float)):
            raise TypeError("minimum_trust must be a finite number between 0 and 1")
        if not math.isfinite(float(minimum_trust)):
            raise ValueError("minimum_trust must be finite")
        if not 0.0 <= minimum_trust <= 1.0:
            raise ValueError("minimum_trust must be between 0 and 1")
        self.store = store
        self.action_policy = action_policy
        self.pipeline = pipeline or MemoryGuidancePipeline()
        self.query_builder = query_builder or _default_query_builder
        self.minimum_trust = minimum_trust
        self._decision_history: list[MemoryGuidanceDecision] = []

    @property
    def decision_history(self) -> tuple[MemoryGuidanceDecision, ...]:
        """Return immutable decisions made since this policy was created."""

        return tuple(self._decision_history)

    def select_guidance(self, current_state: str) -> MemoryGuidanceDecision:
        """Retrieve and reconstruct the strongest trusted guidance candidate."""
        if not isinstance(current_state, str) or not current_state.strip():
            raise ValueError("current_state must be a non-empty string")

        query = self.query_builder(current_state)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query_builder must return a non-empty string")

        candidates = self.pipeline.build_candidates(
            self.store,
            query=query,
            current_state=current_state,
            minimum_trust=self.minimum_trust,
        )
        if not candidates:
            decision = MemoryGuidanceDecision(
                memory_id=None,
                guidance="",
                similarity=0.0,
                trust_confidence=0.0,
            )
        else:
            decision = _decision_from_candidate(candidates[0])
        self._decision_history.append(decision)
        return decision

    def __call__(self, current_state: str) -> str:
        """Generate an action after retrieving state-aligned memory guidance."""
        decision = self.select_guidance(current_state)
        action = self.action_policy(current_state, decision.guidance)
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action_policy must return a non-empty string")
        return action


def _decision_from_candidate(candidate: MemoryCandidate) -> MemoryGuidanceDecision:
    """Convert an internal pipeline candidate into the public trace contract."""

    return MemoryGuidanceDecision(
        memory_id=candidate.memory_id,
        guidance=candidate.reconstruction.guidance,
        similarity=candidate.similarity,
        trust_confidence=candidate.trust.confidence,
    )


def _default_query_builder(current_state: str) -> str:
    """Use the current state itself as the retrieval query."""
    return current_state.strip()
