"""Memory-guided policy composition primitives.

The policy adapter keeps action generation separate from retrieval and
reconstruction. A learned or hand-written action policy receives the current
state together with deterministic memory guidance selected by the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

from .pipeline import MemoryGuidancePipeline
from .store import MemoryStore

GuidedActionPolicy = Callable[[str, str], str]
QueryBuilder = Callable[[str], str]


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
        if not 0.0 <= minimum_trust <= 1.0:
            raise ValueError("minimum_trust must be between 0 and 1")
        self.store = store
        self.action_policy = action_policy
        self.pipeline = pipeline or MemoryGuidancePipeline()
        self.query_builder = query_builder or _default_query_builder
        self.minimum_trust = minimum_trust

    def __call__(self, current_state: str) -> str:
        """Generate an action after retrieving state-aligned memory guidance."""
        if not current_state.strip():
            raise ValueError("current_state must not be empty")

        query = self.query_builder(current_state)
        if not query.strip():
            raise ValueError("query_builder must return a non-empty query")

        candidates = self.pipeline.build_candidates(
            self.store,
            query=query,
            current_state=current_state,
            minimum_trust=self.minimum_trust,
        )
        guidance = candidates[0].reconstruction.guidance if candidates else ""
        action = self.action_policy(current_state, guidance)
        if not action.strip():
            raise ValueError("action_policy must return a non-empty action")
        return action


def _default_query_builder(current_state: str) -> str:
    """Use the current state itself as the retrieval query."""
    return current_state.strip()
