"""Failure-memory capture and retrieval helpers.

Failure memories preserve what went wrong, the action involved, and the
state in which the failure was observed. This module contains deterministic
policy only; learned failure prediction belongs behind a separate interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import MemoryKind, MemoryRecord


@dataclass(frozen=True, slots=True)
class FailureObservation:
    """Minimal evidence required to persist a failed interaction."""

    state: str
    action: str
    outcome: str
    reward: float = 0.0
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.state.strip() or not self.action.strip() or not self.outcome.strip():
            raise ValueError("state, action, and outcome must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class FailureMemoryBuilder:
    """Convert failed interaction evidence into typed failure memories."""

    def build(self, memory_id: str, observation: FailureObservation) -> MemoryRecord:
        """Create a failure memory with conservative initial confidence."""
        if not memory_id.strip():
            raise ValueError("memory_id must not be empty")
        return MemoryRecord(
            memory_id=memory_id,
            state=observation.state.strip(),
            action=observation.action.strip(),
            outcome=observation.outcome.strip(),
            kind=MemoryKind.FAILURE,
            reward=observation.reward,
            failures=1,
            confidence=observation.confidence,
            metadata={"failure_type": "observed"},
        )

    @staticmethod
    def update(memory: MemoryRecord, success: bool, transferred: bool = False) -> None:
        """Record later evidence without changing the memory's semantic kind."""
        if memory.kind is not MemoryKind.FAILURE:
            raise ValueError("only failure memories can be updated by FailureMemoryBuilder")
        memory.record_use(success=success, transferred=transferred)
