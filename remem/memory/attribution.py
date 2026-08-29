"""Outcome attribution for memory-guided transfer attempts."""

from __future__ import annotations

from dataclasses import dataclass

from .policy import MemoryGuidanceDecision
from .store import MemoryStore


@dataclass(frozen=True, slots=True)
class MemoryTransferOutcome:
    """Observed outcome of one action decision informed by a memory."""

    memory_id: str
    success: bool


class MemoryTransferRecorder:
    """Attribute observed outcomes to memories that supplied guidance."""

    def record(
        self,
        store: MemoryStore,
        decision: MemoryGuidanceDecision,
        *,
        success: bool,
    ) -> MemoryTransferOutcome | None:
        """Record a transfer attempt when a concrete memory was selected.

        A missing memory selection is intentionally a no-op: self-reasoning
        actions must not inflate transfer statistics. The selected record is
        updated in place so its lifecycle and trust computations immediately
        observe the new evidence.
        """

        if decision.memory_id is None:
            return None
        memory = store.get(decision.memory_id)
        if memory is None:
            raise KeyError(f"Memory '{decision.memory_id}' does not exist")
        memory.record_use(success=success, transferred=True)
        return MemoryTransferOutcome(memory_id=memory.memory_id, success=success)
