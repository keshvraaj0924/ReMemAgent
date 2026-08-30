"""Deterministic in-memory storage for research and unit-test workloads."""

from __future__ import annotations

from collections.abc import Iterable

from .types import MemoryRecord, MemoryStatus, MemoryTransferOutcome


class MemoryStore:
    """Store memory records with explicit lifecycle-aware operations."""

    def __init__(self, memories: Iterable[MemoryRecord] | None = None) -> None:
        self._memories: dict[str, MemoryRecord] = {}
        if memories is not None:
            for memory in memories:
                self.add(memory)

    def add(self, memory: MemoryRecord) -> None:
        """Insert a memory, rejecting duplicate identifiers."""
        if memory.memory_id in self._memories:
            raise ValueError(f"Memory '{memory.memory_id}' already exists")
        self._memories[memory.memory_id] = memory

    def upsert(self, memory: MemoryRecord) -> None:
        """Insert or replace a memory by identifier."""
        self._memories[memory.memory_id] = memory

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Return a memory by identifier, or ``None`` when absent."""
        return self._memories.get(memory_id)

    def record_transfer_outcome(self, memory_id: str, success: bool) -> None:
        """Record an observed transfer outcome for a stored memory."""

        self.record_transfer_observation(
            MemoryTransferOutcome(memory_id=memory_id, success=success)
        )

    def record_transfer_observation(self, outcome: MemoryTransferOutcome) -> None:
        """Record a typed transfer observation against a stored memory.

        The typed boundary allows environment and benchmark adapters to carry
        episode identity without coupling those integrations to ``MemoryRecord``.
        Unknown memory IDs are rejected so attribution cannot silently disappear.
        """

        memory = self.get(outcome.memory_id)
        if memory is None:
            raise KeyError(f"Memory '{outcome.memory_id}' does not exist")
        memory.record_use(success=outcome.success, transferred=True)

    def remove(self, memory_id: str) -> MemoryRecord | None:
        """Remove and return a memory when it exists."""
        return self._memories.pop(memory_id, None)

    def active(self) -> list[MemoryRecord]:
        """Return only active memories in deterministic identifier order."""
        return self._by_status(MemoryStatus.ACTIVE)

    def all(self) -> list[MemoryRecord]:
        """Return all memories in deterministic identifier order."""
        return sorted(self._memories.values(), key=lambda memory: memory.memory_id)

    def by_status(self, status: MemoryStatus) -> list[MemoryRecord]:
        """Return memories matching an explicit lifecycle status."""
        return self._by_status(status)

    def _by_status(self, status: MemoryStatus) -> list[MemoryRecord]:
        return sorted(
            (memory for memory in self._memories.values() if memory.status is status),
            key=lambda memory: memory.memory_id,
        )

    def __len__(self) -> int:
        return len(self._memories)
