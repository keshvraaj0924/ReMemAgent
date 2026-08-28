"""Deterministic in-memory storage for research and unit-test workloads."""

from __future__ import annotations

from collections.abc import Iterable

from .types import MemoryRecord


class MemoryStore:
    """Store memory records with explicit lifecycle operations."""

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

    def remove(self, memory_id: str) -> MemoryRecord | None:
        """Remove and return a memory when it exists."""
        return self._memories.pop(memory_id, None)

    def active(self) -> list[MemoryRecord]:
        """Return active memories in deterministic identifier order."""
        return sorted(
            (memory for memory in self._memories.values() if memory.kind is not None),
            key=lambda memory: memory.memory_id,
        )

    def all(self) -> list[MemoryRecord]:
        """Return all memories in deterministic identifier order."""
        return sorted(self._memories.values(), key=lambda memory: memory.memory_id)

    def __len__(self) -> int:
        return len(self._memories)
