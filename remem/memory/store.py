"""Deterministic memory-store abstraction for local experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from .types import MemoryKind, MemoryRecord, RetrievedMemory


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Controls candidate selection without coupling to an embedding backend."""

    top_k: int = 5
    minimum_similarity: float = 0.0
    include_failures: bool = True


class InMemoryStore:
    """Small deterministic store used by tests and synthetic benchmarks."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def add(self, memory: MemoryRecord) -> None:
        """Insert or replace a memory by its stable identifier."""
        if not memory.memory_id.strip():
            raise ValueError("memory_id must not be empty")
        self._records[memory.memory_id] = memory

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Return a memory by identifier."""
        return self._records.get(memory_id)

    def remove(self, memory_id: str) -> bool:
        """Remove a memory and report whether it existed."""
        return self._records.pop(memory_id, None) is not None

    def all(self) -> list[MemoryRecord]:
        """Return all memories in insertion order."""
        return list(self._records.values())

    def retrieve(
        self,
        similarity_scores: dict[str, float],
        policy: RetrievalPolicy | None = None,
    ) -> list[RetrievedMemory]:
        """Rank supplied candidates by similarity and return the top matches."""
        active_policy = policy or RetrievalPolicy()
        if active_policy.top_k <= 0:
            return []

        candidates: list[tuple[float, MemoryRecord]] = []
        for memory_id, similarity in similarity_scores.items():
            memory = self._records.get(memory_id)
            if memory is None or similarity < active_policy.minimum_similarity:
                continue
            if memory.kind == MemoryKind.FAILURE and not active_policy.include_failures:
                continue
            candidates.append((float(similarity), memory))

        candidates.sort(key=lambda item: (-item[0], item[1].memory_id))
        return [
            RetrievedMemory(memory=memory, similarity=similarity)
            for similarity, memory in candidates[: active_policy.top_k]
        ]

    def extend(self, memories: Iterable[MemoryRecord]) -> None:
        """Insert multiple memories using the same validation path as ``add``."""
        for memory in memories:
            self.add(memory)
