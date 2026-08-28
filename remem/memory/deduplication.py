"""Deterministic semantic-deduplication baseline for memory candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import lexical_similarity
from .types import MemoryRecord


@dataclass(frozen=True, slots=True)
class DeduplicationPolicy:
    """Controls when two memories are considered near-duplicates."""

    similarity_threshold: float = 0.85

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")


class MemoryDeduplicator:
    """Remove redundant memories while preserving the strongest evidence."""

    def __init__(self, policy: DeduplicationPolicy | None = None) -> None:
        self.policy = policy or DeduplicationPolicy()

    def deduplicate(self, memories: list[MemoryRecord]) -> list[MemoryRecord]:
        """Return memories with near-duplicate records removed.

        Records are considered in deterministic identifier order. When two
        records overlap beyond the configured threshold, the record with the
        stronger empirical evidence is retained.
        """
        retained: list[MemoryRecord] = []
        for candidate in sorted(memories, key=lambda memory: memory.memory_id):
            duplicate_index = self._find_duplicate_index(candidate, retained)
            if duplicate_index is None:
                retained.append(candidate)
                continue
            if self._evidence_score(candidate) > self._evidence_score(retained[duplicate_index]):
                retained[duplicate_index] = candidate
        return retained

    def _find_duplicate_index(
        self,
        candidate: MemoryRecord,
        retained: list[MemoryRecord],
    ) -> int | None:
        for index, existing in enumerate(retained):
            similarity = lexical_similarity(
                self._memory_text(candidate),
                self._memory_text(existing),
            )
            if similarity >= self.policy.similarity_threshold:
                return index
        return None

    @staticmethod
    def _memory_text(memory: MemoryRecord) -> str:
        return " ".join((memory.state, memory.action, memory.outcome))

    @staticmethod
    def _evidence_score(memory: MemoryRecord) -> float:
        return memory.reward + memory.empirical_success_rate + 0.01 * memory.uses
