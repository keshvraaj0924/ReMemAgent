"""Deterministic retrieval primitives for memory experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import MemoryRecord, RetrievedMemory

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Controls deterministic lexical retrieval behavior."""

    top_k: int = 3
    minimum_similarity: float = 0.0
    include_failures: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")


class MemoryRetriever:
    """Retrieve candidate memories without requiring an embedding service."""

    def __init__(self, policy: RetrievalPolicy | None = None) -> None:
        self.policy = policy or RetrievalPolicy()

    def retrieve(self, query: str, memories: list[MemoryRecord]) -> list[RetrievedMemory]:
        """Return the highest-overlap memories using deterministic ranking."""
        if not query.strip():
            raise ValueError("query must not be empty")

        candidates: list[RetrievedMemory] = []
        for memory in memories:
            if not self.policy.include_failures and memory.kind.value == "failure":
                continue
            similarity = lexical_similarity(query, self._search_text(memory))
            if similarity >= self.policy.minimum_similarity:
                candidates.append(RetrievedMemory(memory=memory, similarity=similarity))

        candidates.sort(
            key=lambda candidate: (-candidate.similarity, candidate.memory.memory_id)
        )
        return candidates[: self.policy.top_k]

    @staticmethod
    def _search_text(memory: MemoryRecord) -> str:
        return " ".join((memory.state, memory.action, memory.outcome, memory.metadata.get("summary", "")))


def lexical_similarity(left_text: str, right_text: str) -> float:
    """Calculate Jaccard similarity over normalized word tokens."""
    left_tokens = set(_TOKEN_PATTERN.findall(left_text.lower()))
    right_tokens = set(_TOKEN_PATTERN.findall(right_text.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
