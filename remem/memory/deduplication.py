"""Memory deduplication primitives.

The default implementation uses token-set Jaccard similarity so the research
engine remains dependency-free. Production vector backends can implement the
same protocol with embedding similarity.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .types import MemoryRecord

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def normalized_tokens(text: str) -> set[str]:
    """Tokenize text into normalized lexical units."""
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def lexical_similarity(left_text: str, right_text: str) -> float:
    """Return token-set Jaccard similarity in the inclusive range [0, 1]."""
    left_tokens = normalized_tokens(left_text)
    right_tokens = normalized_tokens(right_text)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class MemoryDeduplicator:
    """Keep only memories that are sufficiently distinct from existing ones."""

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.similarity_threshold = similarity_threshold

    def is_duplicate(self, candidate: MemoryRecord, existing_memories: Sequence[MemoryRecord]) -> bool:
        """Check whether candidate content is already represented."""
        return any(
            lexical_similarity(candidate.content, existing_memory.content) >= self.similarity_threshold
            for existing_memory in existing_memories
        )
