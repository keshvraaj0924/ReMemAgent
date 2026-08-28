"""Composable deterministic pipeline for memory-guided decisions.

The pipeline intentionally separates retrieval, trust estimation, and
reconstruction. Each component can later be replaced by a learned model
without changing the orchestration contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from .reconstruction import MemoryReconstructor, Reconstruction
from .retrieval import MemoryRetriever
from .store import MemoryStore
from .trust import MemoryTrustScorer, TrustScore


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A retrieved memory with its trust assessment and reconstruction."""

    memory_id: str
    similarity: float
    trust: TrustScore
    reconstruction: Reconstruction


class MemoryGuidancePipeline:
    """Retrieve, score, and reconstruct memories for a current state."""

    def __init__(
        self,
        retriever: MemoryRetriever | None = None,
        trust_scorer: MemoryTrustScorer | None = None,
        reconstructor: MemoryReconstructor | None = None,
    ) -> None:
        self.retriever = retriever or MemoryRetriever()
        self.trust_scorer = trust_scorer or MemoryTrustScorer()
        self.reconstructor = reconstructor or MemoryReconstructor()

    def build_candidates(
        self,
        store: MemoryStore,
        *,
        query: str,
        current_state: str,
        minimum_trust: float = 0.0,
    ) -> list[MemoryCandidate]:
        """Build deterministic guidance candidates from active memories."""
        if not 0.0 <= minimum_trust <= 1.0:
            raise ValueError("minimum_trust must be between 0 and 1")

        retrieved = self.retriever.retrieve(query, store.active())
        candidates: list[MemoryCandidate] = []
        for item in retrieved:
            trust = self.trust_scorer.score(item.memory, similarity=item.similarity)
            reconstruction = self.reconstructor.reconstruct(
                item.memory,
                current_state,
                context_alignment=item.similarity,
            )
            if reconstruction.rejected or trust.confidence < minimum_trust:
                continue
            candidates.append(
                MemoryCandidate(
                    memory_id=item.memory.memory_id,
                    similarity=item.similarity,
                    trust=trust,
                    reconstruction=reconstruction,
                )
            )

        candidates.sort(key=lambda candidate: (-candidate.trust.confidence, candidate.memory_id))
        return candidates
