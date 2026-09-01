"""Ingest executed episodes into the lifecycle-aware memory store."""

from __future__ import annotations

from dataclasses import dataclass

from remem.execution import EpisodeResult

from .deduplication import MemoryDeduplicator
from .episode import EpisodeMemoryAttribution, EpisodeMemoryRecorder
from .store import MemoryStore
from .types import MemoryRecord


@dataclass(frozen=True, slots=True)
class MemoryIngestionResult:
    """Summary of memories proposed and retained during ingestion."""

    proposed_memories: tuple[MemoryRecord, ...]
    retained_memories: tuple[MemoryRecord, ...]


class EpisodeMemoryIngestor:
    """Convert episode trajectories into deduplicated stored memories."""

    def __init__(
        self,
        recorder: EpisodeMemoryRecorder | None = None,
        deduplicator: MemoryDeduplicator | None = None,
    ) -> None:
        """Create an ingestor from replaceable deterministic components."""

        self.recorder = recorder or EpisodeMemoryRecorder()
        self.deduplicator = deduplicator or MemoryDeduplicator()

    def ingest(
        self,
        store: MemoryStore,
        *,
        episode_id: str,
        episode: EpisodeResult,
        attribution: EpisodeMemoryAttribution,
    ) -> MemoryIngestionResult:
        """Record, deduplicate, and add new memories to ``store``.

        Existing memories are treated as the authoritative retained set for
        duplicate detection. A duplicate candidate is not inserted; this
        keeps ingestion idempotent for repeated episode identifiers and
        prevents the store from accumulating semantically redundant records.
        """

        proposed = self.recorder.record(episode_id, episode, attribution)
        retained: list[MemoryRecord] = []
        existing = store.all()
        for memory in self.deduplicator.deduplicate(proposed):
            if self.deduplicator.is_duplicate(memory, existing + retained):
                continue
            retained.append(memory)
            store.add(memory)

        return MemoryIngestionResult(
            proposed_memories=tuple(proposed),
            retained_memories=tuple(retained),
        )
