"""Memory lifecycle operations: scoring, aging, and consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import fsum

from .types import MemoryKind, MemoryRecord, MemoryStatus

SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    """Configuration controlling memory aging and consolidation."""

    stale_after_days: float = 30.0
    retire_after_days: float = 90.0
    minimum_transferability: float = 0.25
    consolidation_threshold: int = 3

    def __post_init__(self) -> None:
        if self.stale_after_days < 0 or self.retire_after_days <= 0:
            raise ValueError("lifecycle durations must be non-negative")
        if self.stale_after_days > self.retire_after_days:
            raise ValueError("stale_after_days cannot exceed retire_after_days")
        if not 0.0 <= self.minimum_transferability <= 1.0:
            raise ValueError("minimum_transferability must be between 0 and 1")
        if self.consolidation_threshold < 2:
            raise ValueError("consolidation_threshold must be at least 2")


class MemoryLifecycle:
    """Manage memory health, status transitions, and consolidation."""

    def __init__(self, policy: LifecyclePolicy | None = None) -> None:
        self.policy = policy or LifecyclePolicy()

    def health_score(self, memory: MemoryRecord, now: datetime | None = None) -> float:
        """Calculate a bounded health score from evidence and freshness."""

        current_time = now or datetime.now(timezone.utc)
        age_days = _elapsed_days(memory.created_at, current_time)
        freshness = max(0.0, 1.0 - age_days / self.policy.retire_after_days)
        score = fsum(
            (
                0.35 * memory.empirical_success_rate,
                0.30 * memory.transferability,
                0.20 * memory.confidence,
                0.15 * freshness,
            )
        )
        return max(0.0, min(1.0, score))

    def refresh_status(self, memory: MemoryRecord, now: datetime | None = None) -> MemoryStatus:
        """Update and return lifecycle status based on usage and transfer evidence."""

        current_time = now or datetime.now(timezone.utc)
        last_used_time = memory.last_used_at or memory.created_at
        idle_days = _elapsed_days(last_used_time, current_time)
        transfer_is_poor = (
            memory.transfer_attempts > 0
            and memory.transferability < self.policy.minimum_transferability
        )

        if idle_days >= self.policy.retire_after_days or transfer_is_poor:
            memory.status = MemoryStatus.RETIRED
        elif idle_days >= self.policy.stale_after_days:
            memory.status = MemoryStatus.STALE
        else:
            memory.status = MemoryStatus.ACTIVE
        return memory.status

    def should_consolidate(self, memories: list[MemoryRecord]) -> bool:
        """Return whether memories form a valid episodic consolidation group."""

        return (
            len(memories) >= self.policy.consolidation_threshold
            and all(memory.kind == MemoryKind.EPISODIC for memory in memories)
            and all(memory.status != MemoryStatus.RETIRED for memory in memories)
        )

    def consolidate(
        self,
        memories: list[MemoryRecord],
        memory_id: str,
        state_or_summary: str,
        action: str | None = None,
        summary: str | None = None,
    ) -> MemoryRecord:
        """Create semantic memory from compatible episodic evidence.

        ``state_or_summary`` supports the historical three-argument API where
        only the consolidated memory identifier and summary were supplied.
        """

        if not self.should_consolidate(memories):
            raise ValueError("insufficient compatible episodic memories for consolidation")

        if summary is None:
            state = ""
            consolidated_action = ""
            consolidated_summary = state_or_summary
        else:
            state = state_or_summary
            consolidated_action = action or ""
            consolidated_summary = summary

        if not memory_id.strip() or not consolidated_summary.strip():
            raise ValueError("memory_id and summary must not be empty")

        total_successes = sum(memory.successes for memory in memories)
        total_failures = sum(memory.failures for memory in memories)
        total_uses = sum(memory.uses for memory in memories)
        average_reward = fsum(memory.reward for memory in memories) / len(memories)
        confidence = min(0.99, max(memory.confidence for memory in memories) + 0.05)

        consolidated_memory = MemoryRecord(
            memory_id=memory_id,
            state=state,
            action=consolidated_action,
            outcome=consolidated_summary,
            kind=MemoryKind.SEMANTIC,
            reward=average_reward,
            uses=total_uses,
            successes=total_successes,
            failures=total_failures,
            confidence=confidence,
            metadata={"derived_from": [memory.memory_id for memory in memories]},
        )
        for memory in memories:
            memory.status = MemoryStatus.RETIRED
        return consolidated_memory


def _elapsed_days(start_time: datetime, end_time: datetime) -> float:
    """Return non-negative elapsed days and reject naive datetimes."""

    if start_time.tzinfo is None or end_time.tzinfo is None:
        raise ValueError("lifecycle timestamps must be timezone-aware")
    return max(0.0, (end_time - start_time).total_seconds() / SECONDS_PER_DAY)
