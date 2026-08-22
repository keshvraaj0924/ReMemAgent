"""Memory lifecycle operations: scoring, aging, and consolidation."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from .types import MemoryKind, MemoryRecord, MemoryStatus

@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    stale_after_days: float = 30.0
    retire_after_days: float = 90.0
    min_transferability: float = 0.25
    consolidation_threshold: int = 3

class MemoryLifecycle:
    def __init__(self, policy: LifecyclePolicy | None = None) -> None:
        self.policy = policy or LifecyclePolicy()

    def health_score(self, memory: MemoryRecord, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        age_days = max(0.0, (now - memory.created_at).total_seconds() / 86400)
        freshness = max(0.0, 1.0 - age_days / self.policy.retire_after_days)
        return 0.35 * memory.empirical_success + 0.30 * memory.transferability + 0.20 * memory.confidence + 0.15 * freshness

    def refresh_status(self, memory: MemoryRecord, now: datetime | None = None) -> MemoryStatus:
        now = now or datetime.now(timezone.utc)
        last = memory.last_used_at or memory.created_at
        idle_days = max(0.0, (now - last).total_seconds() / 86400)
        if idle_days >= self.policy.retire_after_days or (memory.transfer_attempts and memory.transferability < self.policy.min_transferability):
            memory.status = MemoryStatus.RETIRED
        elif idle_days >= self.policy.stale_after_days:
            memory.status = MemoryStatus.STALE
        else:
            memory.status = MemoryStatus.ACTIVE
        return memory.status

    def should_consolidate(self, memories: list[MemoryRecord]) -> bool:
        return len(memories) >= self.policy.consolidation_threshold and all(m.kind == MemoryKind.EPISODIC for m in memories)

    def consolidate(self, memories: list[MemoryRecord], memory_id: str, summary: str) -> MemoryRecord:
        if not self.should_consolidate(memories):
            raise ValueError("insufficient compatible episodic memories for consolidation")
        successes = sum(m.successes for m in memories)
        uses = sum(m.uses for m in memories)
        failures = sum(m.failures for m in memories)
        record = MemoryRecord(memory_id=memory_id, content=summary, kind=MemoryKind.SEMANTIC, uses=uses, successes=successes, failures=failures, confidence=min(0.99, max(m.confidence for m in memories) + 0.05), metadata={"derived_from": [m.memory_id for m in memories]})
        for memory in memories:
            memory.status = MemoryStatus.RETIRED
        return record
