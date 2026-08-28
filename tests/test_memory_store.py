"""Tests for lifecycle-aware deterministic memory storage."""

from remem.memory.store import MemoryStore
from remem.memory.types import MemoryKind, MemoryRecord, MemoryStatus


def make_memory(
    memory_id: str,
    kind: MemoryKind = MemoryKind.EPISODIC,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        state="the agent sees a container",
        action="inspect the container",
        outcome="successful",
        kind=kind,
        status=status,
    )


def test_active_returns_only_active_memories() -> None:
    store = MemoryStore(
        [
            make_memory("active"),
            make_memory("stale", status=MemoryStatus.STALE),
            make_memory("retired", status=MemoryStatus.RETIRED),
        ]
    )

    assert [memory.memory_id for memory in store.active()] == ["active"]


def test_by_status_is_deterministic() -> None:
    store = MemoryStore(
        [
            make_memory("memory_b", status=MemoryStatus.STALE),
            make_memory("memory_a", status=MemoryStatus.STALE),
        ]
    )

    assert [memory.memory_id for memory in store.by_status(MemoryStatus.STALE)] == [
        "memory_a",
        "memory_b",
    ]


def test_all_preserves_access_to_non_active_memories() -> None:
    store = MemoryStore([make_memory("retired", status=MemoryStatus.RETIRED)])

    assert [memory.memory_id for memory in store.all()] == ["retired"]
