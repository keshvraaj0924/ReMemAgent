"""Tests for lifecycle-aware deterministic memory storage."""

import pytest

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


def test_record_transfer_outcome_updates_usage_and_transfer_counters() -> None:
    memory = make_memory("transferable")
    store = MemoryStore([memory])

    store.record_transfer_outcome("transferable", success=True)
    store.record_transfer_outcome("transferable", success=False)

    assert memory.uses == 2
    assert memory.successes == 1
    assert memory.failures == 1
    assert memory.transfer_attempts == 2
    assert memory.transfer_successes == 1


def test_record_transfer_outcome_rejects_unknown_memory() -> None:
    store = MemoryStore()

    with pytest.raises(KeyError, match="missing"):
        store.record_transfer_outcome("missing", success=True)
