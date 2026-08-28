"""Tests for deterministic memory lifecycle policy."""

from datetime import datetime, timedelta, timezone

import pytest

from remem.memory.lifecycle import LifecyclePolicy, MemoryLifecycle
from remem.memory.types import MemoryKind, MemoryRecord, MemoryStatus


def make_memory(**kwargs: object) -> MemoryRecord:
    defaults = {
        "memory_id": "memory-1",
        "state": "state",
        "action": "action",
        "outcome": "outcome",
    }
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


def test_health_score_is_bounded() -> None:
    memory = make_memory(confidence=1.0, uses=4, successes=4)
    lifecycle = MemoryLifecycle()
    now = datetime.now(timezone.utc)

    score = lifecycle.health_score(memory, now=now)

    assert 0.0 <= score <= 1.0


def test_stale_memory_is_retired_when_transferability_is_poor() -> None:
    memory = make_memory(
        transfer_attempts=4,
        transfer_successes=0,
    )
    lifecycle = MemoryLifecycle(LifecyclePolicy(minimum_transferability=0.25))

    status = lifecycle.refresh_status(memory, now=memory.created_at + timedelta(days=1))

    assert status is MemoryStatus.RETIRED


def test_unused_recent_memory_remains_active() -> None:
    memory = make_memory()
    lifecycle = MemoryLifecycle()

    status = lifecycle.refresh_status(memory, now=memory.created_at + timedelta(days=1))

    assert status is MemoryStatus.ACTIVE


def test_consolidation_requires_compatible_episodic_memories() -> None:
    lifecycle = MemoryLifecycle(LifecyclePolicy(consolidation_threshold=2))
    memories = [make_memory(memory_id="a"), make_memory(memory_id="b")]

    assert lifecycle.should_consolidate(memories) is True


def test_consolidation_rejects_failure_memories() -> None:
    lifecycle = MemoryLifecycle(LifecyclePolicy(consolidation_threshold=2))
    memories = [
        make_memory(memory_id="a"),
        make_memory(memory_id="b", kind=MemoryKind.FAILURE),
    ]

    assert lifecycle.should_consolidate(memories) is False
    with pytest.raises(ValueError):
        lifecycle.consolidate(memories, "semantic-1", "state", "action", "summary")


def test_consolidation_retires_source_episodes() -> None:
    lifecycle = MemoryLifecycle(LifecyclePolicy(consolidation_threshold=2))
    memories = [
        make_memory(memory_id="a", uses=2, successes=2, reward=1.0),
        make_memory(memory_id="b", uses=2, successes=1, failures=1, reward=0.0),
    ]

    consolidated = lifecycle.consolidate(
        memories,
        "semantic-1",
        "shared state",
        "shared action",
        "validated summary",
    )

    assert consolidated.kind is MemoryKind.SEMANTIC
    assert consolidated.metadata["derived_from"] == ["a", "b"]
    assert all(memory.status is MemoryStatus.RETIRED for memory in memories)
