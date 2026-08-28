"""Tests for failure-memory capture."""

import pytest

from remem.memory.failure import FailureMemoryBuilder, FailureObservation
from remem.memory.types import MemoryKind, MemoryRecord


def test_builder_creates_failure_memory() -> None:
    memory = FailureMemoryBuilder().build(
        "failure-1", FailureObservation("closed cabinet", "place apple", "placement failed")
    )
    assert memory.kind is MemoryKind.FAILURE
    assert memory.failures == 1
    assert memory.state == "closed cabinet"


def test_builder_rejects_empty_observation_fields() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        FailureObservation("", "place apple", "failed")


def test_builder_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        FailureObservation("state", "action", "failed", confidence=1.5)


def test_update_preserves_failure_kind_and_records_transfer() -> None:
    memory = FailureMemoryBuilder().build(
        "failure-1", FailureObservation("state", "action", "failed")
    )
    FailureMemoryBuilder.update(memory, success=False, transferred=True)
    assert memory.kind is MemoryKind.FAILURE
    assert memory.transfer_attempts == 1
    assert memory.transfer_successes == 0


def test_update_rejects_non_failure_memory() -> None:
    memory = MemoryRecord("memory-1", "state", "action", "outcome")
    with pytest.raises(ValueError, match="only failure memories"):
        FailureMemoryBuilder.update(memory, success=False)
