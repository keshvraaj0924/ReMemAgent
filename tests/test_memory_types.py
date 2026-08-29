"""Tests for core memory domain contracts."""

import pytest

from remem.memory.types import MemoryRecord, RetrievedMemory


def create_memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory-1",
        state="A closed cabinet contains an apple.",
        action="Open the cabinet before placing the apple.",
        outcome="The placement succeeds after opening the cabinet.",
    )


def test_memory_record_supports_legacy_minimal_construction() -> None:
    memory = MemoryRecord("memory-1", "state")

    assert memory.action == ""
    assert memory.outcome == ""


def test_retrieved_memory_accepts_bounded_similarity() -> None:
    retrieved_memory = RetrievedMemory(memory=create_memory(), similarity=0.8)

    assert retrieved_memory.memory.memory_id == "memory-1"
    assert retrieved_memory.similarity == 0.8


@pytest.mark.parametrize("similarity", [-0.01, 1.01])
def test_retrieved_memory_rejects_invalid_similarity(similarity: float) -> None:
    with pytest.raises(ValueError, match="similarity"):
        RetrievedMemory(memory=create_memory(), similarity=similarity)


def test_empirical_success_rate_uses_observed_outcomes() -> None:
    memory = create_memory()
    memory.record_use(success=True)
    memory.record_use(success=False)
    memory.record_use(success=True)

    assert memory.empirical_success_rate == pytest.approx(2 / 3)


def test_transferability_uses_transfer_outcomes_only() -> None:
    memory = create_memory()
    memory.record_use(success=True, transferred=True)
    memory.record_use(success=False, transferred=True)
    memory.record_use(success=True, transferred=False)

    assert memory.transferability == pytest.approx(0.5)
