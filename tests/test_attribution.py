"""Regression coverage for memory transfer outcome attribution."""

import pytest

from remem.memory import MemoryGuidanceDecision, MemoryRecord, MemoryStore, MemoryTransferRecorder


def test_records_successful_transfer_on_selected_memory() -> None:
    memory = MemoryRecord(memory_id="m1", state="open drawer", action="open drawer")
    store = MemoryStore([memory])
    decision = MemoryGuidanceDecision(
        memory_id="m1",
        guidance="Previously opening the drawer exposed the key.",
        similarity=0.9,
        trust_confidence=0.8,
    )

    outcome = MemoryTransferRecorder().record(store, decision, success=True)

    assert outcome is not None
    assert outcome.memory_id == "m1"
    assert outcome.success is True
    assert memory.transfer_attempts == 1
    assert memory.transfer_successes == 1
    assert memory.uses == 1
    assert memory.successes == 1


def test_records_failed_transfer_without_counting_success() -> None:
    memory = MemoryRecord(memory_id="m1", state="open drawer", action="open drawer")
    store = MemoryStore([memory])
    decision = MemoryGuidanceDecision(
        memory_id="m1",
        guidance="Previously opening the drawer exposed the key.",
        similarity=0.9,
        trust_confidence=0.8,
    )

    MemoryTransferRecorder().record(store, decision, success=False)

    assert memory.transfer_attempts == 1
    assert memory.transfer_successes == 0
    assert memory.failures == 1


def test_self_reasoning_decision_is_not_attributed_as_transfer() -> None:
    decision = MemoryGuidanceDecision(
        memory_id=None,
        guidance="",
        similarity=0.0,
        trust_confidence=0.0,
    )

    outcome = MemoryTransferRecorder().record(MemoryStore(), decision, success=True)

    assert outcome is None


def test_missing_selected_memory_is_rejected() -> None:
    decision = MemoryGuidanceDecision(
        memory_id="missing",
        guidance="guidance",
        similarity=0.8,
        trust_confidence=0.8,
    )

    with pytest.raises(KeyError, match="missing"):
        MemoryTransferRecorder().record(MemoryStore(), decision, success=True)
