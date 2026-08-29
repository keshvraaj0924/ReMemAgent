"""Tests for descriptive transferability metrics."""

import pytest

from remem.memory import MemoryRecord
from remem.memory.transferability import measure_transferability


def test_transferability_reports_observed_success_rate() -> None:
    memory = MemoryRecord(
        memory_id="m1",
        state="open drawer",
        transfer_attempts=4,
        transfer_successes=3,
    )

    metrics = measure_transferability(memory)

    assert metrics.attempts == 4
    assert metrics.successes == 3
    assert metrics.failures == 1
    assert metrics.empirical_rate == pytest.approx(0.75)
    assert 0.0 < metrics.lower_confidence_bound < metrics.empirical_rate


def test_unused_memory_has_neutral_empirical_rate_but_no_evidence() -> None:
    memory = MemoryRecord(memory_id="unused", state="state")

    metrics = measure_transferability(memory)

    assert metrics.empirical_rate == 0.5
    assert metrics.lower_confidence_bound == 0.0
    assert metrics.attempts == 0


def test_transferability_rejects_negative_confidence_parameter() -> None:
    memory = MemoryRecord(memory_id="m1", state="state")

    with pytest.raises(ValueError, match="confidence_z"):
        measure_transferability(memory, confidence_z=-1.0)


def test_transferability_rejects_inconsistent_counts() -> None:
    memory = MemoryRecord(
        memory_id="m1",
        state="state",
        transfer_attempts=1,
        transfer_successes=2,
    )

    with pytest.raises(ValueError, match="consistent"):
        measure_transferability(memory)
