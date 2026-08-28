"""Tests for deterministic trust and transferability scoring."""

from remem.memory.trust import MemoryTrustScorer
from remem.memory.types import MemoryRecord


def make_memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory-1",
        state="closed cabinet",
        action="open cabinet",
        outcome="placement succeeded",
    )


def test_trust_uses_recorded_transferability() -> None:
    memory = make_memory()
    memory.record_use(success=True, transferred=True)
    memory.record_use(success=False, transferred=True)

    score = MemoryTrustScorer().score(memory, similarity=0.8)

    assert score.transferability == 0.5
    assert score.historical_success == 0.5


def test_explicit_transferability_overrides_recorded_value() -> None:
    memory = make_memory()
    memory.record_use(success=True, transferred=True)

    score = MemoryTrustScorer().score(
        memory,
        similarity=0.8,
        transferability=0.2,
    )

    assert score.transferability == 0.2


def test_trust_confidence_is_weighted_and_bounded() -> None:
    score = MemoryTrustScorer().score(
        make_memory(),
        similarity=1.0,
        transferability=1.0,
        freshness=1.0,
    )

    assert score.confidence == 0.85
    assert 0.0 <= score.confidence <= 1.0
