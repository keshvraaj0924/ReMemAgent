"""Tests for counterfactual routing domain objects."""

import pytest

from remem.memory.types import CounterfactualScore, MemoryDecision


def test_counterfactual_score_computes_utility_delta() -> None:
    score = CounterfactualScore(with_memory=0.9, without_memory=0.6)

    assert score.delta == pytest.approx(0.3)


def test_memory_decision_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        MemoryDecision(
            route="memory",
            confidence=1.1,
            expected_delta=0.2,
            reason="invalid confidence",
        )
