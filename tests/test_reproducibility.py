"""Tests for reproducible experiment fingerprints."""

from experiments.reproducibility import fingerprint_cases
from experiments.synthetic_negative_transfer import BenchmarkCase


def test_case_fingerprint_is_stable_for_identical_inputs() -> None:
    cases = [
        BenchmarkCase("a", 0.8, 0.4),
        BenchmarkCase("b", 0.3, 0.7),
    ]

    assert fingerprint_cases(cases) == fingerprint_cases(cases)


def test_case_fingerprint_changes_when_case_values_change() -> None:
    original = [BenchmarkCase("a", 0.8, 0.4)]
    changed = [BenchmarkCase("a", 0.81, 0.4)]

    assert fingerprint_cases(original) != fingerprint_cases(changed)


def test_case_fingerprint_preserves_case_order() -> None:
    first = [BenchmarkCase("a", 0.8, 0.4), BenchmarkCase("b", 0.3, 0.7)]
    reversed_cases = list(reversed(first))

    assert fingerprint_cases(first) != fingerprint_cases(reversed_cases)


def test_case_fingerprint_changes_when_memory_attribution_changes() -> None:
    original = [BenchmarkCase("a", 0.8, 0.4, memory_id="memory_a")]
    changed = [BenchmarkCase("a", 0.8, 0.4, memory_id="memory_b")]

    assert fingerprint_cases(original) != fingerprint_cases(changed)


def test_case_fingerprint_changes_when_transfer_outcome_changes() -> None:
    successful = [BenchmarkCase("a", 0.8, 0.4, transfer_success=True)]
    failed = [BenchmarkCase("a", 0.8, 0.4, transfer_success=False)]

    assert fingerprint_cases(successful) != fingerprint_cases(failed)
