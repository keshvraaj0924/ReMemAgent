import random

import pytest

from experiments.cli import build_synthetic_cases


def test_build_synthetic_cases_is_reproducible() -> None:
    first = build_synthetic_cases(random.Random(7), 3)
    second = build_synthetic_cases(random.Random(7), 3)

    assert first == second
    assert [case.case_id for case in first] == [
        "synthetic_0000",
        "synthetic_0001",
        "synthetic_0002",
    ]


def test_build_synthetic_cases_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError, match="case_count must be positive"):
        build_synthetic_cases(random.Random(7), 0)
