"""Regression coverage for episode-level RNG isolation across repeated runs."""

import pytest

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    validate_repeated_benchmark_request,
)


def _spec(*, episode_count: int) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld-smoke",
        episode_count=episode_count,
        max_steps=1,
        environment_factory="experiments.smoke_benchmark:build_environment",
        policy_factory=None,
        action_policy_factory="experiments.smoke_benchmark:build_action_policy",
        success_evaluator="experiments.smoke_benchmark:is_success",
        seed=None,
    )


def test_repeated_request_rejects_overlapping_episode_seed_ranges() -> None:
    """Adjacent run seeds cannot silently reuse per-episode RNG seeds."""

    with pytest.raises(
        ValueError,
        match=r"run seed 0 uses \[0, 2\].*run seed 1 uses \[1, 3\]",
    ):
        validate_repeated_benchmark_request(_spec(episode_count=3), (0, 1))


def test_repeated_request_accepts_disjoint_episode_seed_ranges() -> None:
    """Separated run seeds remain valid when every derived episode seed is unique."""

    selected = validate_repeated_benchmark_request(_spec(episode_count=3), (0, 3, 10))

    assert selected == (0, 3, 10)


def test_zero_episode_runs_do_not_create_seed_overlap() -> None:
    """Empty engineering runs consume no episode RNG seeds."""

    selected = validate_repeated_benchmark_request(_spec(episode_count=0), (0, 1))

    assert selected == (0, 1)


def test_overlap_detection_is_independent_of_seed_order() -> None:
    """Validation follows numeric seed ranges rather than caller ordering."""

    with pytest.raises(ValueError, match="overlapping episode seed ranges"):
        validate_repeated_benchmark_request(_spec(episode_count=2), (10, 0, 1))
