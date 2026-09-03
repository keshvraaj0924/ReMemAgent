"""Regression coverage for independent repeated external benchmark runs."""

import pytest

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_repeated_external_benchmarks,
)


def test_repeated_external_benchmarks_isolate_seeded_runs() -> None:
    """Each seed receives an independent benchmark report and memory lifecycle."""

    spec = ExternalBenchmarkSpec(
        benchmark_name="alfworld-smoke",
        episode_count=2,
        max_steps=1,
        environment_factory="experiments.smoke_benchmark:build_environment",
        policy_factory=None,
        action_policy_factory="experiments.smoke_benchmark:build_action_policy",
        success_evaluator="experiments.smoke_benchmark:is_success",
        seed=999,
    )

    reports = run_repeated_external_benchmarks(spec, [0, 10])

    assert [report.seed for report in reports] == [0, 10]
    assert all(report.configuration is not None for report in reports)
    assert [report.configuration.seed for report in reports if report.configuration] == [0, 10]
    assert all(report.benchmark_name == "alfworld-smoke" for report in reports)
    assert [report.success_count for report in reports] == [2, 0]


def test_repeated_external_benchmarks_reject_empty_seeds() -> None:
    """A repeated experiment must identify at least one deterministic seed."""

    spec = ExternalBenchmarkSpec(
        benchmark_name="alfworld-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory="experiments.smoke_benchmark:build_environment",
        policy_factory=None,
        action_policy_factory="experiments.smoke_benchmark:build_action_policy",
        success_evaluator="experiments.smoke_benchmark:is_success",
    )

    with pytest.raises(ValueError, match="^seeds must contain at least one seed$"):
        run_repeated_external_benchmarks(spec, [])


def test_repeated_external_benchmarks_reject_duplicate_seeds() -> None:
    """Duplicate seeds would make the repeated result artifact ambiguous."""

    spec = ExternalBenchmarkSpec(
        benchmark_name="alfworld-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory="experiments.smoke_benchmark:build_environment",
        policy_factory=None,
        action_policy_factory="experiments.smoke_benchmark:build_action_policy",
        success_evaluator="experiments.smoke_benchmark:is_success",
    )

    with pytest.raises(ValueError, match="^seeds must be unique$"):
        run_repeated_external_benchmarks(spec, [7, 7])
