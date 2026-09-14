from __future__ import annotations

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments import paired_benchmark
from remem.benchmark import BenchmarkRunReport


def _spec() -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=1,
        max_steps=2,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def test_single_seed_condition_rejects_wrong_benchmark_identity(monkeypatch) -> None:
    spec = _spec()
    wrong_report = BenchmarkRunReport(
        benchmark_name="webshop",
        episodes=(),
        final_memory_count=0,
        seed=11,
    )
    monkeypatch.setattr(
        paired_benchmark,
        "run_repeated_external_benchmarks",
        lambda *_args, **_kwargs: (wrong_report,),
    )

    with pytest.raises(RuntimeError, match="wrong benchmark"):
        paired_benchmark._run_single_seed_condition(
            spec,
            11,
            condition_role="baseline",
        )


def test_single_seed_condition_accepts_matching_benchmark_identity(monkeypatch) -> None:
    spec = _spec()
    expected_report = BenchmarkRunReport(
        benchmark_name="alfworld",
        episodes=(),
        final_memory_count=0,
        seed=11,
    )
    monkeypatch.setattr(
        paired_benchmark,
        "run_repeated_external_benchmarks",
        lambda *_args, **_kwargs: (expected_report,),
    )

    report = paired_benchmark._run_single_seed_condition(
        spec,
        11,
        condition_role="baseline",
    )

    assert report is expected_report
