from __future__ import annotations

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import run_paired_external_benchmarks


def _spec(policy_factory: str) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=2,
        max_steps=4,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory=policy_factory,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def test_paired_execution_rejects_missing_single_seed_report(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    calls: list[str] = []

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda spec: None,
    )

    def fake_run(spec: ExternalBenchmarkSpec, seeds: tuple[int, ...]):
        calls.append(spec.policy_factory or "")
        return ()

    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        fake_run,
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.compare_benchmark_reports",
        lambda *args, **kwargs: pytest.fail("comparison must not run with missing paired data"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"exactly one report for baseline seed 11; received 0",
    ):
        run_paired_external_benchmarks(baseline, treatment, (11,))

    assert calls == ["tests.test_external_benchmark:make_policy"]


def test_paired_execution_rejects_multiple_single_seed_reports(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda spec: None,
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        lambda spec, seeds: (object(), object()),
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.compare_benchmark_reports",
        lambda *args, **kwargs: pytest.fail("comparison must not run with duplicate paired data"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"exactly one report for baseline seed 11; received 2",
    ):
        run_paired_external_benchmarks(baseline, treatment, (11,))


def test_paired_execution_rejects_non_report_single_seed_result(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    calls: list[str] = []

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda spec: None,
    )

    def fake_run(spec: ExternalBenchmarkSpec, seeds: tuple[int, ...]):
        calls.append(spec.policy_factory or "")
        return (object(),)

    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        fake_run,
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.compare_benchmark_reports",
        lambda *args, **kwargs: pytest.fail("comparison must not run with invalid report types"),
    )

    with pytest.raises(
        TypeError,
        match=r"must return a BenchmarkRunReport for baseline seed 11; received object",
    ):
        run_paired_external_benchmarks(baseline, treatment, (11,))

    assert calls == ["tests.test_external_benchmark:make_policy"]
