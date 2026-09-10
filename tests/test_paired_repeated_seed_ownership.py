from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import (
    preflight_paired_external_benchmarks,
    run_paired_external_benchmarks,
)


def _spec(policy_factory: str) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="synthetic-eval",
        episode_count=2,
        max_steps=4,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory=policy_factory,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def test_paired_execution_rejects_treatment_seed_before_any_condition_runs(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = replace(
        _spec("tests.test_external_benchmark:make_alternate_policy"),
        seed=19,
    )

    def fail_if_called(*args, **kwargs) -> None:
        pytest.fail("no callable resolution or measured execution may start")

    monkeypatch.setattr("experiments.paired_benchmark.validate_external_benchmark", fail_if_called)
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="spec.seed must be None"):
        run_paired_external_benchmarks(baseline, treatment, (11, 17))


def test_paired_preflight_rejects_treatment_seed_before_runtime_probe(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = replace(
        _spec("tests.test_external_benchmark:make_alternate_policy"),
        seed=23,
    )

    def fail_if_called(*args, **kwargs) -> None:
        pytest.fail("no callable resolution or runtime preflight may start")

    monkeypatch.setattr("experiments.paired_benchmark.validate_external_benchmark", fail_if_called)
    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_repeated_external_benchmark_runtime",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="spec.seed must be None"):
        preflight_paired_external_benchmarks(baseline, treatment, (11, 17))
