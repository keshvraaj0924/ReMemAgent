from __future__ import annotations

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import (
    run_paired_external_benchmarks,
    run_paired_external_benchmarks_with_preflight,
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


@pytest.mark.parametrize(
    ("baseline_label", "treatment_label", "error_type", "message"),
    [
        ("memory", " MEMORY ", ValueError, "distinct conditions"),
        ("", "memory", ValueError, "baseline_label"),
        ("baseline", "   ", ValueError, "treatment_label"),
        (1, "memory", TypeError, "baseline_label"),
        ("baseline", None, TypeError, "treatment_label"),
    ],
)
def test_paired_execution_rejects_invalid_labels_before_callable_validation(
    monkeypatch,
    baseline_label,
    treatment_label,
    error_type,
    message,
) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda *args, **kwargs: pytest.fail("callable validation must not start"),
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("measured execution must not start"),
    )

    with pytest.raises(error_type, match=message):
        run_paired_external_benchmarks(
            baseline,
            treatment,
            (11, 17),
            baseline_label=baseline_label,
            treatment_label=treatment_label,
        )


def test_paired_execution_with_preflight_rejects_invalid_labels_before_preflight(
    monkeypatch,
) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    monkeypatch.setattr(
        "experiments.paired_benchmark.preflight_paired_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("runtime preflight must not start"),
    )

    with pytest.raises(ValueError, match="distinct conditions"):
        run_paired_external_benchmarks_with_preflight(
            baseline,
            treatment,
            (11, 17),
            baseline_label="baseline",
            treatment_label="BASELINE",
        )
