from __future__ import annotations

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import preflight_paired_external_benchmarks


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


def test_paired_preflight_resolves_both_conditions_before_runtime_probe(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:missing_policy")
    events: list[str] = []

    def fake_validate(spec: ExternalBenchmarkSpec) -> None:
        events.append(f"validate:{spec.policy_factory}")
        if spec.policy_factory and spec.policy_factory.endswith("missing_policy"):
            raise ImportError("missing treatment policy")

    def fake_runtime_preflight(*args, **kwargs) -> None:
        pytest.fail("runtime preflight must not start until both callable sets resolve")

    monkeypatch.setattr("experiments.paired_benchmark.validate_external_benchmark", fake_validate)
    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_repeated_external_benchmark_runtime",
        fake_runtime_preflight,
    )

    with pytest.raises(ImportError, match="missing treatment policy"):
        preflight_paired_external_benchmarks(baseline, treatment, (11, 17))

    assert events == [
        "validate:tests.test_external_benchmark:make_policy",
        "validate:tests.test_external_benchmark:missing_policy",
    ]
