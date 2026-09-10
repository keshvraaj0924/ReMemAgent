from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import (
    preflight_paired_external_benchmarks,
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


def test_run_paired_external_benchmarks_uses_same_seeds(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_validate(spec: ExternalBenchmarkSpec) -> None:
        calls.append((f"validate:{spec.policy_factory}", ()))

    def fake_run(spec: ExternalBenchmarkSpec, seeds: tuple[int, ...]):
        calls.append((spec.policy_factory or "", seeds))
        return tuple()

    def fake_compare(baseline_reports, treatment_reports, *, baseline_label, treatment_label):
        assert baseline_reports == treatment_reports == ()
        return (baseline_label, treatment_label)

    monkeypatch.setattr("experiments.paired_benchmark.validate_external_benchmark", fake_validate)
    monkeypatch.setattr("experiments.paired_benchmark.run_repeated_external_benchmarks", fake_run)
    monkeypatch.setattr("experiments.paired_benchmark.compare_benchmark_reports", fake_compare)

    result = run_paired_external_benchmarks(
        baseline,
        treatment,
        (11, 17),
        baseline_label="no-memory",
        treatment_label="memory",
    )

    assert calls == [
        ("validate:tests.test_external_benchmark:make_policy", ()),
        ("validate:tests.test_external_benchmark:make_memory_policy", ()),
        ("tests.test_external_benchmark:make_policy", (11, 17)),
        ("tests.test_external_benchmark:make_memory_policy", (11, 17)),
    ]
    assert result.comparison == ("no-memory", "memory")


def test_run_paired_external_benchmarks_validates_both_policies_before_execution(
    monkeypatch,
) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:missing_policy")
    events: list[str] = []

    def fake_validate(spec: ExternalBenchmarkSpec) -> None:
        events.append(f"validate:{spec.policy_factory}")
        if spec.policy_factory.endswith("missing_policy"):
            raise ImportError("missing policy")

    monkeypatch.setattr("experiments.paired_benchmark.validate_external_benchmark", fake_validate)
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("execution must not start after validation failure"),
    )

    with pytest.raises(ImportError, match="missing policy"):
        run_paired_external_benchmarks(baseline, treatment, (11, 17))

    assert events == [
        "validate:tests.test_external_benchmark:make_policy",
        "validate:tests.test_external_benchmark:missing_policy",
    ]


def test_run_paired_external_benchmarks_rejects_evaluation_drift(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = replace(
        baseline, max_steps=5, policy_factory="tests.test_external_benchmark:make_memory_policy"
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("execution must not start after validation failure"),
    )

    with pytest.raises(ValueError, match="max_steps"):
        run_paired_external_benchmarks(baseline, treatment, (11, 17))


def test_run_paired_external_benchmarks_rejects_identical_policy_configuration(
    monkeypatch,
) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_policy")
    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda *args, **kwargs: pytest.fail("callable validation must not start"),
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        lambda *args, **kwargs: pytest.fail("execution must not start"),
    )

    with pytest.raises(ValueError, match="distinct policy configurations"):
        run_paired_external_benchmarks(baseline, treatment, (11, 17))


def test_preflight_paired_external_benchmarks_checks_both_conditions(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    calls: list[tuple[str | None, tuple[int, ...], str | None]] = []

    def fake_preflight(spec, seeds, *, probe_action):
        calls.append((spec.policy_factory, tuple(seeds), probe_action))
        return ()

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_repeated_external_benchmark_runtime",
        fake_preflight,
    )

    preflight_paired_external_benchmarks(
        baseline,
        treatment,
        (3, 5),
        probe_action="look",
    )

    assert calls == [
        ("tests.test_external_benchmark:make_policy", (3, 5), "look"),
        ("tests.test_external_benchmark:make_memory_policy", (3, 5), "look"),
    ]


def test_run_paired_external_benchmarks_with_preflight_orders_preflight_before_run(
    monkeypatch,
) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    events: list[str] = []

    def fake_preflight(*args, **kwargs):
        events.append("preflight")

    def fake_run(*args, **kwargs):
        events.append("run")
        return "result"

    monkeypatch.setattr(
        "experiments.paired_benchmark.preflight_paired_external_benchmarks",
        fake_preflight,
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_paired_external_benchmarks",
        fake_run,
    )

    result = run_paired_external_benchmarks_with_preflight(
        baseline,
        treatment,
        (11, 17),
        baseline_label="base",
        treatment_label="memory",
        probe_action="look",
    )

    assert events == ["preflight", "run"]
    assert result == "result"


def test_preflight_rejects_different_benchmark_names(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = replace(
        baseline,
        benchmark_name="other-eval",
        policy_factory="tests.test_external_benchmark:make_memory_policy",
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_repeated_external_benchmark_runtime",
        lambda *args, **kwargs: pytest.fail("preflight must not start after validation failure"),
    )

    with pytest.raises(ValueError, match="benchmark_name"):
        preflight_paired_external_benchmarks(baseline, treatment, (3, 5))
