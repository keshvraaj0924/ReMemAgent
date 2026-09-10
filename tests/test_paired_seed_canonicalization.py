from __future__ import annotations

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import (
    preflight_paired_external_benchmarks,
    run_paired_external_benchmarks,
)

BASELINE_POLICY = "tests.test_external_benchmark:make_policy"
TREATMENT_POLICY = "tests.test_external_benchmark:make_memory_policy"


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


def test_paired_execution_is_invariant_to_input_seed_order(monkeypatch) -> None:
    baseline = _spec(BASELINE_POLICY)
    treatment = _spec(TREATMENT_POLICY)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda spec: None,
    )

    def fake_run(spec: ExternalBenchmarkSpec, seeds: tuple[int, ...]):
        seed = seeds[0]
        calls.append((spec.policy_factory or "", seed))
        return ((spec.policy_factory, seed),)

    def fake_compare(baseline_reports, treatment_reports, *, baseline_label, treatment_label):
        assert baseline_reports == ((BASELINE_POLICY, 11), (BASELINE_POLICY, 17))
        assert treatment_reports == ((TREATMENT_POLICY, 11), (TREATMENT_POLICY, 17))
        return (baseline_label, treatment_label)

    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        fake_run,
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.compare_benchmark_reports",
        fake_compare,
    )

    result = run_paired_external_benchmarks(
        baseline,
        treatment,
        (17, 11),
        baseline_label="no-memory",
        treatment_label="memory",
    )

    assert calls == [
        (BASELINE_POLICY, 11),
        (TREATMENT_POLICY, 11),
        (TREATMENT_POLICY, 17),
        (BASELINE_POLICY, 17),
    ]
    assert result.baseline_reports == ((BASELINE_POLICY, 11), (BASELINE_POLICY, 17))
    assert result.treatment_reports == ((TREATMENT_POLICY, 11), (TREATMENT_POLICY, 17))


def test_paired_preflight_is_invariant_to_input_seed_order(monkeypatch) -> None:
    baseline = _spec(BASELINE_POLICY)
    treatment = _spec(TREATMENT_POLICY)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda spec: None,
    )

    def fake_preflight(
        spec: ExternalBenchmarkSpec,
        seeds: tuple[int, ...],
        *,
        probe_action: str | None,
    ) -> None:
        assert probe_action == "look"
        calls.append((spec.policy_factory or "", seeds[0]))

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_repeated_external_benchmark_runtime",
        fake_preflight,
    )

    preflight_paired_external_benchmarks(
        baseline,
        treatment,
        (5, 3),
        probe_action="look",
    )

    assert calls == [
        (BASELINE_POLICY, 3),
        (TREATMENT_POLICY, 3),
        (TREATMENT_POLICY, 5),
        (BASELINE_POLICY, 5),
    ]
