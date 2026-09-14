from __future__ import annotations

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import PairedSeedExecution, run_paired_external_benchmarks


def _spec(policy_factory: str) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=1,
        max_steps=3,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory=policy_factory,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def test_paired_execution_roles_stay_stable_while_labels_are_normalized(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda _spec: None,
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        lambda spec, seeds: ((spec.policy_factory, seeds[0]),),
    )

    captured_labels: list[tuple[str, str]] = []

    def fake_compare(
        baseline_reports,
        treatment_reports,
        *,
        baseline_label: str,
        treatment_label: str,
    ):
        del baseline_reports, treatment_reports
        captured_labels.append((baseline_label, treatment_label))
        return (baseline_label, treatment_label)

    monkeypatch.setattr(
        "experiments.paired_benchmark.compare_benchmark_reports",
        fake_compare,
    )

    result = run_paired_external_benchmarks(
        baseline,
        treatment,
        (17, 11),
        baseline_label="  no-memory  ",
        treatment_label="  memory  ",
    )

    assert captured_labels == [("no-memory", "memory")]
    assert result.execution_order == (
        PairedSeedExecution(
            seed=11,
            first_condition="baseline",
            second_condition="treatment",
        ),
        PairedSeedExecution(
            seed=17,
            first_condition="treatment",
            second_condition="baseline",
        ),
    )
