from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from experiments.benchmark_report import save_paired_benchmark_result
from experiments.benchmark_statistics import compare_benchmark_reports
from experiments.external_benchmark import ExternalBenchmarkSpec, run_repeated_external_benchmarks


def _spec(policy_factory: str) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=1,
        max_steps=1,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory=policy_factory,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def _save_pair(tmp_path: Path, treatment_policy: str, filename: str) -> dict[str, object]:
    baseline = run_repeated_external_benchmarks(
        _spec("tests.test_external_benchmark:make_policy"), (7, 11)
    )
    treatment = run_repeated_external_benchmarks(_spec(treatment_policy), (7, 11))
    comparison = compare_benchmark_reports(
        baseline,
        treatment,
        baseline_label="baseline",
        treatment_label="treatment",
    )
    output = save_paired_benchmark_result(
        baseline,
        treatment,
        comparison,
        tmp_path / filename,
        runtime_provenance={"schema_version": 1},
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_paired_identity_changes_when_only_treatment_policy_changes(tmp_path: Path) -> None:
    first = _save_pair(
        tmp_path,
        "tests.test_external_benchmark:make_memory_policy",
        "first.json",
    )
    second = _save_pair(
        tmp_path,
        "tests.test_external_benchmark:make_invalid_policy",
        "second.json",
    )

    assert first["experiment_identity"] != second["experiment_identity"]
    assert first["configuration_fingerprint"] != second["configuration_fingerprint"]


def test_paired_protocol_validation_ignores_policy_identity(tmp_path: Path) -> None:
    baseline = run_repeated_external_benchmarks(
        _spec("tests.test_external_benchmark:make_policy"), (7, 11)
    )
    treatment = run_repeated_external_benchmarks(
        _spec("tests.test_external_benchmark:make_memory_policy"), (7, 11)
    )
    drifted_treatment = tuple(
        replace(
            report,
            configuration=replace(report.configuration, max_steps=2)
            if report.configuration is not None
            else None,
        )
        for report in treatment
    )
    comparison = compare_benchmark_reports(
        baseline,
        drifted_treatment,
        baseline_label="baseline",
        treatment_label="treatment",
    )

    try:
        save_paired_benchmark_result(
            baseline,
            drifted_treatment,
            comparison,
            tmp_path / "drift.json",
        )
    except ValueError as exc:
        assert "share configuration apart from the seed and policy" in str(exc)
    else:
        raise AssertionError("protocol drift must be rejected")
