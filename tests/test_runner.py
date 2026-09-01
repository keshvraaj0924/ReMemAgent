import json

import pytest

from experiments.runner import (
    ExperimentConfig,
    run_repeated_ablations,
    run_reproducible_ablation,
    save_repeated_reports,
    save_report,
)
from experiments.synthetic_negative_transfer import BenchmarkCase


def _make_cases(rng) -> list[BenchmarkCase]:
    values = [rng.random() for _ in range(2)]
    return [
        BenchmarkCase("case_a", values[0], 0.5),
        BenchmarkCase("case_b", values[1], 0.5),
    ]


def test_same_seed_produces_identical_report() -> None:
    config = ExperimentConfig(seed=17)

    first = run_reproducible_ablation(_make_cases, config)
    second = run_reproducible_ablation(_make_cases, config)

    assert first == second


def test_different_seed_changes_generated_cases() -> None:
    first = run_reproducible_ablation(_make_cases, ExperimentConfig(seed=1))
    second = run_reproducible_ablation(_make_cases, ExperimentConfig(seed=2))

    assert first.case_ids == second.case_ids
    assert first.results != second.results
    assert first.experiment_fingerprint != second.experiment_fingerprint


def test_configuration_change_changes_experiment_fingerprint() -> None:
    first = run_reproducible_ablation(_make_cases, ExperimentConfig(minimum_delta=0.05))
    second = run_reproducible_ablation(_make_cases, ExperimentConfig(minimum_delta=0.10))

    assert first.case_fingerprint == second.case_fingerprint
    assert first.experiment_fingerprint != second.experiment_fingerprint


def test_save_report_writes_stable_json(tmp_path) -> None:
    report = run_reproducible_ablation(_make_cases, ExperimentConfig(seed=9))
    output_path = save_report(report, tmp_path / "nested" / "report.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 9
    assert payload["case_ids"] == ["case_a", "case_b"]
    assert payload["case_fingerprint"] == report.case_fingerprint
    assert payload["experiment_fingerprint"] == report.experiment_fingerprint
    assert [item["strategy"] for item in payload["results"]] == [
        "memory_always",
        "self_reasoning_always",
        "counterfactual",
    ]


def test_runner_rejects_invalid_case_factory_output() -> None:
    def duplicate_cases(_rng):
        return [BenchmarkCase("same", 0.5, 0.4), BenchmarkCase("same", 0.3, 0.6)]

    with pytest.raises(ValueError, match="case_id values must be unique"):
        run_reproducible_ablation(duplicate_cases)


def test_repeated_runner_preserves_requested_seed_order() -> None:
    reports = run_repeated_ablations(_make_cases, [11, 7, 19])

    assert [report.seed for report in reports] == [11, 7, 19]
    assert len({report.experiment_fingerprint for report in reports}) == 3


def test_repeated_runner_uses_shared_protocol_configuration() -> None:
    reports = run_repeated_ablations(
        _make_cases,
        [1, 2],
        ExperimentConfig(minimum_delta=0.2),
    )

    assert all(report.minimum_delta == 0.2 for report in reports)


@pytest.mark.parametrize("seeds", [[], [3, 3]])
def test_repeated_runner_rejects_invalid_seed_sets(seeds) -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_repeated_ablations(_make_cases, seeds)


def test_save_repeated_reports_preserves_each_report(tmp_path) -> None:
    reports = run_repeated_ablations(_make_cases, [5, 6])
    output_path = save_repeated_reports(reports, tmp_path / "reports.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["seeds"] == [5, 6]
    assert [item["seed"] for item in payload["reports"]] == [5, 6]
    assert payload["reports"][0]["experiment_fingerprint"] == reports[0].experiment_fingerprint


def test_save_repeated_reports_rejects_duplicate_seed_reports(tmp_path) -> None:
    report = run_reproducible_ablation(_make_cases, ExperimentConfig(seed=4))

    with pytest.raises(ValueError, match="report seeds"):
        save_repeated_reports([report, report], tmp_path / "reports.json")
