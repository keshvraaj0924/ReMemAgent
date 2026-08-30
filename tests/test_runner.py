import json

import pytest

from experiments.runner import ExperimentConfig, run_reproducible_ablation, save_report
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
