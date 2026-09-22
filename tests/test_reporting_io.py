import json

import pytest

from remem.reporting import ExperimentSummary
from remem.reporting_io import load_experiment_summary, save_experiment_summary


def _summary() -> ExperimentSummary:
    return ExperimentSummary.from_dict(
        {
            "counters": {"retrieval.total": 4, "retrieval.accepted": 3},
            "mean_durations": {"retrieval": 0.3},
            "counter_rates": {"retrieval.acceptance_rate": 0.75},
        }
    )


def test_experiment_summary_round_trip_is_deterministic(tmp_path) -> None:
    destination = tmp_path / "summary.json"

    assert save_experiment_summary(_summary(), destination) == destination
    restored = load_experiment_summary(destination)

    assert restored.to_dict() == _summary().to_dict()
    assert destination.read_text(encoding="utf-8") == (
        '{"counter_rates":{"retrieval.acceptance_rate":0.75},'
        '"counters":{"retrieval.accepted":3,"retrieval.total":4},'
        '"mean_durations":{"retrieval":0.3}}\n'
    )


def test_save_experiment_summary_revalidates_direct_construction(tmp_path) -> None:
    malformed = ExperimentSummary(counters={"episodes": -1}, mean_durations={}, counter_rates={})

    with pytest.raises(ValueError, match="non-negative"):
        save_experiment_summary(malformed, tmp_path / "summary.json")


def test_save_experiment_summary_rejects_invalid_inputs(tmp_path) -> None:
    with pytest.raises(TypeError, match="ExperimentSummary"):
        save_experiment_summary({}, tmp_path / "summary.json")  # type: ignore[arg-type]

    missing_parent = tmp_path / "missing" / "summary.json"
    with pytest.raises(FileNotFoundError, match="parent directory"):
        save_experiment_summary(_summary(), missing_parent)


def test_load_experiment_summary_rejects_invalid_json(tmp_path) -> None:
    destination = tmp_path / "summary.json"
    destination.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="valid JSON"):
        load_experiment_summary(destination)


def test_load_experiment_summary_rejects_non_object_root(tmp_path) -> None:
    destination = tmp_path / "summary.json"
    destination.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON root"):
        load_experiment_summary(destination)


def test_load_experiment_summary_rejects_invalid_schema(tmp_path) -> None:
    destination = tmp_path / "summary.json"
    destination.write_text(
        json.dumps({"counters": {}, "mean_durations": {}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        load_experiment_summary(destination)
