"""Tests for reproducible experiment execution and metadata persistence."""

import json
import random

import pytest

from experiments.reproducibility import (
    EXPERIMENT_PROTOCOL_VERSION,
    ROUTING_HEURISTIC_VERSION,
)
from experiments.runner import (
    ExperimentConfig,
    run_reproducible_ablation,
    run_repeated_ablations,
    save_report,
)
from experiments.synthetic_negative_transfer import BenchmarkCase


def _cases_factory(generator: random.Random) -> list[BenchmarkCase]:
    return [
        BenchmarkCase("case_a", generator.random(), 0.5, memory_id="memory_a"),
        BenchmarkCase("case_b", generator.random(), 0.4, memory_id="memory_b"),
    ]


def test_reproducible_ablation_reuses_seeded_case_sequence() -> None:
    config = ExperimentConfig(seed=7)

    first = run_reproducible_ablation(_cases_factory, config)
    second = run_reproducible_ablation(_cases_factory, config)

    assert first.case_fingerprint == second.case_fingerprint
    assert first.experiment_fingerprint == second.experiment_fingerprint
    assert first.results == second.results


def test_saved_report_contains_protocol_metadata(tmp_path) -> None:
    report = run_reproducible_ablation(_cases_factory)
    output_path = save_report(report, tmp_path / "report.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["protocol_version"] == EXPERIMENT_PROTOCOL_VERSION
    assert payload["routing_heuristic_version"] == ROUTING_HEURISTIC_VERSION
    assert payload["experiment_fingerprint"] == report.experiment_fingerprint


@pytest.mark.parametrize("seed", [True, False, "7", 7.0])
def test_experiment_config_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        ExperimentConfig(seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "minimum_delta",
    [True, False, float("nan"), float("inf"), float("-inf")],
)
def test_experiment_config_rejects_invalid_minimum_delta(minimum_delta: object) -> None:
    expected_exception = TypeError if isinstance(minimum_delta, bool) else ValueError
    with pytest.raises(expected_exception):
        ExperimentConfig(minimum_delta=minimum_delta)  # type: ignore[arg-type]


def test_repeated_ablations_reject_non_integer_seed_values() -> None:
    with pytest.raises(TypeError, match="seeds must contain only integers"):
        run_repeated_ablations(_cases_factory, [1, "2"])  # type: ignore[list-item]


def test_repeated_ablations_reject_boolean_seed_values() -> None:
    with pytest.raises(TypeError, match="seeds must contain only integers"):
        run_repeated_ablations(_cases_factory, [1, True])  # type: ignore[list-item]
