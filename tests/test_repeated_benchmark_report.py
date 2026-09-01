"""Tests for repeated benchmark report persistence."""

import json

import pytest

from experiments.benchmark_report import save_repeated_benchmark_reports
from experiments.external_benchmark import ExternalBenchmarkSpec, run_external_benchmark


def _smoke_spec(benchmark_name: str, seed: int) -> ExternalBenchmarkSpec:
    """Build a deterministic external benchmark specification."""

    return ExternalBenchmarkSpec(
        benchmark_name=benchmark_name,
        episode_count=2,
        max_steps=1,
        environment_factory="experiments.smoke_benchmark:build_environment",
        policy_factory=None,
        action_policy_factory="experiments.smoke_benchmark:build_action_policy",
        success_evaluator="experiments.smoke_benchmark:is_success",
        seed=seed,
    )


def test_save_repeated_benchmark_reports_preserves_seed_order(tmp_path) -> None:
    """Repeated serialization preserves requested seed ordering and provenance."""

    reports = (
        run_external_benchmark(_smoke_spec("alfworld-smoke", 0)),
        run_external_benchmark(_smoke_spec("alfworld-smoke", 10)),
    )
    output_path = save_repeated_benchmark_reports(reports, tmp_path / "repeated.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["benchmark_name"] == "alfworld-smoke"
    assert payload["seeds"] == [0, 10]
    assert payload["reports"][0]["seed"] == 0
    assert payload["reports"][1]["seed"] == 10


def test_save_repeated_benchmark_reports_rejects_empty_reports(tmp_path) -> None:
    """Empty repeated artifacts are rejected instead of looking completed."""

    with pytest.raises(ValueError, match="^reports must contain at least one benchmark report$"):
        save_repeated_benchmark_reports([], tmp_path / "repeated.json")


def test_save_repeated_benchmark_reports_rejects_mixed_benchmarks(tmp_path) -> None:
    """One repeated artifact cannot combine unrelated benchmark families."""

    first = run_external_benchmark(_smoke_spec("alfworld-smoke", 0))
    second = run_external_benchmark(_smoke_spec("webshop-smoke", 10))

    with pytest.raises(ValueError, match="^repeated benchmark reports must use one benchmark name$"):
        save_repeated_benchmark_reports([first, second], tmp_path / "repeated.json")


def test_save_repeated_benchmark_reports_rejects_duplicate_seeds(tmp_path) -> None:
    """Repeated artifacts reject duplicate seeds so rows remain attributable."""

    report = run_external_benchmark(_smoke_spec("alfworld-smoke", 3))

    with pytest.raises(ValueError, match="^benchmark report seeds must be unique$"):
        save_repeated_benchmark_reports([report, report], tmp_path / "repeated.json")
