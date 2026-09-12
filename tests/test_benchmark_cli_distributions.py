from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import experiments.benchmark_cli as benchmark_cli
from experiments.benchmark_distribution_config import BENCHMARK_EPISODE_DURATION_METRIC


def _arguments(
    tmp_path: Path,
    *,
    repeated: bool = False,
    with_observability: bool = False,
) -> Namespace:
    return Namespace(
        benchmark="webshop-smoke",
        episodes=2,
        max_steps=1,
        seed=None if repeated else 7,
        seeds="7,11" if repeated else None,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        action_policy_factory=None,
        minimum_trust=0.0,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        transfer_success_evaluator=None,
        output=tmp_path / "report.json",
        manifest=None,
        observability_output=(tmp_path / "observability.json") if with_observability else None,
        distribution_output=tmp_path / "duration-distribution.json",
        episode_duration_buckets="0.001,0.01,0.1,1.0",
        overwrite=False,
        preflight=False,
        runtime_preflight=False,
        repeated_runtime_preflight=False,
        preflight_before_run=False,
        probe_action=None,
        require_code_revision=None,
        require_clean_working_tree=False,
        require_dependency_version=None,
        source_checkout=None,
        require_source_revision=None,
        allow_dirty_source_checkout=None,
    )


def _load_histogram(path: Path) -> dict[str, object]:
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 1
    return sidecar["duration_histograms"][BENCHMARK_EPISODE_DURATION_METRIC]


def test_main_persists_single_run_duration_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    assert benchmark_cli.main() == 0

    histogram = _load_histogram(arguments.distribution_output)
    assert histogram["upper_bounds"] == [0.001, 0.01, 0.1, 1.0]
    assert histogram["count"] == 2
    assert sum(histogram["bucket_counts"]) == 2
    assert histogram["total"] >= 0.0
    assert histogram["mean"] >= 0.0


def test_main_persists_repeated_run_duration_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, repeated=True)
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    assert benchmark_cli.main() == 0

    histogram = _load_histogram(arguments.distribution_output)
    assert histogram["count"] == 4
    assert sum(histogram["bucket_counts"]) == 4


def test_main_uses_same_duration_measurements_for_aggregate_and_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, with_observability=True)
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    assert benchmark_cli.main() == 0

    aggregate = json.loads(arguments.observability_output.read_text(encoding="utf-8"))
    histogram = _load_histogram(arguments.distribution_output)
    assert aggregate["durations_seconds"][BENCHMARK_EPISODE_DURATION_METRIC] == pytest.approx(
        histogram["total"]
    )
    assert histogram["count"] == aggregate["counters"]["benchmark.episodes.completed"]


def test_main_rejects_existing_distribution_sidecar_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.distribution_output.write_text("existing", encoding="utf-8")
    run_called = False
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal run_called
        run_called = True
        raise AssertionError("measured execution must not start")

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", fail_if_called)

    with pytest.raises(FileExistsError, match="distribution observability sidecar"):
        benchmark_cli.main()

    assert run_called is False


def test_main_rejects_distribution_path_alias_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, with_observability=True)
    arguments.distribution_output = arguments.observability_output
    run_called = False
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal run_called
        run_called = True
        raise AssertionError("measured execution must not start")

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", fail_if_called)

    with pytest.raises(ValueError, match="different path"):
        benchmark_cli.main()

    assert run_called is False


@pytest.mark.parametrize(
    ("distribution_output", "episode_duration_buckets", "expected_message"),
    [
        (None, "0.1,1.0", "requires --distribution-output"),
        (Path("duration.json"), None, "requires --episode-duration-buckets"),
    ],
)
def test_main_rejects_incomplete_distribution_configuration_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    distribution_output: Path | None,
    episode_duration_buckets: str | None,
    expected_message: str,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.distribution_output = (
        tmp_path / distribution_output if distribution_output is not None else None
    )
    arguments.episode_duration_buckets = episode_duration_buckets
    run_called = False
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal run_called
        run_called = True
        raise AssertionError("measured execution must not start")

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", fail_if_called)

    with pytest.raises(ValueError, match=expected_message):
        benchmark_cli.main()

    assert run_called is False
