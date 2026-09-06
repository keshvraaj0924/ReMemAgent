from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli


def _arguments(tmp_path: Path, *, repeated: bool = False) -> Namespace:
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
        observability_output=tmp_path / "observability.json",
        overwrite=False,
        preflight=False,
        runtime_preflight=False,
        repeated_runtime_preflight=False,
        preflight_before_run=False,
        probe_action=None,
    )


def test_main_persists_single_run_observability_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: _arguments(tmp_path))

    assert benchmark_cli.main() == 0

    snapshot = json.loads(
        (tmp_path / "observability.json").read_text(encoding="utf-8")
    )
    assert snapshot["counters"]["benchmark.runs"] == 1.0
    assert snapshot["counters"]["benchmark.runs.completed"] == 1.0
    assert snapshot["counters"]["benchmark.episodes.started"] == 2.0
    assert snapshot["counters"]["benchmark.episodes.completed"] == 2.0
    assert snapshot["counters"]["benchmark.episodes.succeeded"] == 2.0
    assert snapshot["counters"]["benchmark.episodes.successful"] == 2.0
    assert snapshot["counters"]["benchmark.transfers.attributed"] == 0.0
    assert snapshot["durations_seconds"]["benchmark.episode.duration_seconds"] >= 0.0


def test_main_persists_repeated_run_observability_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, repeated=True)
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    assert benchmark_cli.main() == 0

    snapshot = json.loads(
        (tmp_path / "observability.json").read_text(encoding="utf-8")
    )
    assert snapshot["counters"]["benchmark.runs"] == 1.0
    assert snapshot["counters"]["benchmark.runs.completed"] == 1.0
    assert snapshot["counters"]["benchmark.episodes.completed"] == 4.0
    assert snapshot["counters"]["benchmark.episodes.successful"] == 4.0


def test_main_rejects_existing_observability_snapshot_before_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    arguments.observability_output.write_text("existing", encoding="utf-8")
    run_called = False

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def fail_if_called(*args, **kwargs):
        nonlocal run_called
        run_called = True
        raise AssertionError("measured execution must not start")

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", fail_if_called)

    try:
        benchmark_cli.main()
    except FileExistsError as exc:
        assert "observability snapshot" in str(exc)
    else:
        raise AssertionError("existing observability output should fail")

    assert run_called is False
