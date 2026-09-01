from __future__ import annotations

import json
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli
from experiments.external_benchmark import ExternalBenchmarkSpec


def test_parse_seeds_accepts_ordered_unique_integers() -> None:
    assert benchmark_cli._parse_seeds("11, 7, 23") == (11, 7, 23)


def test_parse_seeds_rejects_duplicates() -> None:
    try:
        benchmark_cli._parse_seeds("11,11")
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate seeds must be rejected")


def test_main_persists_statistics_for_repeated_runs(monkeypatch, tmp_path: Path) -> None:
    arguments = type("Arguments", (), {
        "benchmark": "synthetic-eval",
        "episodes": 2,
        "max_steps": 4,
        "seed": None,
        "seeds": "1,2",
        "environment_factory": "example:make_environment",
        "policy_factory": "example:make_policy",
        "action_policy_factory": None,
        "minimum_trust": 0.0,
        "success_evaluator": "example:is_success",
        "transfer_success_evaluator": None,
        "output": tmp_path / "repeated.json",
    })()
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda: benchmark_cli.collect_runtime_provenance(environment={"REMEM_GIT_COMMIT": "abc"}),
    )

    captured: dict[str, object] = {}

    def fake_run(spec: ExternalBenchmarkSpec, seeds: tuple[int, ...]):
        captured["spec"] = spec
        captured["seeds"] = seeds
        return (_report(1, True, 1.0), _report(2, False, 0.0))

    monkeypatch.setattr(benchmark_cli, "run_repeated_external_benchmarks", fake_run)

    def fake_save(reports, path, *, runtime_provenance, statistics):
        payload = {
            "seeds": [report.seed for report in reports],
            "runtime_provenance": runtime_provenance,
            "statistics": statistics,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    monkeypatch.setattr(benchmark_cli, "save_repeated_benchmark_reports", fake_save)

    assert benchmark_cli.main() == 0
    assert captured["seeds"] == (1, 2)
    persisted = json.loads((tmp_path / "repeated.json").read_text(encoding="utf-8"))
    assert persisted["seeds"] == [1, 2]
    assert persisted["statistics"]["success_rate"]["mean"] == 0.5


def _report(seed: int, success: bool, reward: float):
    from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunReport
    from remem.execution import EpisodeResult

    return BenchmarkRunReport(
        benchmark_name="synthetic-eval",
        episodes=(
            BenchmarkEpisodeReport(
                episode_id=f"episode-{seed}",
                episode=EpisodeResult(
                    initial_observation="start",
                    steps=(),
                    total_reward=reward,
                    terminated=True,
                    truncated=False,
                ),
                episode_success=success,
                retained_memory_count=0,
            ),
        ),
        final_memory_count=0,
        seed=seed,
    )
