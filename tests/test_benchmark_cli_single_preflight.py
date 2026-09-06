from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli
from experiments.external_benchmark import ExternalBenchmarkSpec
from remem.environments import EnvironmentContractReport


def _arguments(tmp_path: Path) -> Namespace:
    return Namespace(
        benchmark="webshop-eval",
        episodes=2,
        max_steps=5,
        seed=17,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        action_policy_factory=None,
        minimum_trust=0.25,
        success_evaluator="example:is_success",
        transfer_success_evaluator=None,
        output=tmp_path / "report.json",
        manifest=None,
        overwrite=False,
        preflight=False,
        runtime_preflight=False,
        repeated_runtime_preflight=False,
        preflight_before_run=True,
        probe_action="look",
        seeds=None,
    )


def test_single_run_preflights_before_execution(monkeypatch, tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    events: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def runtime_preflight(spec: ExternalBenchmarkSpec, *, probe_action: str | None):
        events.append("preflight")
        captured["spec"] = spec
        captured["probe_action"] = probe_action
        return EnvironmentContractReport(initial_observation="ready")

    monkeypatch.setattr(benchmark_cli, "validate_external_benchmark_runtime", runtime_preflight)
    monkeypatch.setattr(
        benchmark_cli,
        "run_external_benchmark",
        lambda spec: events.append("run") or object(),
    )
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", lambda value, path, **_: path)

    assert benchmark_cli.main() == 0
    assert events == ["preflight", "run"]
    assert captured["probe_action"] == "look"
    assert isinstance(captured["spec"], ExternalBenchmarkSpec)
    assert captured["spec"].seed == 17
