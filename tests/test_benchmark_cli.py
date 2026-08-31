from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli


def test_main_wires_external_factories_and_persists_report(monkeypatch, tmp_path: Path) -> None:
    environment_factory = object()
    policy_factory = object()
    success_evaluator = object()
    report = object()
    output_path = tmp_path / "report.json"
    arguments = Namespace(
        benchmark="webshop-eval",
        episodes=3,
        max_steps=7,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        success_evaluator="example:is_success",
        transfer_success_evaluator=None,
        output=output_path,
    )
    loaded_specs: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def load_callable(specification: str) -> object:
        loaded_specs.append(specification)
        return environment_factory if "environment" in specification else policy_factory

    monkeypatch.setattr(benchmark_cli, "load_callable", load_callable)
    monkeypatch.setattr(benchmark_cli, "load_typed_callable", lambda specification: success_evaluator)

    def run_external_benchmark(**kwargs: object) -> object:
        captured.update(kwargs)
        return report

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", run_external_benchmark)
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", lambda value, path: path)

    assert benchmark_cli.main() == 0
    assert loaded_specs == ["example:make_environment", "example:make_policy"]
    assert captured["benchmark_name"] == "webshop-eval"
    assert captured["episode_count"] == 3
    assert captured["max_steps"] == 7
    assert captured["environment_factory"] is environment_factory
    assert captured["policy_factory"] is policy_factory
    assert captured["success_evaluator"] is success_evaluator
    assert captured["transfer_success_evaluator"] is None
