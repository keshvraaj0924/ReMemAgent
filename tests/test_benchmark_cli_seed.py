from argparse import Namespace
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli


def test_main_forwards_seed_to_external_benchmark(monkeypatch, tmp_path: Path) -> None:
    arguments = Namespace(
        benchmark="alfworld",
        episodes=2,
        max_steps=5,
        seed=73,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        success_evaluator="example:is_success",
        transfer_success_evaluator=None,
        output=tmp_path / "report.json",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(benchmark_cli, "load_callable", lambda specification: object())
    monkeypatch.setattr(benchmark_cli, "load_typed_callable", lambda specification: object())

    def run_external_benchmark(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", run_external_benchmark)
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", lambda report, path: path)

    assert benchmark_cli.main() == 0
    assert captured["seed"] == 73
