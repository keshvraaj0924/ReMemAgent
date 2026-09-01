from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli
from experiments.external_benchmark import ExternalBenchmarkSpec


def test_main_builds_external_spec_and_persists_report(monkeypatch, tmp_path: Path) -> None:
    report = object()
    output_path = tmp_path / "report.json"
    arguments = Namespace(
        benchmark="webshop-eval",
        episodes=3,
        max_steps=7,
        seed=41,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        action_policy_factory=None,
        minimum_trust=0.25,
        success_evaluator="example:is_success",
        transfer_success_evaluator="example:is_transfer_success",
        output=output_path,
    )
    captured: dict[str, ExternalBenchmarkSpec] = {}
    captured_provenance: dict[str, str] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def run_external_benchmark(spec: ExternalBenchmarkSpec) -> object:
        captured["spec"] = spec
        return report

    def save_report(value: object, path: Path, *, runtime_provenance: dict[str, str]) -> Path:
        assert value is report
        captured_provenance.update(runtime_provenance)
        return path

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", run_external_benchmark)
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", save_report)
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda: benchmark_cli.collect_runtime_provenance(environment={"REMEM_GIT_COMMIT": "abc123"}),
    )

    assert benchmark_cli.main() == 0
    spec = captured["spec"]
    assert spec.benchmark_name == "webshop-eval"
    assert spec.episode_count == 3
    assert spec.max_steps == 7
    assert spec.seed == 41
    assert spec.environment_factory == "example:make_environment"
    assert spec.policy_factory == "example:make_policy"
    assert spec.action_policy_factory is None
    assert spec.minimum_trust == 0.25
    assert spec.success_evaluator == "example:is_success"
    assert spec.transfer_success_evaluator == "example:is_transfer_success"
    assert captured_provenance["code_revision"] == "abc123"


def test_main_builds_memory_guided_spec(monkeypatch, tmp_path: Path) -> None:
    report = object()
    arguments = Namespace(
        benchmark="alfworld-eval",
        episodes=2,
        max_steps=5,
        seed=9,
        environment_factory="example:make_environment",
        policy_factory=None,
        action_policy_factory="model_policy:make_action_policy",
        minimum_trust=0.8,
        success_evaluator="example:is_success",
        transfer_success_evaluator=None,
        output=tmp_path / "report.json",
    )
    captured: dict[str, ExternalBenchmarkSpec] = {}

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "run_external_benchmark",
        lambda spec: captured.setdefault("spec", spec) or report,
    )
    monkeypatch.setattr(benchmark_cli, "save_benchmark_report", lambda value, path, **_: path)

    assert benchmark_cli.main() == 0
    spec = captured["spec"]
    assert spec.policy_factory is None
    assert spec.action_policy_factory == "model_policy:make_action_policy"
    assert spec.minimum_trust == 0.8
