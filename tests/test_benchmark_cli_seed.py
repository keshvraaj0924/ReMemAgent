from argparse import Namespace
from pathlib import Path

import experiments.benchmark_cli as benchmark_cli
from experiments.external_benchmark import ExternalBenchmarkSpec


def test_main_forwards_seed_to_external_benchmark(monkeypatch, tmp_path: Path) -> None:
    arguments = Namespace(
        benchmark="alfworld",
        episodes=2,
        max_steps=5,
        seed=73,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        action_policy_factory=None,
        minimum_trust=0.0,
        success_evaluator="example:is_success",
        transfer_success_evaluator=None,
        output=tmp_path / "report.json",
        manifest=None,
        preflight=False,
        runtime_preflight=False,
        repeated_runtime_preflight=False,
        preflight_before_run=False,
        probe_action=None,
        seeds=None,
    )
    captured: dict[str, object] = {}
    original_collect_runtime_provenance = benchmark_cli.collect_runtime_provenance

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    def run_external_benchmark(spec: ExternalBenchmarkSpec) -> object:
        captured["spec"] = spec
        return object()

    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", run_external_benchmark)
    monkeypatch.setattr(
        benchmark_cli,
        "save_benchmark_report",
        lambda report, path, **_: path,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda **kwargs: original_collect_runtime_provenance(
            environment={"REMEM_GIT_COMMIT": "abc"}
        ),
    )

    assert benchmark_cli.main() == 0
    spec = captured["spec"]
    assert isinstance(spec, ExternalBenchmarkSpec)
    assert spec.seed == 73
