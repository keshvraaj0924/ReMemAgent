from __future__ import annotations

from remem.benchmark import BenchmarkSuiteRunner

from experiments.external_benchmark import ExternalBenchmarkSpec, run_repeated_external_benchmarks


def test_repeated_benchmarks_reuse_injected_runner(monkeypatch) -> None:
    spec = ExternalBenchmarkSpec(
        benchmark_name="synthetic-eval",
        episode_count=1,
        max_steps=1,
        environment_factory="example:make_environment",
        policy_factory="example:make_policy",
        success_evaluator="example:is_success",
    )
    runner = BenchmarkSuiteRunner()
    observed: list[tuple[int | None, BenchmarkSuiteRunner | None]] = []

    def fake_run_external_benchmark(
        current_spec: ExternalBenchmarkSpec,
        *,
        runner: BenchmarkSuiteRunner | None = None,
    ):
        observed.append((current_spec.seed, runner))
        return object()

    monkeypatch.setattr(
        "experiments.external_benchmark.run_external_benchmark",
        fake_run_external_benchmark,
    )

    reports = run_repeated_external_benchmarks(spec, (3, 5), runner=runner)

    assert len(reports) == 2
    assert observed == [(3, runner), (5, runner)]
