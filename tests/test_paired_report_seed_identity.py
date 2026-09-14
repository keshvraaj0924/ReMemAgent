from __future__ import annotations

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import run_paired_external_benchmarks
from remem.benchmark import BenchmarkRunReport


def _spec(policy_factory: str) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=2,
        max_steps=4,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory=policy_factory,
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


@pytest.mark.parametrize("reported_seed", [99, None])
def test_paired_execution_rejects_report_for_unrequested_seed(
    monkeypatch,
    reported_seed: int | None,
) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    calls: list[str] = []

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_external_benchmark",
        lambda spec: None,
    )

    def fake_run(spec: ExternalBenchmarkSpec, seeds: tuple[int, ...]):
        calls.append(spec.policy_factory or "")
        return (
            BenchmarkRunReport(
                benchmark_name=spec.benchmark_name,
                episodes=(),
                final_memory_count=0,
                seed=reported_seed,
            ),
        )

    monkeypatch.setattr(
        "experiments.paired_benchmark.run_repeated_external_benchmarks",
        fake_run,
    )
    monkeypatch.setattr(
        "experiments.paired_benchmark.compare_benchmark_reports",
        lambda *args, **kwargs: pytest.fail("comparison must not run with mismatched paired seeds"),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"wrong seed for baseline: requested 11, received {reported_seed!r}",
    ):
        run_paired_external_benchmarks(baseline, treatment, (11,))

    assert calls == ["tests.test_external_benchmark:make_policy"]
