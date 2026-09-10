from __future__ import annotations

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_benchmark import preflight_paired_external_benchmarks


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


def test_paired_preflight_counterbalances_condition_order_by_seed(monkeypatch) -> None:
    baseline = _spec("tests.test_external_benchmark:make_policy")
    treatment = _spec("tests.test_external_benchmark:make_memory_policy")
    calls: list[tuple[str | None, tuple[int, ...], str | None]] = []

    def fake_validate_runtime(spec, seeds, *, probe_action):
        calls.append((spec.policy_factory, tuple(seeds), probe_action))

    monkeypatch.setattr(
        "experiments.paired_benchmark.validate_repeated_external_benchmark_runtime",
        fake_validate_runtime,
    )

    preflight_paired_external_benchmarks(
        baseline,
        treatment,
        (3, 5, 7),
        probe_action="look",
    )

    assert calls == [
        ("tests.test_external_benchmark:make_policy", (3,), "look"),
        ("tests.test_external_benchmark:make_memory_policy", (3,), "look"),
        ("tests.test_external_benchmark:make_memory_policy", (5,), "look"),
        ("tests.test_external_benchmark:make_policy", (5,), "look"),
        ("tests.test_external_benchmark:make_policy", (7,), "look"),
        ("tests.test_external_benchmark:make_memory_policy", (7,), "look"),
    ]
