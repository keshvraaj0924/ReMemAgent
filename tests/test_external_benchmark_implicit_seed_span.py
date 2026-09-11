import pytest

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    MAX_EXTERNAL_BENCHMARK_SEED,
)


def _build_spec(*, episode_count: int) -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="webshop-smoke",
        episode_count=episode_count,
        max_steps=1,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def test_unseeded_external_benchmark_rejects_derived_seed_overflow() -> None:
    with pytest.raises(ValueError, match="derived episode seed range exceeds"):
        _build_spec(episode_count=MAX_EXTERNAL_BENCHMARK_SEED + 2)


def test_unseeded_external_benchmark_accepts_full_supported_seed_span() -> None:
    spec = _build_spec(episode_count=MAX_EXTERNAL_BENCHMARK_SEED + 1)

    assert spec.seed is None
    assert spec.episode_count == MAX_EXTERNAL_BENCHMARK_SEED + 1
