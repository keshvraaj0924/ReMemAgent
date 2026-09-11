from typing import Any, Callable

import pytest

from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    MAX_EXTERNAL_BENCHMARK_SEED,
    run_external_benchmark,
    run_repeated_external_benchmarks,
)
from remem.memory.store import MemoryStore


CREATED_SEEDS: list[int] = []


class _SeedEnvironment:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def reset(self) -> str:
        return f"seed-{self.seed}"

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, object]]:
        return "done", 1.0, True, False, {"action": action}


def make_environment(seed: int) -> _SeedEnvironment:
    CREATED_SEEDS.append(seed)
    return _SeedEnvironment(seed)


def make_policy(seed: int, store: MemoryStore) -> Callable[[str], str]:
    del store
    return lambda _observation: f"act-{seed}"


def evaluate_success(episode: Any) -> bool:
    return episode.total_reward > 0


@pytest.mark.parametrize("seed", [-1, MAX_EXTERNAL_BENCHMARK_SEED + 1])
def test_external_benchmark_spec_rejects_run_seed_outside_supported_domain(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be between 0 and 4294967295 inclusive"):
        _build_spec(seed=seed)


def test_external_benchmark_spec_rejects_derived_episode_seed_overflow() -> None:
    with pytest.raises(ValueError, match="derived episode seed range exceeds"):
        _build_spec(seed=MAX_EXTERNAL_BENCHMARK_SEED - 1, episode_count=3)


def test_external_benchmark_accepts_maximum_seed_for_single_episode() -> None:
    CREATED_SEEDS.clear()

    report = run_external_benchmark(
        _build_spec(seed=MAX_EXTERNAL_BENCHMARK_SEED, episode_count=1)
    )

    assert report.seed == MAX_EXTERNAL_BENCHMARK_SEED
    assert CREATED_SEEDS == [MAX_EXTERNAL_BENCHMARK_SEED]


def test_repeated_external_benchmark_rejects_derived_seed_overflow_before_construction() -> None:
    CREATED_SEEDS.clear()

    with pytest.raises(ValueError, match="derived episode seed range exceeds"):
        run_repeated_external_benchmarks(
            _build_spec(seed=None, episode_count=2),
            (MAX_EXTERNAL_BENCHMARK_SEED,),
        )

    assert CREATED_SEEDS == []


def _build_spec(**overrides: Any) -> ExternalBenchmarkSpec:
    values: dict[str, Any] = {
        "benchmark_name": "webshop-seed-span",
        "episode_count": 1,
        "max_steps": 1,
        "environment_factory": "tests.test_external_benchmark_seed_span:make_environment",
        "policy_factory": "tests.test_external_benchmark_seed_span:make_policy",
        "success_evaluator": "tests.test_external_benchmark_seed_span:evaluate_success",
        "seed": 7,
    }
    values.update(overrides)
    return ExternalBenchmarkSpec(**values)
