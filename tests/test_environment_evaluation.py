from __future__ import annotations

from collections.abc import Callable

import pytest

from remem.environment_evaluation import evaluate_environment_seeds
from remem.environments.base import EnvironmentAdapter, StepResult


class SeededEnvironment(EnvironmentAdapter):
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.step_count = 0
        self.closed = False

    def reset(self) -> str:
        return f"seed:{self.seed}"

    def step(self, action: str) -> StepResult:
        self.step_count += 1
        return StepResult(
            observation=f"{action}:{self.step_count}",
            reward=float(self.seed),
            terminated=self.step_count == 1,
        )

    def close(self) -> None:
        self.closed = True


def _policy_factory(seed: int) -> Callable[[str, tuple[object, ...]], str]:
    def policy(observation: str, history: tuple[object, ...]) -> str:
        del observation, history
        return f"act-{seed}"

    return policy


def test_evaluate_environment_seeds_preserves_order_and_aggregates() -> None:
    environments: dict[int, SeededEnvironment] = {}

    def environment_factory(seed: int) -> EnvironmentAdapter:
        environment = SeededEnvironment(seed)
        environments[seed] = environment
        return environment

    evaluation = evaluate_environment_seeds(
        environment_factory,
        _policy_factory,
        seeds=(3, 1, 2),
        max_steps=2,
    )

    assert tuple(episode.seed for episode in evaluation.episodes) == (3, 1, 2)
    assert evaluation.episode_count == 3
    assert evaluation.mean_reward == pytest.approx(2.0)
    assert evaluation.mean_steps == pytest.approx(1.0)
    assert evaluation.termination_rate == pytest.approx(1.0)
    assert evaluation.truncation_rate == pytest.approx(0.0)
    assert all(environment.closed for environment in environments.values())


def test_evaluate_environment_seeds_rejects_empty_suite_before_execution() -> None:
    called = False

    def environment_factory(seed: int) -> EnvironmentAdapter:
        nonlocal called
        called = True
        return SeededEnvironment(seed)

    with pytest.raises(ValueError, match="seeds must not be empty"):
        evaluate_environment_seeds(
            environment_factory,
            _policy_factory,
            seeds=(),
            max_steps=1,
        )

    assert called is False


def test_evaluate_environment_seeds_rejects_duplicate_seeds_before_execution() -> None:
    called = False

    def environment_factory(seed: int) -> EnvironmentAdapter:
        nonlocal called
        called = True
        return SeededEnvironment(seed)

    with pytest.raises(ValueError, match="seeds must be unique"):
        evaluate_environment_seeds(
            environment_factory,
            _policy_factory,
            seeds=(4, 4),
            max_steps=1,
        )

    assert called is False


def test_evaluate_environment_seeds_rejects_invalid_step_budget_before_execution() -> None:
    called = False

    def environment_factory(seed: int) -> EnvironmentAdapter:
        nonlocal called
        called = True
        return SeededEnvironment(seed)

    with pytest.raises(ValueError, match="max_steps must be positive"):
        evaluate_environment_seeds(
            environment_factory,
            _policy_factory,
            seeds=(1,),
            max_steps=0,
        )

    assert called is False
