from __future__ import annotations

import pytest

from remem.environment_runner import run_environment_episode
from remem.environments.base import EnvironmentAdapter, StepResult


class ScriptedEnvironment(EnvironmentAdapter):
    def __init__(self, results: list[StepResult]) -> None:
        self.results = results
        self.actions: list[str] = []
        self.closed = False

    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> StepResult:
        self.actions.append(action)
        return self.results[len(self.actions) - 1]

    def close(self) -> None:
        self.closed = True


def test_runner_records_transitions_and_termination() -> None:
    environment = ScriptedEnvironment(
        [
            StepResult("middle", 0.25, False),
            StepResult("goal", 1.0, True),
        ]
    )
    seen_history_lengths: list[int] = []

    def policy(observation: str, history: tuple[object, ...]) -> str:
        seen_history_lengths.append(len(history))
        return "advance"

    result = run_environment_episode(environment, policy, max_steps=5)

    assert result.initial_observation == "start"
    assert result.total_reward == pytest.approx(1.25)
    assert result.step_count == 2
    assert result.terminated is True
    assert result.truncated is False
    assert seen_history_lengths == [0, 1]
    assert environment.closed is True


def test_runner_marks_step_budget_as_truncation() -> None:
    environment = ScriptedEnvironment(
        [StepResult("one", 0.0, False), StepResult("two", 0.0, False)]
    )

    result = run_environment_episode(environment, lambda _observation, _history: "wait", max_steps=2)

    assert result.terminated is False
    assert result.truncated is True
    assert result.step_count == 2


def test_runner_preserves_environment_truncation() -> None:
    environment = ScriptedEnvironment([StepResult("timeout", 0.0, False, truncated=True)])

    result = run_environment_episode(environment, lambda _observation, _history: "wait", max_steps=5)

    assert result.terminated is False
    assert result.truncated is True
    assert result.step_count == 1


def test_runner_closes_environment_when_policy_fails() -> None:
    environment = ScriptedEnvironment([])

    def failing_policy(_observation: str, _history: tuple[object, ...]) -> str:
        raise RuntimeError("policy failed")

    with pytest.raises(RuntimeError, match="policy failed"):
        run_environment_episode(environment, failing_policy, max_steps=1)

    assert environment.closed is True


def test_runner_rejects_invalid_step_budget_before_reset() -> None:
    environment = ScriptedEnvironment([])

    with pytest.raises(ValueError, match="max_steps must be positive"):
        run_environment_episode(environment, lambda _observation, _history: "wait", max_steps=0)

    assert environment.closed is False
