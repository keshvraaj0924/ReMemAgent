"""Tests for generic benchmark episode execution."""

from collections.abc import Callable

import pytest

from remem.environments.base import StepResult
from remem.execution import EpisodeRunner


class FakeEnvironment:
    """Small deterministic adapter test double."""

    def __init__(self, results: list[StepResult]) -> None:
        self._results = iter(results)
        self.actions: list[str] = []
        self.reset_calls = 0

    def reset(self, **kwargs: object) -> str:
        self.reset_calls += 1
        return str(kwargs.get("initial", "start"))

    def step(self, action: str) -> StepResult:
        self.actions.append(action)
        return next(self._results)

    def close(self) -> None:
        return None


def test_runner_records_trajectory_until_termination() -> None:
    environment = FakeEnvironment(
        [
            StepResult("middle", 0.5, False, False, {}),
            StepResult("goal", 1.0, True, False, {"success": True}),
        ]
    )
    policy: Callable[[str], str] = lambda observation: f"act:{observation}"

    result = EpisodeRunner().run(environment, policy, max_steps=5)

    assert result.initial_observation == "start"
    assert [step.action for step in result.steps] == ["act:start", "act:middle"]
    assert result.total_reward == 1.5
    assert result.terminated is True
    assert result.truncated is False
    assert result.completed is True


def test_runner_stops_at_step_limit_without_fabricating_completion() -> None:
    environment = FakeEnvironment([StepResult("next", 0.25, False, False, {})])

    result = EpisodeRunner().run(environment, lambda _: "wait", max_steps=1)

    assert len(result.steps) == 1
    assert result.total_reward == 0.25
    assert result.completed is False
    assert result.terminated is False
    assert result.truncated is False


@pytest.mark.parametrize("max_steps", [0, -1])
def test_runner_rejects_invalid_step_limit(max_steps: int) -> None:
    environment = FakeEnvironment([])

    with pytest.raises(ValueError, match="max_steps"):
        EpisodeRunner().run(environment, lambda _: "wait", max_steps=max_steps)


def test_runner_rejects_empty_policy_action() -> None:
    environment = FakeEnvironment([])

    with pytest.raises(ValueError, match="non-empty action"):
        EpisodeRunner().run(environment, lambda _: "  ", max_steps=1)
