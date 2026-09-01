from __future__ import annotations

import pytest

from remem.environments.base import StepResult
from remem.environments.validation import validate_environment_contract


class FakeEnvironment:
    def __init__(self, observation: object = "start") -> None:
        self.observation = observation
        self.closed = False
        self.actions: list[str] = []

    def reset(self, **kwargs: object) -> object:
        return self.observation

    def step(self, action: str) -> StepResult:
        self.actions.append(action)
        return StepResult(
            observation="next",
            reward=1.0,
            terminated=False,
            truncated=False,
        )

    def close(self) -> None:
        self.closed = True


def test_validate_environment_contract_checks_reset_and_closes() -> None:
    environment = FakeEnvironment()

    report = validate_environment_contract(environment, reset_kwargs={"seed": 7})

    assert report.initial_observation == "start"
    assert report.step_result is None
    assert environment.closed is True


def test_validate_environment_contract_checks_one_step() -> None:
    environment = FakeEnvironment()

    report = validate_environment_contract(environment, probe_action="look")

    assert report.step_result is not None
    assert report.step_result.observation == "next"
    assert environment.actions == ["look"]
    assert environment.closed is True


def test_validate_environment_contract_rejects_non_string_reset_observation() -> None:
    environment = FakeEnvironment(observation={"text": "start"})

    with pytest.raises(TypeError, match="reset observation must be a string"):
        validate_environment_contract(environment)

    assert environment.closed is True


def test_validate_environment_contract_rejects_empty_probe_action() -> None:
    environment = FakeEnvironment()

    with pytest.raises(ValueError, match="probe_action"):
        validate_environment_contract(environment, probe_action=" ")

    assert environment.closed is True


def test_validate_environment_contract_rejects_non_finite_reward() -> None:
    class NonFiniteEnvironment(FakeEnvironment):
        def step(self, action: str) -> StepResult:
            return StepResult(
                observation="next",
                reward=float("nan"),
                terminated=False,
                truncated=False,
            )

    environment = NonFiniteEnvironment()

    with pytest.raises(ValueError, match="reward must be finite"):
        validate_environment_contract(environment, probe_action="look")

    assert environment.closed is True


def test_validate_environment_contract_rejects_wrong_step_result_type() -> None:
    class InvalidEnvironment(FakeEnvironment):
        def step(self, action: str) -> object:
            return object()

    environment = InvalidEnvironment()

    with pytest.raises(TypeError, match="must be a StepResult"):
        validate_environment_contract(environment, probe_action="look")

    assert environment.closed is True
