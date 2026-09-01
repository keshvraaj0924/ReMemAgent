"""Runtime validation for normalized external environment adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from remem.environments.base import EnvironmentAdapter, StepResult


@dataclass(frozen=True, slots=True)
class EnvironmentContractReport:
    """Observed values from a successful environment contract probe."""

    initial_observation: str
    step_result: StepResult | None = None


def validate_environment_contract(
    environment: EnvironmentAdapter,
    *,
    reset_kwargs: dict[str, Any] | None = None,
    probe_action: str | None = None,
) -> EnvironmentContractReport:
    """Validate reset and, optionally, one normalized environment step.

    The validator intentionally requires a caller-provided ``probe_action``
    before invoking ``step`` because there is no universally safe no-op action
    across ALFWorld, WebShop, or arbitrary adapters. The environment is always
    closed after the probe, including when validation fails.
    """

    try:
        initial_observation = environment.reset(**(reset_kwargs or {}))
        _validate_observation(initial_observation, field_name="reset observation")

        if probe_action is None:
            return EnvironmentContractReport(initial_observation=initial_observation)

        if not isinstance(probe_action, str) or not probe_action.strip():
            raise ValueError("probe_action must be a non-empty string")

        step_result = environment.step(probe_action)
        _validate_step_result(step_result)
        return EnvironmentContractReport(
            initial_observation=initial_observation,
            step_result=step_result,
        )
    finally:
        _close_environment(environment)


def _validate_observation(observation: object, *, field_name: str) -> None:
    """Validate the observation type required by the normalized contract."""

    if not isinstance(observation, str):
        raise TypeError(f"{field_name} must be a string, got {type(observation).__name__}")


def _validate_step_result(step_result: object) -> None:
    """Validate the normalized result returned by one environment step."""

    if not isinstance(step_result, StepResult):
        raise TypeError(
            "step result must be a StepResult, "
            f"got {type(step_result).__name__}"
        )
    _validate_observation(step_result.observation, field_name="step observation")
    if not math.isfinite(step_result.reward):
        raise ValueError("step reward must be finite")
    if not isinstance(step_result.terminated, bool):
        raise TypeError("step_result.terminated must be a bool")
    if not isinstance(step_result.truncated, bool):
        raise TypeError("step_result.truncated must be a bool")


def _close_environment(environment: EnvironmentAdapter) -> None:
    """Close an environment when its implementation exposes cleanup."""

    close = getattr(environment, "close", None)
    if callable(close):
        close()
