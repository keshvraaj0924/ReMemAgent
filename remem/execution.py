"""Generic episode execution over normalized benchmark environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from remem.environments.base import EnvironmentAdapter, StepResult


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    """One action and normalized environment transition in an episode."""

    step_index: int
    observation: str
    action: str
    result: StepResult


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Complete trajectory produced by one environment episode."""

    initial_observation: str
    steps: tuple[EpisodeStep, ...]
    total_reward: float
    terminated: bool
    truncated: bool

    @property
    def completed(self) -> bool:
        """Return whether the environment ended before the step limit."""

        return self.terminated or self.truncated


Policy = Callable[[str], str]


class EpisodeRunner:
    """Execute a policy against an environment without benchmark-specific logic."""

    def run(
        self,
        environment: EnvironmentAdapter,
        policy: Policy,
        *,
        max_steps: int,
        reset_kwargs: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        """Run one episode and return its immutable trajectory record."""

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        initial_observation = environment.reset(**(reset_kwargs or {}))
        observation = initial_observation
        steps: list[EpisodeStep] = []
        total_reward = 0.0

        for step_index in range(max_steps):
            action = policy(observation)
            if not action.strip():
                raise ValueError("policy must return a non-empty action")

            result = environment.step(action)
            steps.append(EpisodeStep(step_index, observation, action, result))
            total_reward += result.reward
            observation = result.observation

            if result.done:
                return EpisodeResult(
                    initial_observation=initial_observation,
                    steps=tuple(steps),
                    total_reward=total_reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )

        return EpisodeResult(
            initial_observation=initial_observation,
            steps=tuple(steps),
            total_reward=total_reward,
            terminated=False,
            truncated=False,
        )
