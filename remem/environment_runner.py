"""Bounded execution of normalized text-environment episodes."""

from __future__ import annotations

from collections.abc import Callable

from remem.environments.base import EnvironmentAdapter
from remem.execution import EpisodeResult, EpisodeStep

ActionPolicy = Callable[[str, tuple[EpisodeStep, ...]], str]


def run_environment_episode(
    environment: EnvironmentAdapter,
    policy: ActionPolicy,
    *,
    max_steps: int,
) -> EpisodeResult:
    """Run one bounded episode and always release environment resources.

    The policy receives the current observation and an immutable snapshot of the
    transitions already observed. Hitting ``max_steps`` before the environment
    finishes is represented as runner truncation. Environment-reported
    truncation is preserved separately from termination.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    steps: list[EpisodeStep] = []
    total_reward = 0.0
    terminated = False
    environment_truncated = False
    initial_observation = ""

    try:
        initial_observation = environment.reset()
        observation = initial_observation

        for step_index in range(max_steps):
            action = policy(observation, tuple(steps))
            if not isinstance(action, str):
                raise TypeError("policy must return an action string")
            if not action.strip():
                raise ValueError("policy must return a non-empty action string")

            result = environment.step(action)
            steps.append(
                EpisodeStep(
                    step_index=step_index,
                    observation=observation,
                    action=action,
                    result=result,
                )
            )
            total_reward += result.reward
            observation = result.observation
            if result.done:
                terminated = result.terminated
                environment_truncated = result.truncated
                break
    finally:
        environment.close()

    runner_truncated = not terminated and not environment_truncated and len(steps) == max_steps
    return EpisodeResult(
        initial_observation=initial_observation,
        steps=tuple(steps),
        total_reward=total_reward,
        terminated=terminated,
        truncated=environment_truncated or runner_truncated,
    )


__all__ = ["ActionPolicy", "run_environment_episode"]
