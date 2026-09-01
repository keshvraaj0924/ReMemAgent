"""Deterministic end-to-end smoke benchmark for the external integration boundary.

This module is intentionally tiny and benchmark-independent. It mimics the
single-item batch shape expected by :class:`AlfWorldAdapter` while proving that
an external caller can supply an environment, a learned-component-shaped action
policy, and a success evaluator through the same path used by real experiments.
It is a regression fixture, not a scientific benchmark and must not be used for
performance claims.
"""

from __future__ import annotations

from typing import Any

from remem.memory.policy import GuidedActionPolicy

SMOKE_OBSERVATION = "task: open the drawer"
SMOKE_ACTION = "open drawer"
SMOKE_SUCCESS_OBSERVATION = "drawer opened"


class SmokeAlfWorldEnvironment:
    """Minimal batch-shaped environment compatible with ``AlfWorldAdapter``."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.closed = False
        self.step_count = 0

    def reset(self, **kwargs: Any) -> list[str]:
        """Reset the episode and return one textual task observation."""

        self.step_count = 0
        return [SMOKE_OBSERVATION]

    def step(
        self, actions: list[str]
    ) -> tuple[list[str], list[float], list[bool], list[dict[str, Any]]]:
        """Accept the target action and finish the deterministic episode."""

        if len(actions) != 1:
            raise ValueError("smoke environment expects one action")
        self.step_count += 1
        success = actions[0] == SMOKE_ACTION
        observation = SMOKE_SUCCESS_OBSERVATION if success else "drawer remains closed"
        return [observation], [1.0 if success else 0.0], [True], [{"seed": self.seed}]

    def close(self) -> None:
        """Mark the environment as closed."""

        self.closed = True


def build_environment(seed: int) -> SmokeAlfWorldEnvironment:
    """Build one deterministic ALFWorld-shaped smoke environment."""

    return SmokeAlfWorldEnvironment(seed)


def build_action_policy(seed: int) -> GuidedActionPolicy:
    """Build an action policy that requires memory guidance after warm-up.

    Seed zero represents a baseline warm-up episode that establishes the useful
    memory. Later seeds intentionally return the target action only when the
    ReMemAgent guidance contains that action, making the integration test prove
    that guidance crosses the policy boundary.
    """

    def action_policy(current_state: str, guidance: str) -> str:
        """Select the target action only when the expected evidence is present."""

        if seed == 0:
            return SMOKE_ACTION
        if f"Relevant action: {SMOKE_ACTION}" in guidance:
            return SMOKE_ACTION
        return "wait"

    return action_policy


def is_success(episode: Any) -> bool:
    """Return whether the smoke episode reached the successful terminal state."""

    return (
        bool(episode.steps)
        and episode.steps[-1].action == SMOKE_ACTION
        and episode.total_reward > 0
    )


__all__ = [
    "SMOKE_ACTION",
    "SMOKE_OBSERVATION",
    "SMOKE_SUCCESS_OBSERVATION",
    "SmokeAlfWorldEnvironment",
    "build_action_policy",
    "build_environment",
    "is_success",
]
