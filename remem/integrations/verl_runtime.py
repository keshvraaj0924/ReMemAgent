"""Optional runtime bridge to the installed verl AgentLoop API.

The core integration in :mod:`remem.integrations.verl` deliberately avoids a
verl dependency. This module provides the complementary runtime boundary for
experiments that have verl installed: it imports the concrete AgentLoop classes
lazily and maps validated ReMemAgent trajectories into them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from remem.integrations.verl import VerlTrajectory

AgentLoopOutputFactory = Callable[..., object]
TrajectoryRunner = Callable[
    [dict[str, Any], Mapping[str, object]], Awaitable[VerlTrajectory]
]


def build_verl_agent_loop_output(
    trajectory: VerlTrajectory,
    *,
    output_factory: AgentLoopOutputFactory | None = None,
) -> object:
    """Build the installed verl ``AgentLoopOutput`` for a trajectory.

    ``output_factory`` is injectable for tests and for compatible downstream
    verl forks. When omitted, the current upstream experimental Agent Loop
    location is imported lazily. ReMemAgent never imports verl at module import
    time, so the dependency-free research core remains usable without it.

    Reward is mapped to verl's ``reward_score`` field and ReMemAgent metadata is
    preserved under ``extra_fields['remem_metadata']``. Response log
    probabilities are passed through unchanged when present.
    """

    factory = output_factory or _load_agent_loop_output_factory()
    payload: dict[str, Any] = {
        "prompt_ids": list(trajectory.prompt_ids),
        "response_ids": list(trajectory.response_ids),
        "response_mask": list(trajectory.response_mask),
        "reward_score": trajectory.reward,
        "extra_fields": {"remem_metadata": dict(trajectory.metadata)},
    }
    if trajectory.response_logprobs is not None:
        payload["response_logprobs"] = list(trajectory.response_logprobs)
    return factory(**payload)


def build_verl_agent_loop_class(
    runner: TrajectoryRunner,
    *,
    output_factory: AgentLoopOutputFactory | None = None,
) -> type[object]:
    """Build a concrete ``verl.AgentLoopBase`` subclass around a runner.

    The returned class implements only verl's ``run`` seam. The injected
    ``runner`` owns model generation, environment interaction, reward
    computation, and tokenization; it receives the sampling parameters plus
    dataset fields forwarded by verl. The runner must return a validated
    :class:`VerlTrajectory`, which this bridge converts to the installed
    ``AgentLoopOutput`` type.

    The base class constructor is intentionally inherited unchanged so verl
    remains responsible for constructing the loop with its current runtime
    dependencies. This avoids coupling ReMemAgent to a particular experimental
    constructor signature.
    """

    agent_loop_base, resolved_output_factory = _load_agent_loop_runtime_types(
        output_factory=output_factory
    )

    class ReMemAgentLoop(agent_loop_base):  # type: ignore[misc, valid-type]
        """verl AgentLoop implementation backed by an injected runner."""

        async def run(
            self,
            sampling_params: dict[str, Any],
            **kwargs: object,
        ) -> object:
            """Execute the injected runner and return a concrete verl output."""

            trajectory = await runner(sampling_params, kwargs)
            if not isinstance(trajectory, VerlTrajectory):
                raise TypeError("verl agent-loop runner must return VerlTrajectory")
            return build_verl_agent_loop_output(
                trajectory,
                output_factory=resolved_output_factory,
            )

    ReMemAgentLoop.__name__ = "ReMemAgentLoop"
    ReMemAgentLoop.__qualname__ = "ReMemAgentLoop"
    return ReMemAgentLoop


def _load_agent_loop_runtime_types(
    *,
    output_factory: AgentLoopOutputFactory | None = None,
) -> tuple[type[object], AgentLoopOutputFactory]:
    """Resolve verl's base and output types lazily with precise errors."""

    try:
        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopBase,
            AgentLoopOutput,
        )
    except ModuleNotFoundError as exc:
        if exc.name is not None and (
            exc.name == "verl" or exc.name.startswith("verl.")
        ):
            raise RuntimeError(
                "verl is required for runtime AgentLoop construction; "
                "install a compatible verl release or pass runtime types through "
                "a downstream adapter"
            ) from exc
        raise
    return AgentLoopBase, output_factory or AgentLoopOutput


def _load_agent_loop_output_factory() -> AgentLoopOutputFactory:
    """Resolve the current upstream verl AgentLoopOutput class lazily.

    Only a genuinely missing ``verl`` installation is converted into the
    integration-specific runtime error. Import failures raised by a present
    but broken/incompatible ``verl`` dependency chain are preserved so the
    caller receives the original diagnostic rather than a misleading message.
    """

    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    except ModuleNotFoundError as exc:
        if exc.name is not None and (
            exc.name == "verl" or exc.name.startswith("verl.")
        ):
            raise RuntimeError(
                "verl is required for runtime AgentLoopOutput construction; "
                "install a compatible verl release or pass output_factory explicitly"
            ) from exc
        raise
    return AgentLoopOutput
