"""Optional runtime bridge to the installed verl AgentLoopOutput model.

The core integration in :mod:`remem.integrations.verl` deliberately avoids a
verl dependency. This module provides the complementary runtime boundary for
experiments that have verl installed: it imports the concrete AgentLoopOutput
model lazily and maps a validated ReMemAgent trajectory into it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from remem.integrations.verl import VerlTrajectory

AgentLoopOutputFactory = Callable[..., object]


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
