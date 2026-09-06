"""Concrete optional bridge for registering ReMemAgent with verl AgentLoopBase.

This module keeps the dependency on ``verl`` lazy. Callers provide the actual
model/environment runner, while the returned class owns only the framework
adapter contract and token-output validation. This makes the boundary usable
with the upstream experimental Agent Loop without coupling the research core
to a particular inference server or model stack.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from remem.integrations.verl_contract import validate_agent_loop_output

ExternalAgentRunner = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Awaitable[Mapping[str, object]],
]


def build_verl_agent_loop_class(
    runner: ExternalAgentRunner,
) -> type[object]:
    """Build an ``AgentLoopBase`` subclass around an injected async runner.

    The runner receives ``sampling_params`` and the dataset fields supplied to
    ``AgentLoopBase.run`` as two mappings. It must return the token-level
    ReMemAgent/verl contract. The generated class validates that contract before
    returning a concrete ``AgentLoopOutput`` instance.

    ``verl`` is imported only when this factory is called. A missing installation
    therefore does not affect normal ReMemAgent imports or dependency-free
    research experiments.
    """

    if not callable(runner):
        raise TypeError("runner must be callable")

    agent_loop_base, output_factory = _load_verl_types()

    class ReMemAgentAgentLoop(agent_loop_base):  # type: ignore[misc,valid-type]
        """verl AgentLoopBase adapter backed by an injected async runner."""

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> object:
            """Run the injected agent and return a validated verl output."""

            output = await runner(sampling_params, kwargs)
            validated = validate_agent_loop_output(output)
            return output_factory(**validated.to_dict())

    return ReMemAgentAgentLoop


def _load_verl_types() -> tuple[type[object], Callable[..., object]]:
    """Load the current upstream AgentLoopBase and AgentLoopOutput lazily."""

    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
    except ModuleNotFoundError as exc:
        if exc.name is not None and (
            exc.name == "verl" or exc.name.startswith("verl.")
        ):
            raise RuntimeError(
                "verl is required to build an AgentLoopBase adapter; "
                "install a compatible verl release"
            ) from exc
        raise
    return AgentLoopBase, AgentLoopOutput
