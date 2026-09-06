"""Tests for the optional concrete verl AgentLoopOutput bridge."""

from dataclasses import dataclass, field

import pytest

from remem.integrations.verl import VerlTrajectory
from remem.integrations.verl_runtime import build_verl_agent_loop_output


@dataclass(frozen=True)
class FakeAgentLoopOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    reward_score: float
    extra_fields: dict[str, object] = field(default_factory=dict)
    response_logprobs: list[float] | None = None


def _trajectory() -> VerlTrajectory:
    return VerlTrajectory(
        prompt_ids=(1, 2),
        response_ids=(3, 4),
        response_mask=(1, 0),
        reward=0.75,
        metadata={"memory_ids": ["memory-a"], "step_count": 2},
        response_logprobs=(-0.1, -0.2),
    )


def test_build_verl_agent_loop_output_maps_core_fields() -> None:
    output = build_verl_agent_loop_output(
        _trajectory(),
        output_factory=FakeAgentLoopOutput,
    )

    assert output == FakeAgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 0],
        reward_score=0.75,
        response_logprobs=[-0.1, -0.2],
        extra_fields={
            "remem_metadata": {"memory_ids": ["memory-a"], "step_count": 2}
        },
    )


def test_build_verl_agent_loop_output_omits_absent_logprobs() -> None:
    trajectory = VerlTrajectory(
        prompt_ids=(1,),
        response_ids=(2,),
        response_mask=(1,),
        reward=1.0,
        metadata={},
    )

    output = build_verl_agent_loop_output(
        trajectory,
        output_factory=FakeAgentLoopOutput,
    )

    assert output.response_logprobs is None


def test_build_verl_agent_loop_output_reports_missing_verl(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_verl_import(name: str, *args: object, **kwargs: object):
        if name.startswith("verl"):
            raise ModuleNotFoundError("verl unavailable", name="verl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_verl_import)

    with pytest.raises(RuntimeError, match="verl is required"):
        build_verl_agent_loop_output(_trajectory())


def test_build_verl_agent_loop_output_preserves_transitive_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_transitive_import(name: str, *args: object, **kwargs: object):
        if name.startswith("verl"):
            raise ModuleNotFoundError("broken dependency", name="ray")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_transitive_import)

    with pytest.raises(ModuleNotFoundError, match="broken dependency"):
        build_verl_agent_loop_output(_trajectory())
