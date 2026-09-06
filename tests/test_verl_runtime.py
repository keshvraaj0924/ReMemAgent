"""Tests for the optional concrete verl AgentLoop runtime bridge."""

from dataclasses import dataclass, field
import sys
import types

import pytest

from remem.integrations.verl import VerlTrajectory
from remem.integrations.verl_runtime import (
    build_verl_agent_loop_class,
    build_verl_agent_loop_output,
)


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


def test_verl_trajectory_detaches_metadata_mapping() -> None:
    metadata = {"memory_ids": ["memory-a"]}
    trajectory = VerlTrajectory(
        prompt_ids=(1,),
        response_ids=(2,),
        response_mask=(1,),
        reward=1.0,
        metadata=metadata,
    )

    metadata["memory_ids"] = ["memory-b"]

    assert trajectory.metadata == {"memory_ids": ["memory-a"]}


def test_build_verl_agent_loop_class_forwards_runner_and_output() -> None:
    class FakeAgentLoopBase:
        pass

    module = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    module.AgentLoopBase = FakeAgentLoopBase
    module.AgentLoopOutput = FakeAgentLoopOutput
    modules = {
        "verl": types.ModuleType("verl"),
        "verl.experimental": types.ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": types.ModuleType("verl.experimental.agent_loop"),
        "verl.experimental.agent_loop.agent_loop": module,
    }
    original_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)

    calls: list[tuple[dict[str, object], dict[str, object]]] = []

    async def runner(
        sampling_params: dict[str, object],
        fields: dict[str, object],
    ) -> VerlTrajectory:
        calls.append((sampling_params, fields))
        return _trajectory()

    try:
        loop_class = build_verl_agent_loop_class(runner)
        output = __import__("asyncio").run(
            loop_class().run({"temperature": 0.2}, prompt="hello", seed=7)
        )
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

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
    assert calls == [({"temperature": 0.2}, {"prompt": "hello", "seed": 7})]


def test_build_verl_agent_loop_class_rejects_invalid_runner_result() -> None:
    class FakeAgentLoopBase:
        pass

    module = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    module.AgentLoopBase = FakeAgentLoopBase
    module.AgentLoopOutput = FakeAgentLoopOutput
    modules = {
        "verl": types.ModuleType("verl"),
        "verl.experimental": types.ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": types.ModuleType("verl.experimental.agent_loop"),
        "verl.experimental.agent_loop.agent_loop": module,
    }
    original_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)

    async def runner(
        sampling_params: dict[str, object],
        fields: dict[str, object],
    ) -> object:
        return object()

    try:
        loop_class = build_verl_agent_loop_class(runner)
        with pytest.raises(TypeError, match="must return VerlTrajectory"):
            __import__("asyncio").run(loop_class().run({}, prompt="hello"))
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


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
