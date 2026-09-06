"""Regression tests for the concrete verl AgentLoopBase bridge."""

from __future__ import annotations

import asyncio

import pytest

import remem.integrations.verl_agent_loop as verl_agent_loop


class FakeAgentLoopBase:
    """Minimal stand-in for verl's AgentLoopBase in dependency-free tests."""


class FakeAgentLoopOutput:
    """Minimal stand-in for verl's AgentLoopOutput model."""

    def __init__(self, **fields: object) -> None:
        self.fields = fields


class CustomAgentLoopOutput(FakeAgentLoopOutput):
    """Stand-in output type supplied by a compatible verl fork."""


def _fake_verl_types(
    *,
    output_factory: verl_agent_loop.AgentLoopOutputFactory | None = None,
) -> tuple[type[object], verl_agent_loop.AgentLoopOutputFactory]:
    """Return fake runtime types while honoring the injected factory contract."""

    return FakeAgentLoopBase, output_factory or FakeAgentLoopOutput


def test_build_verl_agent_loop_class_runs_injected_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The concrete adapter forwards sampling parameters and dataset fields."""

    monkeypatch.setattr(verl_agent_loop, "_load_verl_types", _fake_verl_types)
    calls: list[tuple[dict[str, object], dict[str, object]]] = []

    async def runner(
        sampling_params: dict[str, object],
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        calls.append((sampling_params, kwargs))
        return {
            "prompt_ids": [1, 2],
            "response_ids": [3, 4],
            "response_mask": [1, 0],
        }

    loop_class = verl_agent_loop.build_verl_agent_loop_class(runner)
    output = asyncio.run(
        loop_class().run(
            {"temperature": 0.2},
            raw_prompt=[{"role": "user", "content": "hello"}],
        )
    )

    assert isinstance(output, FakeAgentLoopOutput)
    assert output.fields == {
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "response_mask": [1, 0],
        "extra_fields": {},
    }
    assert calls == [
        (
            {"temperature": 0.2},
            {"raw_prompt": [{"role": "user", "content": "hello"}]},
        )
    ]


def test_build_verl_agent_loop_class_accepts_custom_output_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatible forks can inject their concrete AgentLoopOutput model."""

    monkeypatch.setattr(verl_agent_loop, "_load_verl_types", _fake_verl_types)

    async def runner(
        sampling_params: dict[str, object],
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        return {"prompt_ids": [1], "response_ids": [2], "response_mask": [1]}

    loop_class = verl_agent_loop.build_verl_agent_loop_class(
        runner,
        output_factory=CustomAgentLoopOutput,
    )
    output = asyncio.run(loop_class().run({}))

    assert isinstance(output, CustomAgentLoopOutput)


def test_build_verl_agent_loop_class_rejects_invalid_runner() -> None:
    """The factory fails before importing verl for non-callable runners."""

    with pytest.raises(TypeError, match="runner must be callable"):
        verl_agent_loop.build_verl_agent_loop_class(None)  # type: ignore[arg-type]


def test_build_verl_agent_loop_class_rejects_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed runner output cannot reach the external training framework."""

    monkeypatch.setattr(verl_agent_loop, "_load_verl_types", _fake_verl_types)

    async def runner(
        sampling_params: dict[str, object],
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        return {
            "prompt_ids": [1],
            "response_ids": [-2],
            "response_mask": [1],
        }

    loop_class = verl_agent_loop.build_verl_agent_loop_class(runner)

    with pytest.raises(ValueError, match="response_ids must contain non-negative token IDs"):
        asyncio.run(loop_class().run({}))


def test_build_verl_agent_loop_class_preserves_extra_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic agent-loop fields survive validation and reach AgentLoopOutput."""

    monkeypatch.setattr(verl_agent_loop, "_load_verl_types", _fake_verl_types)

    async def runner(
        sampling_params: dict[str, object],
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        return {
            "prompt_ids": [1],
            "response_ids": [2],
            "response_mask": [1],
            "extra_fields": {"memory_ids": ["mem-1"]},
        }

    loop_class = verl_agent_loop.build_verl_agent_loop_class(runner)
    output = asyncio.run(loop_class().run({}))

    assert output.fields["extra_fields"] == {"memory_ids": ["mem-1"]}
