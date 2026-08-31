"""Tests for externally supplied benchmark execution boundaries."""

from __future__ import annotations

import sys
import types

import pytest

from experiments.external_benchmark import load_callable, run_external_benchmark
from remem.environments.base import StepResult


class _Environment:
    """Minimal environment used to exercise the real suite runner."""

    def __init__(self) -> None:
        self.closed = False

    def reset(self) -> str:
        return "task"

    def step(self, action: str) -> StepResult:
        return StepResult("done", 1.0, True, False, {"action": action})

    def close(self) -> None:
        self.closed = True


def _environment_factory(_: int) -> _Environment:
    return _Environment()


def _policy_factory(_: int, _store: object):
    return lambda _observation: "finish"


def _success_evaluator(episode: object) -> bool:
    return bool(getattr(episode, "terminated"))


def test_load_callable_supports_nested_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("test_external_factory_module")
    module.factory = types.SimpleNamespace(value=_environment_factory)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert load_callable("test_external_factory_module:factory.value") is _environment_factory


def test_load_callable_rejects_malformed_specification() -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        load_callable("not-a-spec")


def test_load_callable_rejects_non_callable_target(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("test_non_callable_module")
    module.value = 42
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(TypeError, match="non-callable"):
        load_callable("test_non_callable_module:value")


def test_run_external_benchmark_executes_supplied_environment_and_policy() -> None:
    report = run_external_benchmark(
        benchmark_name="external-smoke",
        episode_count=2,
        max_steps=3,
        environment_factory=_environment_factory,
        policy_factory=_policy_factory,
        success_evaluator=_success_evaluator,
    )

    assert report.benchmark_name == "external-smoke"
    assert len(report.episodes) == 2
    assert report.success_count == 2
    assert report.mean_reward == 1.0
    assert all(episode.episode.terminated for episode in report.episodes)
