from __future__ import annotations

import pytest

import experiments.external_benchmark as external_benchmark
from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    _resolve_policy_factory,
    run_external_benchmark,
)
from remem.environments.base import StepResult
from remem.memory.policy import MemoryGuidedPolicy
from remem.memory.store import MemoryStore


def _base_spec(**overrides: object) -> ExternalBenchmarkSpec:
    values: dict[str, object] = {
        "benchmark_name": "alfworld-eval",
        "episode_count": 1,
        "max_steps": 4,
        "environment_factory": "example:make_environment",
        "policy_factory": None,
        "action_policy_factory": "example:make_action_policy",
        "success_evaluator": "example:is_success",
    }
    values.update(overrides)
    return ExternalBenchmarkSpec(**values)


def test_action_policy_factory_is_composed_with_memory_guidance(monkeypatch) -> None:
    def resolve_callable(specification: str):
        assert specification == "example:make_action_policy"
        return lambda seed: lambda state, guidance: f"action:{seed}:{state}:{guidance}"

    monkeypatch.setattr(external_benchmark, "resolve_callable", resolve_callable)
    spec = _base_spec(minimum_trust=0.7)

    factory = _resolve_policy_factory(spec)
    store = MemoryStore()
    policy = factory(17, store)

    assert isinstance(policy, MemoryGuidedPolicy)
    assert policy.minimum_trust == 0.7
    assert policy("observe") == "action:17:observe:"


def test_action_policy_factory_provenance_remains_distinct(monkeypatch) -> None:
    monkeypatch.setattr(
        external_benchmark,
        "load_benchmark_environment_factory",
        lambda *args: lambda seed: _FakeEnvironment(),
    )
    monkeypatch.setattr(
        external_benchmark,
        "resolve_callable",
        lambda specification: (
            (lambda seed: lambda state, guidance: "look")
            if specification == "example:make_action_policy"
            else (lambda episode: True)
        ),
    )

    report = run_external_benchmark(_base_spec(max_steps=1))

    assert report.configuration is not None
    assert report.configuration.policy_factory is None
    assert report.configuration.action_policy_factory == "example:make_action_policy"


def test_policy_specifications_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _base_spec(policy_factory="example:make_policy")


def test_one_policy_specification_is_required() -> None:
    with pytest.raises(ValueError, match="one of policy_factory"):
        _base_spec(action_policy_factory=None)


def test_minimum_trust_is_validated() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _base_spec(minimum_trust=1.1)


class _FakeEnvironment:
    def reset(self) -> str:
        return "initial"

    def step(self, action: str) -> StepResult:
        return StepResult(
            observation="done",
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"action": action},
        )

    def close(self) -> None:
        return None
