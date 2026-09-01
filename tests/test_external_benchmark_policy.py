from __future__ import annotations

import pytest

import experiments.external_benchmark as external_benchmark
from experiments.external_benchmark import ExternalBenchmarkSpec, _resolve_policy_factory
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


def test_policy_specifications_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _base_spec(policy_factory="example:make_policy")


def test_one_policy_specification_is_required() -> None:
    with pytest.raises(ValueError, match="one of policy_factory"):
        _base_spec(action_policy_factory=None)


def test_minimum_trust_is_validated() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _base_spec(minimum_trust=1.1)
