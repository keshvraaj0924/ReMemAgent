"""Contract tests for benchmark environment adapters."""

from __future__ import annotations

from fractions import Fraction

import pytest

from remem.environments.alfworld import AlfWorldAdapter
from remem.environments.webshop import WebShopAdapter


class _LegacyEnvironment:
    def reset(self):
        return "initial", {"seed": 1}

    def step(self, action: str):
        return b"next", 0.5, True, {"action": action}


class _LegacyTruncatedEnvironment:
    def reset(self):
        return "initial"

    def step(self, action: str):
        return "next", 0.0, True, {"TimeLimit.truncated": True, "action": action}


class _GymnasiumEnvironment:
    def reset(self):
        return "initial"

    def step(self, action: str):
        return "next", 1, False, True, {"action": action}


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_normalizes_legacy_gym_contract(adapter_type) -> None:
    adapter = adapter_type(_LegacyEnvironment())

    result = adapter.step("look")

    assert adapter.reset() == "initial"
    assert result.observation == "next"
    assert result.reward == 0.5
    assert result.terminated is True
    assert result.truncated is False
    assert result.done is True
    assert result.info == {"action": "look"}


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_preserves_legacy_gym_time_limit_truncation(adapter_type) -> None:
    result = adapter_type(_LegacyTruncatedEnvironment()).step("look")

    assert result.terminated is False
    assert result.truncated is True
    assert result.done is True


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_normalizes_gymnasium_contract(adapter_type) -> None:
    result = adapter_type(_GymnasiumEnvironment()).step("search")

    assert result.terminated is False
    assert result.truncated is True
    assert result.done is True


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_rejects_blank_actions_before_external_dispatch(adapter_type) -> None:
    adapter = adapter_type(_LegacyEnvironment())

    with pytest.raises(ValueError, match="non-empty"):
        adapter.step("   ")


class _MalformedEnvironment:
    def reset(self):
        return {"not": "text"}

    def step(self, action: str):
        return "next", "invalid-reward", False, {}


def test_adapter_fails_closed_on_malformed_external_contract() -> None:
    adapter = AlfWorldAdapter(_MalformedEnvironment())

    with pytest.raises(TypeError, match="observation"):
        adapter.reset()
    with pytest.raises(TypeError, match="reward"):
        adapter.step("look")


class _MalformedResetEnvironment(_LegacyEnvironment):
    def __init__(self, reset_result) -> None:
        self.reset_result = reset_result

    def reset(self):
        return self.reset_result


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_rejects_reset_tuple_without_metadata_mapping(adapter_type) -> None:
    adapter = adapter_type(_MalformedResetEnvironment(("initial", "not-info")))

    with pytest.raises(TypeError, match="reset info"):
        adapter.reset()


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_rejects_reset_tuple_with_invalid_arity(adapter_type) -> None:
    adapter = adapter_type(_MalformedResetEnvironment(("initial", {}, "extra")))

    with pytest.raises(ValueError, match="observation and info"):
        adapter.reset()


class _NonFiniteRewardEnvironment:
    def __init__(self, reward: float) -> None:
        self.reward = reward

    def reset(self):
        return "initial"

    def step(self, action: str):
        return "next", self.reward, False, {"action": action}


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_rejects_non_finite_rewards(adapter_type, reward: float) -> None:
    adapter = adapter_type(_NonFiniteRewardEnvironment(reward))

    with pytest.raises(ValueError, match="finite"):
        adapter.step("look")


class _RealRewardEnvironment:
    def reset(self):
        return "initial"

    def step(self, action: str):
        return "next", Fraction(1, 4), False, {"action": action}


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_accepts_standard_real_reward_implementations(adapter_type) -> None:
    result = adapter_type(_RealRewardEnvironment()).step("look")

    assert result.reward == 0.25
    assert isinstance(result.reward, float)


class _MutableInfoEnvironment:
    def __init__(self) -> None:
        self.info = {"trace": {"actions": ["look"]}}

    def reset(self):
        return "initial"

    def step(self, action: str):
        return "next", 0.0, False, self.info


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_detaches_nested_transition_metadata(adapter_type) -> None:
    environment = _MutableInfoEnvironment()
    result = adapter_type(environment).step("look")

    environment.info["trace"]["actions"].append("mutated")

    assert result.info == {"trace": {"actions": ["look"]}}


class _ClosableEnvironment(_LegacyEnvironment):
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_context_manager_closes_wrapped_environment(adapter_type) -> None:
    environment = _ClosableEnvironment()

    with adapter_type(environment) as adapter:
        assert adapter.reset() == "initial"
        assert environment.close_calls == 0

    assert environment.close_calls == 1


@pytest.mark.parametrize("adapter_type", [AlfWorldAdapter, WebShopAdapter])
def test_adapter_close_is_safe_when_environment_has_no_close(adapter_type) -> None:
    adapter_type(_LegacyEnvironment()).close()


class _OfficialAlfWorldBatchEnvironment:
    def __init__(self) -> None:
        self.dispatched_actions: list[list[str]] = []

    def reset(self):
        return ["initial"], {
            "admissible_commands": [["look", "inventory"]],
            "won": [False],
            "extra.gamefile": ["task/game.tw-pddl"],
        }

    def step(self, actions: list[str]):
        self.dispatched_actions.append(actions)
        return ["next"], [1.0], [True], {
            "admissible_commands": [["inventory"]],
            "won": [True],
            "extra.gamefile": ["task/game.tw-pddl"],
        }


def test_alfworld_adapter_supports_official_batch_size_one_contract() -> None:
    environment = _OfficialAlfWorldBatchEnvironment()
    adapter = AlfWorldAdapter(environment)

    assert adapter.reset() == "initial"
    result = adapter.step("look")

    assert environment.dispatched_actions == [["look"]]
    assert result.observation == "next"
    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info == {
        "admissible_commands": ["inventory"],
        "won": True,
        "extra.gamefile": "task/game.tw-pddl",
    }


class _MultiItemAlfWorldBatchEnvironment(_OfficialAlfWorldBatchEnvironment):
    def reset(self):
        return ["first", "second"], {"won": [False, False]}


def test_alfworld_adapter_rejects_multi_episode_batch() -> None:
    adapter = AlfWorldAdapter(_MultiItemAlfWorldBatchEnvironment())

    with pytest.raises(ValueError, match="exactly one item"):
        adapter.reset()


class _MalformedAlfWorldStepEnvironment(_OfficialAlfWorldBatchEnvironment):
    def step(self, actions: list[str]):
        return ["first", "second"], [0.0], [False], {"won": [False]}


def test_alfworld_adapter_rejects_mismatched_batched_step_shape() -> None:
    adapter = AlfWorldAdapter(_MalformedAlfWorldStepEnvironment())
    adapter.reset()

    with pytest.raises(ValueError, match="exactly one item"):
        adapter.step("look")
