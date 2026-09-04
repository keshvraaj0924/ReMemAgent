import math

import pytest

from remem.environments.webshop import WebShopAdapter


class FakeWebShop:
    def __init__(self, step_result):
        self.step_result = step_result

    def reset(self):
        return "initial observation", {"ignored": True}

    def step(self, action):
        assert action == "click[1]"
        return self.step_result

    def close(self):
        pass


def test_webshop_adapter_normalizes_valid_five_value_step() -> None:
    adapter = WebShopAdapter(FakeWebShop(("next", 1.0, True, False, {"score": 1})))

    assert adapter.reset() == "initial observation"
    result = adapter.step("click[1]")

    assert result.observation == "next"
    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info == {"score": 1}


def test_webshop_adapter_rejects_non_boolean_terminal_flags() -> None:
    adapter = WebShopAdapter(FakeWebShop(("next", 0.0, "false", False, {})))

    with pytest.raises(TypeError, match="terminated flag must be a boolean"):
        adapter.step("click[1]")


def test_webshop_adapter_rejects_boolean_reward() -> None:
    adapter = WebShopAdapter(FakeWebShop(("next", True, False, False, {})))

    with pytest.raises(TypeError, match="reward must be a finite numeric value"):
        adapter.step("click[1]")


@pytest.mark.parametrize("reward", [math.nan, math.inf, -math.inf])
def test_webshop_adapter_rejects_non_finite_reward(reward: float) -> None:
    adapter = WebShopAdapter(FakeWebShop(("next", reward, False, False, {})))

    with pytest.raises(ValueError, match="reward must be finite"):
        adapter.step("click[1]")


def test_webshop_adapter_supports_legacy_four_value_step() -> None:
    adapter = WebShopAdapter(FakeWebShop(("next", 0.5, True, {"done": True})))

    result = adapter.step("click[1]")

    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 0.5
