"""Contract tests for the official WebShop text environment boundary."""

from __future__ import annotations

import pytest

from remem.environments.webshop import WebShopAdapter


class _OfficialWebShopEnvironment:
    """Minimal double matching upstream WebAgentTextEnv reset/step outputs."""

    def __init__(self) -> None:
        self.dispatched_actions: list[str] = []

    def reset(self):
        return "instruction [SEP] Search", None

    def step(self, action: str):
        self.dispatched_actions.append(action)
        return "results [SEP] Product", 0.75, False, None


def test_webshop_adapter_supports_official_metadata_sentinels() -> None:
    environment = _OfficialWebShopEnvironment()
    adapter = WebShopAdapter(environment)

    assert adapter.reset() == "instruction [SEP] Search"
    result = adapter.step("search[wireless headphones]")

    assert environment.dispatched_actions == ["search[wireless headphones]"]
    assert result.observation == "results [SEP] Product"
    assert result.reward == 0.75
    assert result.terminated is False
    assert result.truncated is False
    assert result.info == {}


class _MalformedWebShopEnvironment(_OfficialWebShopEnvironment):
    def reset(self):
        return "initial", "invalid-info"

    def step(self, action: str):
        return "next", 0.0, False, "invalid-info"


def test_webshop_adapter_does_not_relax_malformed_metadata() -> None:
    adapter = WebShopAdapter(_MalformedWebShopEnvironment())

    with pytest.raises(TypeError, match="reset info"):
        adapter.reset()
    with pytest.raises(TypeError, match="environment info"):
        adapter.step("search[item]")


class _GymnasiumWebShopEnvironment(_OfficialWebShopEnvironment):
    def step(self, action: str):
        return "next", 1.0, True, False, {"source": "wrapper"}


def test_webshop_adapter_preserves_gymnasium_metadata() -> None:
    result = WebShopAdapter(_GymnasiumWebShopEnvironment()).step("click[buy now]")

    assert result.terminated is True
    assert result.truncated is False
    assert result.info == {"source": "wrapper"}
