"""Tests for external benchmark integration bridges."""

from __future__ import annotations

import pytest

from experiments.integrations import build_environment_factory
from remem.environments.alfworld import AlfWorldAdapter
from remem.environments.webshop import WebShopAdapter


class _Environment:
    """Minimal environment accepted by both protocol adapters."""

    def reset(self) -> str:
        return "task"

    def step(self, action: str):
        return "done", 1.0, True, {"action": action}


def _factory(seed: int) -> _Environment:
    assert seed == 7
    return _Environment()


def test_build_environment_factory_wraps_alfworld() -> None:
    factory = build_environment_factory(_factory, benchmark="ALFWorld")

    environment = factory(7)

    assert isinstance(environment, AlfWorldAdapter)
    assert environment.reset() == "task"
    assert environment.step("look").terminated is True


def test_build_environment_factory_wraps_webshop() -> None:
    factory = build_environment_factory(_factory, benchmark="webshop")

    environment = factory(7)

    assert isinstance(environment, WebShopAdapter)
    assert environment.reset() == "task"
    assert environment.step("search").terminated is True


def test_build_environment_factory_rejects_unknown_benchmark() -> None:
    with pytest.raises(ValueError, match="unsupported benchmark"):
        build_environment_factory(_factory, benchmark="unknown")
