"""Tests for external benchmark environment adapters."""

from __future__ import annotations

import pytest

from remem.environments import AlfWorldAdapter, StepResult, WebShopAdapter


class FakeEnvironment:
    """Small environment double covering legacy and Gymnasium-style APIs."""

    def __init__(self, reset_result: object, step_result: object) -> None:
        self.reset_result = reset_result
        self.step_result = step_result
        self.closed = False
        self.actions: list[object] = []

    def reset(self, **_kwargs: object) -> object:
        return self.reset_result

    def step(self, action: object) -> object:
        self.actions.append(action)
        return self.step_result

    def close(self) -> None:
        self.closed = True


class SingletonBatch:
    """Array-like batch double that is not a ``collections.abc.Sequence``."""

    def __init__(self, value: object) -> None:
        self.value = value

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self.value


def test_alfworld_adapter_normalizes_gymnasium_step() -> None:
    environment = FakeEnvironment(
        ("look around", {"episode": 1}),
        ("found object", 1, True, False, {"score": 1}),
    )
    adapter = AlfWorldAdapter(environment)

    assert adapter.reset() == "look around"
    assert adapter.step("take object") == StepResult(
        observation="found object",
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"score": 1},
    )
    assert environment.actions == [["take object"]]
    adapter.close()
    assert environment.closed


def test_alfworld_adapter_unwraps_upstream_single_item_batches() -> None:
    environment = FakeEnvironment(
        (["look around"], {"episode": [7]}),
        (["found object"], [1.0], [True], [False], {"score": [1], "won": [True]}),
    )
    adapter = AlfWorldAdapter(environment)

    assert adapter.reset() == "look around"
    assert adapter.step("take object") == StepResult(
        observation="found object",
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"score": 1, "won": True},
    )


def test_alfworld_adapter_unwraps_array_like_singleton_batches() -> None:
    environment = FakeEnvironment(
        SingletonBatch("look around"),
        (
            SingletonBatch("found object"),
            SingletonBatch(1.0),
            SingletonBatch(True),
            SingletonBatch(False),
            {"score": SingletonBatch(1)},
        ),
    )
    adapter = AlfWorldAdapter(environment)

    assert adapter.reset() == "look around"
    assert adapter.step("take object") == StepResult(
        observation="found object",
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"score": 1},
    )


def test_webshop_adapter_normalizes_legacy_step() -> None:
    adapter = WebShopAdapter(
        FakeEnvironment("search page", ("result", 0.5, True, {"query": "shoes"}))
    )

    assert adapter.reset() == "search page"
    result = adapter.step("click[12]")

    assert result.observation == "result"
    assert result.reward == 0.5
    assert result.terminated
    assert not result.truncated
    assert result.info == {"query": "shoes"}


def test_adapter_rejects_empty_actions() -> None:
    adapter = AlfWorldAdapter(FakeEnvironment("state", ("next", 0, False, {})))

    with pytest.raises(ValueError, match="non-empty string"):
        adapter.step(" ")


def test_adapter_rejects_missing_environment_methods() -> None:
    with pytest.raises(TypeError, match="reset"):
        AlfWorldAdapter(object())
