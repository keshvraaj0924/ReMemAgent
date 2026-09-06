"""Tests for benchmark-specific learned-policy adapters."""

from __future__ import annotations

import pytest

from remem.integrations.benchmark_policies import (
    build_alfworld_huggingface_policy_factory,
    build_alfworld_prompt,
    build_webshop_huggingface_policy_factory,
    build_webshop_prompt,
    parse_alfworld_action,
    parse_webshop_action,
)


def test_alfworld_prompt_contains_observation_and_action_boundary() -> None:
    prompt = build_alfworld_prompt("You are in a kitchen.")

    assert "Observation:\nYou are in a kitchen." in prompt
    assert prompt.endswith("Action:")


def test_webshop_prompt_contains_observation_and_action_boundary() -> None:
    prompt = build_webshop_prompt("Search results are visible.")

    assert "Observation:\nSearch results are visible." in prompt
    assert prompt.endswith("Action:")


def test_alfworld_parser_accepts_explicit_action() -> None:
    assert parse_alfworld_action("Action: open cabinet 1") == "open cabinet 1"


def test_alfworld_parser_accepts_current_move_verb() -> None:
    assert parse_alfworld_action("move apple 1 to table 1") == "move apple 1 to table 1"


def test_alfworld_parser_rejects_explanatory_prose() -> None:
    with pytest.raises(ValueError, match="unsupported command verb"):
        parse_alfworld_action("I should open the cabinet")


def test_alfworld_parser_rejects_unknown_command() -> None:
    with pytest.raises(ValueError, match="unsupported command verb"):
        parse_alfworld_action("teleport to the kitchen")


def test_alfworld_parser_rejects_multiple_actions() -> None:
    with pytest.raises(ValueError, match="multiple ALFWorld actions"):
        parse_alfworld_action("Action: open cabinet 1\nAction: close cabinet 1")


def test_webshop_parser_accepts_search_action() -> None:
    assert parse_webshop_action("Action: search[wireless keyboard]") == "search[wireless keyboard]"


def test_webshop_parser_accepts_click_action() -> None:
    assert parse_webshop_action("click[< Back to Search]") == "click[< Back to Search]"


def test_webshop_parser_rejects_multiple_actions() -> None:
    with pytest.raises(ValueError, match="exactly one canonical WebShop action"):
        parse_webshop_action("search[keyboard] then click[Buy]")


def test_webshop_parser_rejects_missing_action() -> None:
    with pytest.raises(ValueError, match="exactly one canonical WebShop action"):
        parse_webshop_action("I will search for a keyboard")


def test_alfworld_huggingface_factory_uses_injected_pipeline() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def pipeline_loader(task: str, **kwargs: object):
        calls.append((task, kwargs))

        def generate(prompt: str, **generation_kwargs: object):
            assert prompt.endswith("Action:")
            assert generation_kwargs["do_sample"] is False
            return [{"generated_text": "Action: look"}]

        return generate

    factory = build_alfworld_huggingface_policy_factory(
        "test-model",
        pipeline_loader=pipeline_loader,
    )
    policy = factory(7)

    assert policy("You are in a kitchen.") == "look"
    assert calls == [("text-generation", {"model": "test-model"})]


def test_webshop_huggingface_factory_uses_injected_pipeline() -> None:
    def pipeline_loader(task: str, **kwargs: object):
        assert task == "text-generation"
        assert kwargs["model"] == "test-model"

        def generate(_prompt: str, **_generation_kwargs: object):
            return [{"generated_text": "Action: search[shoes]"}]

        return generate

    factory = build_webshop_huggingface_policy_factory(
        "test-model",
        pipeline_loader=pipeline_loader,
    )
    policy = factory(11)

    assert policy("Find running shoes.") == "search[shoes]"
