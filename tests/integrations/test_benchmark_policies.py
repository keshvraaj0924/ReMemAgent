"""Tests for benchmark-specific learned policy adapters."""

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


def test_alfworld_prompt_preserves_observation_and_action_boundary() -> None:
    prompt = build_alfworld_prompt("You are in the kitchen. There is a cup.")

    assert "Observation:" in prompt
    assert "You are in the kitchen. There is a cup." in prompt
    assert prompt.endswith("Action:")


def test_parse_alfworld_action_prefers_final_action_line() -> None:
    generated = "I should inspect the room.\nAction: take cup from table"

    assert parse_alfworld_action(generated) == "take cup from table"


def test_parse_alfworld_action_rejects_empty_output() -> None:
    with pytest.raises(ValueError, match="generated_text"):
        parse_alfworld_action("  \n")


def test_webshop_prompt_requires_canonical_action_syntax() -> None:
    prompt = build_webshop_prompt("Search results for running shoes")

    assert "search[query]" in prompt
    assert "click[target]" in prompt
    assert "Search results for running shoes" in prompt


def test_parse_webshop_action_extracts_search_action() -> None:
    assert parse_webshop_action("Reasoning...\nsearch[running shoes]") == "search[running shoes]"


def test_parse_webshop_action_extracts_click_action() -> None:
    assert parse_webshop_action("click[buy now]") == "click[buy now]"


def test_parse_webshop_action_rejects_non_action_text() -> None:
    with pytest.raises(ValueError, match="canonical WebShop action"):
        parse_webshop_action("I will buy the product.")


def test_alfworld_factory_binds_benchmark_prompt_and_parser() -> None:
    calls: list[tuple[object, str, dict[str, object]]] = []

    def loader(*args: object, **kwargs: object) -> object:
        def generator(prompt: str, **generation_kwargs: object) -> list[dict[str, str]]:
            calls.append((args, prompt, generation_kwargs))
            return [{"generated_text": "Action: open cabinet"}]

        return generator

    factory = build_alfworld_huggingface_policy_factory("test-model", pipeline_loader=loader)
    policy = factory(7)

    assert policy("The kitchen is visible.") == "open cabinet"
    assert calls[0][0] == ("text-generation",)


def test_webshop_factory_binds_benchmark_prompt_and_parser() -> None:
    def loader(*args: object, **kwargs: object) -> object:
        def generator(prompt: str, **generation_kwargs: object) -> list[dict[str, str]]:
            return [{"generated_text": "click[buy now]"}]

        return generator

    factory = build_webshop_huggingface_policy_factory("test-model", pipeline_loader=loader)
    policy = factory(11)

    assert policy("A product page is open.") == "click[buy now]"
