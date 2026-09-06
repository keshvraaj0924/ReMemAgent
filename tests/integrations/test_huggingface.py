from __future__ import annotations

from typing import Any

import pytest

from remem.integrations.huggingface import build_huggingface_text_action_policy_factory


def test_huggingface_policy_factory_loads_pipeline_lazily_and_parses_action() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def loader(task: str, *, model: str, **kwargs: Any) -> Any:
        calls.append((task, model, kwargs))

        def generate(prompt: str, **generation_kwargs: Any) -> list[dict[str, str]]:
            assert prompt == "OBS: open the drawer"
            assert generation_kwargs["do_sample"] is False
            return [{"generated_text": "thought\nopen drawer"}]

        return generate

    factory = build_huggingface_text_action_policy_factory(
        "test-model",
        prompt_builder=lambda observation: f"OBS: {observation}",
        pipeline_loader=loader,
    )

    assert calls == []
    policy = factory(7)
    assert calls == [("text-generation", "test-model", {})]
    assert policy("open the drawer") == "open drawer"

    second_policy = factory(8)
    assert calls == [("text-generation", "test-model", {})]
    assert second_policy("open the drawer") == "open drawer"


def test_huggingface_policy_factory_supports_custom_action_parser() -> None:
    def loader(*args: Any, **kwargs: Any) -> Any:
        return lambda prompt, **generation_kwargs: [{"generated_text": "Action: take apple"}]

    factory = build_huggingface_text_action_policy_factory(
        "test-model",
        prompt_builder=str,
        action_parser=lambda text: text.removeprefix("Action: ").strip(),
        pipeline_loader=loader,
    )

    assert factory(1)("take apple") == "take apple"


def test_huggingface_policy_factory_rejects_invalid_pipeline_output() -> None:
    factory = build_huggingface_text_action_policy_factory(
        "test-model",
        prompt_builder=str,
        pipeline_loader=lambda *args, **kwargs: lambda prompt, **generation_kwargs: [],
    )

    with pytest.raises(ValueError, match="non-empty list"):
        factory(1)("observe")


def test_huggingface_policy_factory_handles_chat_generation_output() -> None:
    factory = build_huggingface_text_action_policy_factory(
        "test-model",
        prompt_builder=str,
        pipeline_loader=lambda *args, **kwargs: lambda prompt, **generation_kwargs: [
            {
                "generated_text": [
                    {"role": "user", "content": "observe"},
                    {"role": "assistant", "content": "take apple"},
                ]
            }
        ],
    )

    assert factory(1)("observe") == "take apple"
