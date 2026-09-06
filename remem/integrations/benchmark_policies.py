"""Benchmark-specific prompt and action adapters for learned text policies.

The adapters in this module define only the textual contract between an
upstream benchmark and a learned language model. Memory routing remains in
``remem.integrations.policies`` and model loading remains in
``remem.integrations.huggingface``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from remem.integrations.huggingface import build_huggingface_text_action_policy_factory
from remem.integrations.policies import ActionPolicyFactory


_ALFWORLD_ACTION_PREFIX = "Action:"
_WEBSHOP_ACTION_PATTERN = re.compile(r"\b(?:search|click)\[[^\]\n]+\]", re.IGNORECASE)


def build_alfworld_prompt(observation: str) -> str:
    """Build a conservative ALFWorld text-action prompt from one observation."""

    _validate_observation(observation)
    return (
        "You are an ALFWorld text agent. Read the current observation and output "
        "exactly one executable text action. Do not explain your choice.\n\n"
        f"Observation:\n{observation.strip()}\n\nAction:"
    )


def parse_alfworld_action(generated_text: str) -> str:
    """Extract one ALFWorld action from model-generated text."""

    _validate_generated_text(generated_text)
    lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
    for line in reversed(lines):
        normalized = line.strip("` ")
        if normalized.lower().startswith(_ALFWORLD_ACTION_PREFIX.lower()):
            normalized = normalized[len(_ALFWORLD_ACTION_PREFIX) :].strip()
        if normalized:
            return normalized
    raise ValueError("generated text contains no usable ALFWorld action")


def build_webshop_prompt(observation: str) -> str:
    """Build a WebShop prompt requiring one canonical text action."""

    _validate_observation(observation)
    return (
        "You are a WebShop text agent. Output exactly one executable action. "
        "Valid actions are search[query] or click[target]. Do not explain your choice.\n\n"
        f"Observation:\n{observation.strip()}\n\nAction:"
    )


def parse_webshop_action(generated_text: str) -> str:
    """Extract the first canonical WebShop search/click action from model output."""

    _validate_generated_text(generated_text)
    match = _WEBSHOP_ACTION_PATTERN.search(generated_text)
    if match is None:
        raise ValueError("generated text contains no canonical WebShop action")
    return match.group(0).strip()


def build_alfworld_huggingface_policy_factory(
    model_name: str,
    *,
    pipeline_loader: Callable[..., object] | None = None,
    pipeline_kwargs: dict[str, object] | None = None,
    generation_kwargs: dict[str, object] | None = None,
) -> ActionPolicyFactory:
    """Build a Hugging Face action-policy factory using ALFWorld text semantics."""

    return build_huggingface_text_action_policy_factory(
        model_name,
        prompt_builder=build_alfworld_prompt,
        action_parser=parse_alfworld_action,
        pipeline_loader=pipeline_loader,
        pipeline_kwargs=pipeline_kwargs,
        generation_kwargs=generation_kwargs,
    )


def build_webshop_huggingface_policy_factory(
    model_name: str,
    *,
    pipeline_loader: Callable[..., object] | None = None,
    pipeline_kwargs: dict[str, object] | None = None,
    generation_kwargs: dict[str, object] | None = None,
) -> ActionPolicyFactory:
    """Build a Hugging Face action-policy factory using WebShop text semantics."""

    return build_huggingface_text_action_policy_factory(
        model_name,
        prompt_builder=build_webshop_prompt,
        action_parser=parse_webshop_action,
        pipeline_loader=pipeline_loader,
        pipeline_kwargs=pipeline_kwargs,
        generation_kwargs=generation_kwargs,
    )


def _validate_observation(observation: object) -> None:
    """Reject missing benchmark observations before prompt construction."""

    if not isinstance(observation, str) or not observation.strip():
        raise ValueError("observation must be a non-empty string")


def _validate_generated_text(generated_text: object) -> None:
    """Reject missing model output before benchmark-specific parsing."""

    if not isinstance(generated_text, str) or not generated_text.strip():
        raise ValueError("generated_text must be a non-empty string")


__all__ = [
    "build_alfworld_huggingface_policy_factory",
    "build_alfworld_prompt",
    "build_webshop_huggingface_policy_factory",
    "build_webshop_prompt",
    "parse_alfworld_action",
    "parse_webshop_action",
]
