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
_ALFWORLD_ACTION_VERBS = frozenset(
    {
        "pass",
        "go",
        "goto",
        "move",
        "take",
        "pick",
        "put",
        "open",
        "close",
        "toggle",
        "heat",
        "clean",
        "cool",
        "slice",
        "inventory",
        "examine",
        "look",
        "use",
    }
)
_WEBSHOP_ACTION_PREFIX = "Action:"
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
    """Extract one unambiguous ALFWorld action from model-generated text.

    The first token is checked against the known ALFWorld command vocabulary.
    This prevents explanatory prose such as ``I should open the cabinet`` from
    being passed to the environment as though it were an executable action,
    while preserving arbitrary object and receptacle arguments.
    """

    _validate_generated_text(generated_text)
    lines = [line.strip("` ") for line in generated_text.splitlines() if line.strip()]
    action_lines = [
        line[len(_ALFWORLD_ACTION_PREFIX) :].strip()
        for line in lines
        if line.lower().startswith(_ALFWORLD_ACTION_PREFIX.lower())
    ]
    if len(action_lines) == 1 and action_lines[0]:
        return _validate_alfworld_action(action_lines[0])
    if len(action_lines) > 1:
        raise ValueError("generated text contains multiple ALFWorld actions")
    if len(lines) == 1:
        return _validate_alfworld_action(lines[0])
    raise ValueError("generated text contains no unambiguous ALFWorld action")


def build_webshop_prompt(observation: str) -> str:
    """Build a WebShop prompt requiring one canonical text action."""

    _validate_observation(observation)
    return (
        "You are a WebShop text agent. Output exactly one executable action. "
        "Valid actions are search[query] or click[target]. Do not explain your choice.\n\n"
        f"Observation:\n{observation.strip()}\n\nAction:"
    )


def parse_webshop_action(generated_text: str) -> str:
    """Extract one unambiguous canonical WebShop action from model output."""

    _validate_generated_text(generated_text)
    lines = [line.strip("` ") for line in generated_text.splitlines() if line.strip()]
    prefixed_lines = [
        line[len(_WEBSHOP_ACTION_PREFIX) :].strip()
        for line in lines
        if line.lower().startswith(_WEBSHOP_ACTION_PREFIX.lower())
    ]
    if len(prefixed_lines) == 1:
        return _extract_single_webshop_action(prefixed_lines[0])
    if len(prefixed_lines) > 1:
        raise ValueError("generated text contains multiple WebShop action lines")

    matches = _WEBSHOP_ACTION_PATTERN.findall(generated_text)
    if len(matches) != 1:
        raise ValueError("generated text must contain exactly one canonical WebShop action")
    return matches[0].strip()


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


def _extract_single_webshop_action(text: str) -> str:
    """Extract exactly one canonical action from an explicit WebShop action line."""

    matches = _WEBSHOP_ACTION_PATTERN.findall(text)
    if len(matches) != 1:
        raise ValueError("WebShop Action: line must contain exactly one canonical action")
    return matches[0].strip()


def _validate_alfworld_action(action: str) -> str:
    """Require a recognized ALFWorld command verb before execution."""

    normalized_action = action.strip()
    first_token = normalized_action.split(maxsplit=1)[0].lower()
    if first_token not in _ALFWORLD_ACTION_VERBS:
        raise ValueError(
            f"generated ALFWorld action starts with unsupported command verb: {first_token}"
        )
    return normalized_action


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
