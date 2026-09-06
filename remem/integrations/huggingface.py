"""Optional Hugging Face text-generation policy integration.

This module keeps the learned model boundary explicit: ReMemAgent owns only
prompt construction, output parsing, and memory-guided composition. Model
loading and inference remain in the optional Transformers integration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from remem.integrations.policies import ActionPolicyFactory

PromptBuilder = Callable[[str], str]
ActionParser = Callable[[str], str]
PipelineLoader = Callable[..., Any]
TextGenerator = Callable[..., object]


def build_huggingface_text_action_policy_factory(
    model_name: str,
    *,
    prompt_builder: PromptBuilder,
    action_parser: ActionParser | None = None,
    pipeline_loader: PipelineLoader | None = None,
    pipeline_kwargs: Mapping[str, Any] | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> ActionPolicyFactory:
    """Build a lazy Hugging Face text-generation action-policy factory.

    ``model_name`` is loaded once, on the first policy creation, and reused by
    subsequent episodes. The caller supplies benchmark-specific prompt and
    action parsing logic so ALFWorld/WebShop semantics never leak into this
    generic learned-component adapter.

    Greedy generation is the default because it avoids hidden global RNG
    mutation. Sampling can be enabled through ``generation_kwargs`` when the
    caller owns the corresponding reproducibility controls.
    """

    _validate_non_empty_string("model_name", model_name)
    if not callable(prompt_builder):
        raise TypeError("prompt_builder must be callable")
    selected_parser = action_parser or _default_action_parser
    if not callable(selected_parser):
        raise TypeError("action_parser must be callable")

    selected_pipeline_loader = pipeline_loader or _load_huggingface_pipeline
    if not callable(selected_pipeline_loader):
        raise TypeError("pipeline_loader must be callable")

    selected_pipeline_kwargs = dict(pipeline_kwargs or {})
    selected_generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": 32,
        "return_full_text": False,
        **dict(generation_kwargs or {}),
    }
    generator: TextGenerator | None = None

    def create_action_policy(seed: int) -> Callable[[str], str]:
        """Create an action policy for one benchmark episode seed."""

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        nonlocal generator
        if generator is None:
            generator = cast(
                TextGenerator,
                selected_pipeline_loader(
                    "text-generation",
                    model=model_name,
                    **selected_pipeline_kwargs,
                ),
            )

        def select_action(observation: str) -> str:
            """Generate and parse one action from a normalized observation."""

            if not isinstance(observation, str) or not observation.strip():
                raise ValueError("observation must be a non-empty string")
            prompt = prompt_builder(observation)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt_builder must return a non-empty string")
            generated_text = _generate_text(generator, prompt, selected_generation_kwargs)
            action = selected_parser(generated_text)
            if not isinstance(action, str) or not action.strip():
                raise ValueError("action_parser must return a non-empty string")
            return action.strip()

        return select_action

    return create_action_policy


def _load_huggingface_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Load the optional Transformers pipeline without importing it at module load."""

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face integration requires the optional 'huggingface' dependencies; "
            "install rememagent[huggingface] to use this policy factory"
        ) from exc
    return pipeline(*args, **kwargs)


def _generate_text(
    generator: TextGenerator,
    prompt: str,
    generation_kwargs: Mapping[str, Any],
) -> str:
    """Generate one text completion and normalize supported pipeline outputs."""

    raw_output = generator(prompt, **generation_kwargs)
    if not isinstance(raw_output, list) or not raw_output:
        raise ValueError("text-generation pipeline must return a non-empty list")
    first_output = raw_output[0]
    if not isinstance(first_output, dict):
        raise ValueError("text-generation pipeline output must contain dictionaries")
    generated_text = first_output.get("generated_text")
    if isinstance(generated_text, str) and generated_text.strip():
        return generated_text.strip()
    if isinstance(generated_text, list):
        return _extract_chat_text(generated_text)
    raise ValueError("text-generation pipeline returned no usable generated_text")


def _extract_chat_text(messages: list[object]) -> str:
    """Extract the latest textual message from a chat-style generation result."""

    for message in reversed(messages):
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("chat-style generation returned no usable message content")


def _default_action_parser(generated_text: str) -> str:
    """Use the final non-empty generated line as the benchmark action."""

    lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("generated text contains no non-empty action line")
    return lines[-1]


def _validate_non_empty_string(field_name: str, value: object) -> None:
    """Require a non-empty string configuration field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


__all__ = [
    "ActionParser",
    "PipelineLoader",
    "PromptBuilder",
    "build_huggingface_text_action_policy_factory",
]
