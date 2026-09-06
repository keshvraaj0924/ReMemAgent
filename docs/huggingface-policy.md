# Hugging Face policy integration

ReMemAgent now includes an optional Hugging Face text-generation policy boundary in `remem.integrations.huggingface`.

## Purpose

The adapter makes the learned-component boundary executable without putting model loading into the memory engine. It:

1. lazily loads a Hugging Face `text-generation` pipeline;
2. reuses that loaded pipeline across benchmark episodes;
3. converts normalized observations into prompts through a caller-owned `prompt_builder`;
4. converts generated text into benchmark actions through a caller-owned `action_parser`;
5. composes with `build_memory_guided_policy_factory` and the existing external benchmark runner.

The core package remains dependency-free. Install the optional `huggingface` extra only when a Transformers-backed policy is required.

## Determinism

Greedy generation (`do_sample=False`) is the default. The policy factory still accepts the benchmark episode seed because it implements the repository's `ActionPolicyFactory` contract, but the adapter does not mutate global RNG state. If sampling is enabled, callers are responsible for the model/runtime RNG controls needed for their experiment and must record those controls in the benchmark provenance.

## Example boundary

```python
from remem.integrations import build_huggingface_text_action_policy_factory

policy_factory = build_huggingface_text_action_policy_factory(
    "your-org/your-checkpoint",
    prompt_builder=lambda observation: (
        "You are an embodied agent. Return exactly one executable action.\n"
        f"Observation:\n{observation}\nAction:"
    ),
    action_parser=lambda text: text.strip().splitlines()[-1],
)
```

Pass the resulting factory as `action_policy_factory` to the existing external benchmark specification. The benchmark-specific prompt and action grammar remain caller-owned; ReMemAgent does not claim that a generic text-generation model is a valid ALFWorld or WebShop policy without a benchmark-appropriate prompt, checkpoint, and parser.

The integration is covered by dependency-free tests using an injected pipeline loader. Those tests validate lazy model construction, pipeline reuse, prompt/action composition, custom parsing, malformed output rejection, and chat-style generated output. They do not constitute benchmark results.
