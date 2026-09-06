# Learned Benchmark Policies

ReMemAgent now has benchmark-specific adapters for connecting a real text-generation model to the existing ALFWorld and WebShop environment boundaries.

## Separation of responsibilities

- `remem/integrations/huggingface.py` owns optional Transformers model loading and generation.
- `remem/integrations/benchmark_policies.py` owns benchmark prompt construction and action parsing.
- `remem/integrations/policies.py` owns memory-guided composition.
- `remem/environments/alfworld.py` and `remem/environments/webshop.py` own environment normalization.

This keeps learned inference separate from research heuristics and environment mechanics.

## ALFWorld

`build_alfworld_huggingface_policy_factory()` constructs prompts that request one executable text action and parses an `Action:` response into a scalar action suitable for `AlfWorldAdapter`.

The parser accepts either one explicit `Action:` line or one unambiguous non-empty generated line. Multi-line output without an explicit action boundary, and output containing multiple `Action:` lines, is rejected rather than guessing which text should reach the environment.

The adapter deliberately does not invent an ALFWorld action vocabulary. The upstream environment remains authoritative for whether an emitted action is executable.

## WebShop

`build_webshop_huggingface_policy_factory()` requests the canonical WebShop `search[...]` / `click[...]` action form and extracts one such action from the model output.

The parser is intentionally conservative: free-form reasoning without a canonical action, multiple canonical actions, or an explicit `Action:` line containing more than one canonical action is rejected instead of silently selecting one action. This prevents an explanation or ambiguous completion from being treated as an executable environment command.

## Example composition

```python
from remem.integrations import (
    build_alfworld_huggingface_policy_factory,
    build_memory_guided_policy_factory,
)

model_policy_factory = build_alfworld_huggingface_policy_factory(
    "your-model-checkpoint",
)
memory_policy_factory = build_memory_guided_policy_factory(model_policy_factory)
```

The checkpoint, tokenizer behavior, and model quality remain experiment-specific. A generic language model is not itself evidence of ALFWorld or WebShop benchmark performance.

## Reproducibility boundary

The Hugging Face adapter defaults to greedy generation (`do_sample=False`). If sampling is enabled, the experiment must explicitly own the model's RNG and generation configuration. Benchmark execution must still use the repository's seed-aware environment factories and persisted artifacts.

## Current limitation

These adapters establish a real model-to-environment action path, but no benchmark score is claimed until an actual ALFWorld/WebShop installation, concrete checkpoint, multi-seed run, and persisted evidence artifact have all been executed successfully.
