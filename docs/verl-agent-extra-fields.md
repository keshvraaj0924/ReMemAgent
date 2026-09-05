# verl agent-loop extra fields

ReMemAgent preserves the optional trajectory metadata emitted by verl `AgentLoopOutput` when an external agent loop crosses the adapter boundary.

## Contract

The adapter validates the required token fields (`prompt_ids`, `response_ids`, and `response_mask`) and copies `extra_fields` under the reserved trajectory metadata key `verl_extra_fields`.

Optional `response_logprobs` are also validated and preserved one-for-one with `response_ids`. This keeps rollout likelihoods available to downstream consumers that need them for replay, diagnostics, or trainer-specific loss inputs.

The adapter intentionally does not manufacture log probabilities. If the external agent loop does not provide them, the trajectory leaves `response_logprobs` unset.

## Collision behavior

`verl_extra_fields` is reserved by the adapter. If caller-owned research metadata already uses that key, adaptation fails instead of silently overwriting provenance.

## Token alignment

`response_logprobs`, when present, must have exactly one finite value per response token. The response mask remains aligned to the same token sequence, including zero-masked tool or interaction tokens emitted by an external multi-turn loop.

## Scientific boundary

Preserving dynamic fields and rollout log probabilities improves trajectory fidelity but does not establish training correctness, model quality, or benchmark performance. Real verl execution still requires an externally installed verl stack, model checkpoint, tokenizer/processor, rollout infrastructure, and trainer configuration.
