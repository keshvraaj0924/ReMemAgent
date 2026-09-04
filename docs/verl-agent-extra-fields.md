# verl agent-loop extra fields

ReMemAgent preserves the optional `extra_fields` emitted by verl `AgentLoopOutput` when an external agent loop crosses the adapter boundary.

## Contract

The adapter validates the required token fields (`prompt_ids`, `response_ids`, and `response_mask`) and copies `extra_fields` under the reserved trajectory metadata key `verl_extra_fields`.

This is intentionally metadata preservation, not a promise that every field is trainer-compatible. The external trainer remains responsible for interpreting fields such as turn scores, tool rewards, log-probability auxiliaries, or other framework-specific values.

## Collision behavior

`verl_extra_fields` is reserved by the adapter. If caller-owned research metadata already uses that key, adaptation fails instead of silently overwriting provenance.

## Scientific boundary

Preserving dynamic fields improves trajectory fidelity but does not establish training correctness, model quality, or benchmark performance. Real verl execution still requires an externally installed verl stack, model checkpoint, tokenizer/processor, rollout infrastructure, and trainer configuration.
