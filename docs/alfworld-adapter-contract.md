# ALFWorld adapter contract

The ALFWorld adapter is intentionally strict at the boundary between the upstream batch-oriented environment and ReMemAgent's scalar `StepResult` contract.

## Normalization rules

- A singleton batch dimension is removed from observations, rewards, and terminal flags.
- Rewards must already be numeric and finite. Booleans and numeric strings are rejected instead of being coerced.
- `terminated` and `truncated` must be actual booleans after singleton unwrapping. Truthiness conversion is not performed.
- Four-value legacy step results are accepted as `(observation, reward, done, info)` and normalized to `terminated=done`, `truncated=False`.
- Five-value step results are accepted as `(observation, reward, terminated, truncated, info)`.
- Invalid step payloads fail before a malformed `StepResult` can reach episode metrics or training artifacts.

This fail-closed behavior is an integration invariant, not a benchmark result. It prevents ambiguous upstream values from changing episode termination or reward semantics silently.

## Verification

The adapter regression suite covers valid five-value normalization, legacy four-value normalization, malformed terminal flags, boolean rewards, and non-finite rewards.

Real ALFWorld execution still requires the optional upstream installation and a caller-owned environment/model configuration; this repository does not claim that external execution has been run merely because the adapter contract is tested.
