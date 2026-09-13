# ALFWorld adapter contract

The ALFWorld adapter is intentionally strict at the boundary between the upstream batch-oriented environment and ReMemAgent's scalar `StepResult` contract.

## Normalization rules

- A singleton batch dimension is removed from observations, rewards, and terminal flags.
- Singleton unwrapping accepts ordinary Python sequences and array-like objects that expose `len()` plus integer indexing; the core package does not require NumPy.
- Mapping values in `info` are unwrapped per field while the mapping itself remains metadata.
- Rewards must already be numeric and finite. Booleans and numeric strings are rejected instead of being coerced.
- `terminated` and `truncated` must be actual booleans after singleton unwrapping. Truthiness conversion is not performed.
- Four-value legacy step results are accepted as `(observation, reward, done, info)` and normalized to `terminated=done`, `truncated=False`.
- Five-value step results are accepted as `(observation, reward, terminated, truncated, info)`.
- Invalid step payloads fail before a malformed `StepResult` can reach episode metrics or training artifacts.

This fail-closed behavior is an integration invariant, not a benchmark result. It prevents ambiguous upstream values from changing episode termination or reward semantics silently.

## Official environment factory admission

`build_alfworld_text_environment_factory()` validates the object returned by upstream `init_env(batch_size=1)` before it is admitted to measured execution. The initialized environment must expose callable `reset()` and `step()` methods. An invalid initialized object is rejected before an episode begins, and factory-owned upstream objects are closed during failed admission. If `init_env()` returns the same object that was constructed, cleanup occurs once rather than relying on duplicate close calls.

The equivalent WebShop factory applies the same callable `reset()` / `step()` admission contract after `gym.make()`. This keeps malformed optional benchmark installations from failing midway through a measured run.

## Verification

The adapter regression suite covers valid five-value normalization, legacy four-value normalization, array-like singleton batches, malformed terminal flags, boolean rewards, and non-finite rewards. Official-factory tests additionally cover missing `reset()` / `step()` methods and cleanup on rejected environment admission.

Real ALFWorld execution still requires the optional upstream installation and a caller-owned environment/model configuration; this repository does not claim that external execution has been run merely because the adapter contract is tested.
