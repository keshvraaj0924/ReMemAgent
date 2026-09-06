# Concrete verl runtime bridge

ReMemAgent keeps the core `remem.integrations.verl` module dependency-free. When an experiment actually runs inside an environment with [verl](https://github.com/verl-project/verl) installed, `remem.integrations.verl_agent_loop.build_verl_agent_loop_class` provides the final runtime registration boundary for verl's experimental `AgentLoopBase` API.

## AgentLoopBase registration

`build_verl_agent_loop_class` lazily imports `AgentLoopBase` and `AgentLoopOutput`, then returns a small subclass whose `run()` delegates model/environment execution to an injected async runner.

The injected runner receives:

- the `sampling_params` mapping passed by verl;
- the dataset-specific `**kwargs` passed to `AgentLoopBase.run`.

It returns the dependency-free token contract validated by `validate_agent_loop_output`. The adapter then constructs the installed verl `AgentLoopOutput`. ReMemAgent does not own model inference, tokenizer loading, environment lifecycle, batching, reward computation, optimization, or distributed execution.

The optional `output_factory` argument allows a compatible downstream verl fork to supply its concrete output model while keeping the base-class import and execution contract unchanged.

This design follows verl's documented Agent Loop boundary: `AgentLoopBase.run` is the user-defined multi-turn execution point and returns one `AgentLoopOutput` containing prompt tokens, response tokens, and a response mask. The API remains experimental, so the adapter deliberately isolates the import and keeps the runner injectable. citeturn0search0

## Token/output contract

The runtime bridge maps the validated fields directly:

- `prompt_ids` → `AgentLoopOutput.prompt_ids`
- `response_ids` → `AgentLoopOutput.response_ids`
- `response_mask` → `AgentLoopOutput.response_mask`
- optional `response_logprobs` → `AgentLoopOutput.response_logprobs`
- `extra_fields` → `AgentLoopOutput.extra_fields`

No re-tokenization or model-dependent transformation occurs at this boundary. This preserves token-level trajectory fidelity and keeps memory provenance available for research analysis.

`VerlTrajectory` separately provides a typed representation for episodes and GRPO batches. Its top-level metadata mapping is detached when constructed, preventing later replacement of keys in a caller-owned metadata dictionary from changing an already-created trajectory. Nested metadata values remain caller-owned objects and should be treated as immutable after construction.

## Failure semantics

The `verl` import is lazy. A genuinely missing `verl` installation is converted into a focused `RuntimeError`. If `verl` is installed but one of its transitive dependencies is broken or incompatible, the original import exception is preserved. This distinction keeps environment diagnostics actionable.

Malformed runner output fails before it reaches `AgentLoopOutput`: token IDs, response masks, optional log probabilities, and dynamic `extra_fields` are validated by `validate_agent_loop_output`.

## Evidence boundary

The concrete AgentLoopBase adapter is an execution boundary, not evidence of successful RL training. Full GRPO/verl claims still require a real checkpoint, installed training stack, controlled configuration, and reproducible runs. The repository does not report benchmark improvements merely because this adapter can be instantiated in dependency-free tests.
