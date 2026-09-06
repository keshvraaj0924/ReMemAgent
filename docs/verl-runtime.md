# Concrete verl runtime bridge

ReMemAgent keeps the core `remem.integrations.verl` module dependency-free. When an experiment actually runs inside an environment with [verl](https://github.com/verl-project/verl) installed, `remem.integrations.verl_runtime.build_verl_agent_loop_output` provides the final runtime conversion into verl's concrete `AgentLoopOutput` model.

## AgentLoopBase registration

`remem.integrations.verl_agent_loop.build_verl_agent_loop_class` is the concrete framework-registration boundary for verl's experimental `AgentLoopBase` API. It lazily imports `AgentLoopBase` and `AgentLoopOutput`, then returns a small subclass whose `run()` delegates model/environment execution to an injected async runner.

The injected runner receives:

- the `sampling_params` mapping passed by verl;
- the dataset-specific `**kwargs` passed to `AgentLoopBase.run`.

It returns the dependency-free token contract validated by `validate_agent_loop_output`. The adapter then constructs the installed verl `AgentLoopOutput`. ReMemAgent does not own model inference, tokenizer loading, environment lifecycle, batching, reward computation, optimization, or distributed execution.

This design follows verl's documented Agent Loop boundary: `AgentLoopBase.run` is the user-defined multi-turn execution point and returns one `AgentLoopOutput` containing prompt tokens, response tokens, and a response mask. The API remains experimental, so the adapter deliberately isolates the import and keeps the runner injectable. citeturn0search0

## Token/output contract

The runtime bridge maps:

- `prompt_ids` → `AgentLoopOutput.prompt_ids`
- `response_ids` → `AgentLoopOutput.response_ids`
- `response_mask` → `AgentLoopOutput.response_mask`
- trajectory reward → `reward_score` when converting a completed ReMemAgent trajectory
- optional response log probabilities → `response_logprobs`
- ReMemAgent trajectory metadata → `extra_fields["remem_metadata"]`

`VerlTrajectory` detaches the top-level metadata mapping when it is constructed. This prevents later replacement of keys in a caller-owned metadata dictionary from changing an already-created trajectory or its serialized training artifact. Nested metadata values remain caller-owned objects and should be treated as immutable after construction.

The bridge does not tokenize text, run a model, manage an environment, or own distributed execution. Those responsibilities remain with the caller and the installed verl runtime.

The import is lazy and targets the current upstream `verl.experimental.agent_loop.agent_loop` location. Tests inject compatible factories/base classes so CI does not need the heavy verl dependency.

A genuinely missing `verl` installation is converted into a focused `RuntimeError`. If `verl` is installed but one of its transitive dependencies is broken or incompatible, the original import exception is preserved. This distinction keeps environment diagnostics actionable.

The concrete AgentLoopBase adapter is an execution boundary, not evidence of successful RL training. Full GRPO/verl claims still require a real checkpoint, installed training stack, controlled configuration, and reproducible runs.
