# Concrete verl runtime bridge

ReMemAgent keeps the core `remem.integrations.verl` module dependency-free. When an experiment actually runs inside an environment with [verl](https://github.com/verl-project/verl) installed, `remem.integrations.verl_runtime.build_verl_agent_loop_output` provides the final runtime conversion into verl's concrete `AgentLoopOutput` model.

## Contract

The bridge maps:

- `prompt_ids` → `AgentLoopOutput.prompt_ids`
- `response_ids` → `AgentLoopOutput.response_ids`
- `response_mask` → `AgentLoopOutput.response_mask`
- trajectory reward → `reward_score`
- optional response log probabilities → `response_logprobs`
- ReMemAgent trajectory metadata → `extra_fields["remem_metadata"]`

The bridge does not tokenize text, run a model, manage an environment, or own distributed execution. Those responsibilities remain with the caller and the installed verl runtime.

The import is lazy and targets the current upstream `verl.experimental.agent_loop.agent_loop.AgentLoopOutput` location. Tests inject a compatible factory so CI does not need the heavy verl dependency.

A genuinely missing `verl` installation is converted into a focused `RuntimeError`. If `verl` is installed but one of its transitive dependencies is broken or incompatible, the original import exception is preserved. This distinction keeps environment diagnostics actionable.

This adapter is an execution boundary, not evidence of successful RL training. Full GRPO/verl claims still require a real checkpoint, installed training stack, controlled configuration, and reproducible runs.
