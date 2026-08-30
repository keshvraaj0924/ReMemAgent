    encode_completion=tokenizer.encode_completion,
    memory_ids=memory_ids,
)
agent_loop_output = trajectory.to_agent_loop_output()
```

`VerlTrainingBatch` is the next boundary between ReMemAgent's deterministic trajectory representation and external trainer collation. `build_verl_training_batch()` pairs already-encoded trajectories with their precomputed GRPO advantages while preserving order and rejecting alignment errors. It deliberately does not pad, truncate, tensorize, move to devices, or shard data because those operations depend on the selected training stack.

```python
from remem.integrations import build_verl_training_batch

verl_batch = build_verl_training_batch(trajectories, batch.advantages)
rows = verl_batch.to_dicts()
```

`dispatch_verl_training_batch()` is the framework-facing handoff. It passes fresh serialized rows to an injected external consumer, leaving framework-specific collation, tensors, devices, optimization, and distributed execution outside ReMemAgent. The source `VerlTrainingBatch` remains isolated from consumer-side dictionary mutation.

```python
from remem.integrations import dispatch_verl_training_batch

result = dispatch_verl_training_batch(verl_batch, external_trainer.consume)
```

For a real custom verl `AgentLoopBase`, the rollout already owns exact token generation. `validate_agent_loop_output()` validates the resulting `prompt_ids`, `response_ids`, and `response_mask` without decoding or re-encoding them. This is important for multi-turn/tool trajectories because the response mask must distinguish model-generated tokens from environment/tool-response tokens. The validator rejects missing fields, negative/non-integer token IDs, mask length mismatches, and non-binary masks while returning immutable token tuples for downstream research processing.

```python
from remem.integrations import validate_agent_loop_output

validated = validate_agent_loop_output(agent_loop_output.model_dump())
```

The resulting records retain reward and memory metadata for offline experiment analysis, while `to_agent_loop_output()` emits only the framework-facing token fields.

This remains a clean integration boundary rather than a vendored GRPO/verl implementation. The external trainer owns model generation, padding/collation, reward computation, advantage application, optimization, and distributed execution. ReMemAgent validates the trajectory contract and keeps memory provenance attached for research analysis.

## Current limitation

These adapters and integration records provide execution and data-conversion boundaries, not benchmark or training results. No ALFWorld, WebShop, GRPO, or verl performance claim is made until real benchmark environments and training runs are installed and matched experiments are executed.
