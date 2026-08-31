# Environment and Training Adapters

ReMemAgent keeps benchmark and training frameworks outside the research core. The adapters normalize external environment results and training trajectories into small, deterministic contracts that can be tested without installing ALFWorld, WebShop, verl, PyTorch, or a tokenizer.

## Environment execution

`EnvironmentAdapter` exposes `reset()` and `step()` and normalizes legacy four-field and Gymnasium five-field APIs into `StepResult`. The ALFWorld and WebShop adapters wrap caller-supplied environments rather than importing benchmark packages.

`BenchmarkSuite` executes matched cases and records measured outcomes. It does not synthesize benchmark scores.

## GRPO

The GRPO integration creates framework-neutral samples and deterministic group-relative advantages. Reward calculation and model optimization remain external concerns.

## verl trajectory boundary

A caller with a real verl `AgentLoopBase` can pass its token-level output through `validate_agent_loop_output()`. The validator requires `prompt_ids`, `response_ids`, and `response_mask`, rejects invalid IDs and masks, and does not decode or re-encode tokens.

For callers that also need a ReMemAgent research record, `adapt_agent_loop_output()` attaches the externally measured reward and provenance metadata to the validated token sequence:

```python
from remem.integrations import adapt_agent_loop_output

trajectory = adapt_agent_loop_output(
    agent_loop_output.model_dump(),
    reward=episode.total_reward,
    metadata={
        "memory_ids": memory_ids,
        "episode_id": episode_id,
    },
)
```

The adapter rejects non-finite rewards and copies caller-owned metadata so later top-level mutation cannot alter the stored trajectory record. Token IDs and masks are immutable tuples after validation.

### Async AgentLoop bridge

verl's `AgentLoopBase.run` is an async boundary that receives sampling parameters plus dataset-specific keyword fields and returns one token-level `AgentLoopOutput`. ReMemAgent's `AsyncVerlAgentLoop` protocol and `run_agent_loop()` mirror that boundary without importing verl.

```python
from remem.integrations import run_agent_loop

trajectory = await run_agent_loop(
    agent_loop.run,
    sampling_params={"temperature": 0.7},
    raw_prompt=messages,
    reward=episode.total_reward,
    metadata={"memory_ids": memory_ids, "episode_id": episode_id},
)
```

The external `AgentLoopBase` implementation remains responsible for chat templating, tokenization, model generation, tool/environment interaction, and constructing the token-level output. ReMemAgent does not invent prompt IDs, re-tokenize messages, own the inference server, or perform padding. It validates the returned trajectory only after the external coroutine completes.

## Training handoff

`VerlTrainingBatch` is the boundary between ReMemAgent's deterministic trajectory representation and external trainer collation. `build_verl_training_batch()` pairs already-encoded trajectories with their precomputed GRPO advantages while preserving order and rejecting alignment errors. It deliberately does not pad, truncate, tensorize, move to devices, or shard data because those operations depend on the selected training stack.

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

This is an integration boundary, not a vendored GRPO/verl implementation. The external stack still owns model generation, environment interaction, padding/collation, reward computation, advantage application, optimization, and distributed execution. ReMemAgent validates the trajectory contract and keeps memory provenance attached for research analysis.

## Current limitation

These adapters and integration records provide execution and data-conversion boundaries, not benchmark or training results. No ALFWorld, WebShop, GRPO, or verl performance claim is made until real benchmark environments and training runs are installed and matched experiments are executed.
