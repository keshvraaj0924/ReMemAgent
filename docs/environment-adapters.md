# Benchmark environment adapters

ReMemAgent keeps benchmark-specific dependencies outside the memory and routing core. The adapters in `remem.environments` normalize an already-created external environment into the small `EnvironmentAdapter` contract.

## ALFWorld

Create an ALFWorld environment using the benchmark's own setup, then wrap it:

```python
from remem.environments import AlfWorldAdapter

adapter = AlfWorldAdapter(alfworld_environment)
observation = adapter.reset()
result = adapter.step("look")
```

## WebShop

The same contract is available for WebShop:

```python
from remem.environments import WebShopAdapter

adapter = WebShopAdapter(webshop_environment)
observation = adapter.reset()
result = adapter.step("search[shoes]")
```

## Shared benchmark runner

`BenchmarkSuiteRunner` provides the experiment-facing orchestration boundary for either adapter. It creates one environment per episode, shares a `MemoryStore` across episodes, evaluates outcomes through an injected success function, ingests trajectories, and closes environments after execution.

```python
from remem import BenchmarkSuiteRunner

report = BenchmarkSuiteRunner().run(
    benchmark_name="alfworld",
    episode_count=10,
    max_steps=30,
    environment_factory=lambda index: AlfWorldAdapter(make_alfworld_environment(index)),
    policy_factory=lambda index, store: make_policy(store),
    success_evaluator=lambda episode: episode.steps[-1].result.info.get("won", False),
)
```

The runner does not decide what constitutes success and does not construct benchmark environments itself. Those choices remain experiment-specific. A shared store makes cross-episode memory transfer testable, while the runner remains usable for a memory-free baseline by supplying a policy that ignores the store.

## Compatibility contract

The adapters accept both common environment return conventions:

- `reset()` returning an observation or `(observation, info)`
- `step()` returning `(observation, reward, done, info)` or `(observation, reward, terminated, truncated, info)`

Observations are converted to text because the current ReMemAgent memory domain is text-oriented. Rewards are converted to `float`, and `info` is copied into a regular dictionary.

The adapters do not install, configure, or import ALFWorld or WebShop. That setup belongs to benchmark-specific experiment code. This keeps the deterministic research core dependency-free and makes unit tests runnable without external benchmark installations.

## GRPO / agent-training boundary

The `remem.integrations.grpo` module provides a dependency-free conversion from completed `EpisodeResult` trajectories to `GrpoSample` records. Each record contains the initial observation as the prompt, the executed action sequence as the completion, the episode reward, a caller-controlled group identifier, and the memory identifiers that influenced the trajectory.

```python
from remem.integrations import build_grpo_samples

samples = build_grpo_samples(
    episodes,
    decision_histories=decision_histories,
    group_id_builder=lambda index, episode: task_ids[index],
)
training_rows = [sample.to_dict() for sample in samples]
```

The integration also exposes `compute_group_relative_advantages()`. It computes the GRPO-style centered and population-standard-deviation-normalized reward for each sample, preserving input order. Constant-reward groups receive zero advantages. This is a deterministic mathematical boundary; it does not implement a policy-gradient loss, optimizer, rollout engine, or framework-specific trainer.

`build_grpo_batch()` packages the samples and their computed advantages into an immutable `GrpoBatch`. The batch validates non-empty input and sample/advantage alignment and exposes `to_dicts()` for framework-specific dataset writers. This keeps ordering and reward normalization explicit before a trainer-specific collation step.

```python
from remem.integrations import build_grpo_batch

batch = build_grpo_batch(samples)
training_rows = batch.to_dicts()
```

### verl agent-loop boundary

`remem.integrations.verl` adds a dependency-free encoder for the token-level contract expected from a verl `AgentLoopBase`: `prompt_ids`, `response_ids`, and `response_mask`. ReMemAgent does not import verl; the tokenizer is injected by the caller so model and tokenizer choices remain outside the research core.

```python
from remem.integrations import encode_episode_for_verl

trajectory = encode_episode_for_verl(
    episode,
    encode_prompt=tokenizer.encode_prompt,
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

`dispatch_verl_training_batch()` is the framework-facing handoff. It passes immutable, ordered serialized rows to an injected external consumer, leaving framework-specific collation, tensors, devices, optimization, and distributed execution outside ReMemAgent.

```python
from remem.integrations import dispatch_verl_training_batch

result = dispatch_verl_training_batch(verl_batch, external_trainer.consume)
```

`response_mask` is currently all ones because the normalized episode contains only agent actions. A future multi-turn adapter must mark tool/environment response tokens as zero rather than reconstructing token IDs from rendered chat history. This distinction matters for RL training because tokenization and tool parsing can otherwise change the exact sampled trajectory.

The resulting records retain reward and memory metadata for offline experiment analysis, while `to_agent_loop_output()` emits only the framework-facing token fields.

This remains a clean integration boundary rather than a vendored GRPO/verl implementation. The external trainer owns model generation, padding/collation, reward computation, advantage application, optimization, and distributed execution.

## Current limitation

These adapters and integration records provide execution and data-conversion boundaries, not benchmark or training results. No ALFWorld, WebShop, GRPO, or verl performance claim is made until real benchmark environments and training runs are installed and matched experiments are executed.
