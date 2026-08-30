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

```python
from remem.integrations import compute_group_relative_advantages

advantages = compute_group_relative_advantages(samples)
```

This is intentionally a framework-neutral boundary rather than a vendored GRPO or verl dependency. Group identifiers are explicit so multiple completions for the same task can participate in group-relative objectives, while memory metadata remains available for transfer analysis.

## Current limitation

These adapters and integration records provide execution and data-conversion boundaries, not benchmark or training results. No ALFWorld, WebShop, GRPO, or verl performance claim is made until real benchmark environments and training runs are installed and matched experiments are executed.
