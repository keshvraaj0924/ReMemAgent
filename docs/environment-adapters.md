# Environment Adapters

The environment adapter layer converts external agent environments into the small, deterministic interfaces used by the ReMemAgent execution runner. The adapters intentionally avoid benchmark-specific logic in the memory engine.

## ALFWorld / WebShop boundary

ALFWorld and WebShop observations are converted to the text representation expected by the execution layer. The adapters do not install, configure, or import ALFWorld or WebShop. That setup belongs to benchmark-specific experiment code. This keeps the deterministic research core dependency-free and makes unit tests runnable without external benchmark installations.

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
