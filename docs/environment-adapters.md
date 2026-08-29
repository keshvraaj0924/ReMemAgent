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

## Compatibility contract

The adapters accept both common environment return conventions:

- `reset()` returning an observation or `(observation, info)`
- `step()` returning `(observation, reward, done, info)` or `(observation, reward, terminated, truncated, info)`

Observations are converted to text because the current ReMemAgent memory domain is text-oriented. Rewards are converted to `float`, and `info` is copied into a regular dictionary.

The adapters do not install, configure, or import ALFWorld or WebShop. That setup belongs to benchmark-specific experiment code. This keeps the deterministic research core dependency-free and makes unit tests runnable without external benchmark installations.

## Current limitation

These adapters provide the execution boundary, not benchmark results. No ALFWorld or WebShop performance claim is made until real benchmark environments are installed and matched experiments are executed.
