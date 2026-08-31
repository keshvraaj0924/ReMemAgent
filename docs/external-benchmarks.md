# External benchmark execution

ReMemAgent keeps ALFWorld, WebShop, model SDKs, checkpoints, and datasets outside the core package. External experiments provide factories that satisfy the normalized contracts:

- `environment_factory(seed) -> EnvironmentAdapter`
- `policy_factory(seed, memory_store) -> Policy`
- `success_evaluator(episode) -> bool`
- optional `transfer_success_evaluator(memory, episode, step) -> bool`

`experiments.external_benchmark` resolves these callables from explicit `module:attribute` specifications and executes them through `BenchmarkSuiteRunner`.

Example specification:

```text
benchmark_name=alfworld-test
environment_factory=my_alfworld:build_environment
policy_factory=my_policy:build_policy
success_evaluator=my_alfworld:evaluate_success
seed=123
```

The runner passes `seed + episode_index` to both factories. This is deterministic seed plumbing, not a claim that the third-party environment or model is internally deterministic.

Reports contain measured episode-level outcomes and aggregate reward/success/transfer statistics. Arbitrary environment `info` payloads are intentionally excluded from the JSON artifact because external objects do not have a framework-wide serialization contract.

## Current boundary

This is an executable integration boundary, not a bundled benchmark implementation. A real ALFWorld/WebShop experiment still has to supply its own environment construction, model policy, tokenizer/checkpoint configuration, and benchmark-specific success logic. The repository does not claim benchmark results until those dependencies are installed and an experiment is actually executed.
