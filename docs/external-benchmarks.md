# External benchmark execution

ReMemAgent keeps ALFWorld, WebShop, model SDKs, checkpoints, and datasets outside the core package. External experiments provide factories that satisfy the caller-owned raw environment contract and the normalized policy/evaluator contracts:

- `environment_factory(seed) -> raw benchmark environment`
- `policy_factory(seed, memory_store) -> Policy`
- `success_evaluator(episode) -> bool`
- optional `transfer_success_evaluator(memory, episode, step) -> bool`

`experiments.external_benchmark` resolves these callables from explicit `module:attribute` specifications. The raw environment factory is then bound to the benchmark-specific `AlfWorldAdapter` or `WebShopAdapter` before execution through `BenchmarkSuiteRunner`.

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

## Preflight validation

Before launching an expensive benchmark, the CLI can resolve every configured callable without constructing an environment or loading a model:

```bash
remem-benchmark \
  --benchmark alfworld-test \
  --episodes 100 \
  --max-steps 50 \
  --environment-factory my_alfworld:build_environment \
  --policy-factory my_policy:build_policy \
  --success-evaluator my_alfworld:evaluate_success \
  --seed 123 \
  --preflight
```

A successful preflight verifies import paths and callable types only. It does not verify that the external environment can execute a reset/step cycle, that a checkpoint is loadable, or that the model produces valid actions. Those checks require the real dependencies and are intentionally left to the measured run.

## Command-line execution

The `remem-benchmark` entry point accepts the same explicit callable specifications:

```bash
remem-benchmark \
  --benchmark alfworld-test \
  --episodes 100 \
  --max-steps 50 \
  --environment-factory my_alfworld:build_environment \
  --policy-factory my_policy:build_policy \
  --success-evaluator my_alfworld:evaluate_success \
  --seed 123 \
  --output artifacts/alfworld-test.json
```

The CLI does not install benchmark packages or create model checkpoints. Those dependencies remain owned by the experiment environment.

## Current boundary

This is an executable integration boundary, not a bundled benchmark implementation. A real ALFWorld/WebShop experiment still has to supply its own environment construction, model policy, tokenizer/checkpoint configuration, and benchmark-specific success logic. The repository does not claim benchmark results until those dependencies are installed and an experiment is actually executed.
