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

The stronger `--runtime-preflight` path constructs the real environment and probes its reset contract. When a probe action is supplied, it also validates one real normalized step and constructs the configured policy against the observed reset text. The probe is closed before the command returns and is never included in benchmark measurements.

For repeated experiments, `--repeated-runtime-preflight --seeds 11,22,33` runs that runtime probe independently for every requested seed. `--preflight-before-run` provides the corresponding fail-fast measured mode: every requested seed is runtime-preflighted before the first measured run starts. A failed probe prevents the measured repeated run from starting.

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
  --output artifacts/alfworld-test.json \
  --manifest artifacts/alfworld-test.json.manifest.json
```

Benchmark artifacts are protected against accidental replacement. If the report path already exists, the CLI fails before measured execution unless `--overwrite` is supplied. The same protection applies to an explicitly requested integrity-manifest path. This prevents a rerun from silently destroying an earlier experiment artifact; use a new output path for independent runs or opt into replacement deliberately.

The CLI does not install benchmark packages or create model checkpoints. Those dependencies remain owned by the experiment environment.

## Factory lifecycle and reproducibility guarantees

The concrete ALFWorld factory validates the configured environment type and copies the supplied configuration before resolving the optional ALFWorld dependency. The copied configuration is passed to every environment construction, so later mutation of the caller's nested mapping cannot silently change an already-created benchmark factory. The `train_eval` selector is normalized once at factory creation.

The concrete WebShop factory normalizes the observation mode and Gym environment identifier once at factory creation. Both concrete factories scope Python's module-level RNG to the requested episode seed during environment construction and reset, then restore the caller's previous RNG state. The seeded wrappers reject an explicit `seed` argument on reset because ReMemAgent owns that seed contract.

The WebShop bridge also fails fast on Gym 0.24.x. That release is known to invoke `reset`/`step` during `gym.make`, which can violate WebShop's expected environment lifecycle. The factory rejects it before environment construction and reports an actionable compatibility error rather than allowing a misleading runtime failure. Other Gym versions are not declared universally compatible; the actual WebShop installation remains responsible for selecting a supported dependency set.

When an environment cannot satisfy the required adapter contract, `BenchmarkEnvironmentFactory` closes the caller-created environment when a `close()` method is available before propagating the validation error. This prevents partially constructed external environments from leaking resources during preflight or measured setup.

The core integration intentionally does not infer or mutate third-party benchmark state beyond this explicit seed boundary. Deterministic seed plumbing therefore remains distinct from a claim that ALFWorld, WebShop, Gym, or a model implementation is itself deterministic.

## Current boundary

This is an executable integration boundary, not a bundled benchmark implementation. A real ALFWorld/WebShop experiment still has to supply its own environment construction, model policy, tokenizer/checkpoint configuration, and benchmark-specific success logic. The repository does not claim benchmark results until those dependencies are installed and an experiment is actually executed.
