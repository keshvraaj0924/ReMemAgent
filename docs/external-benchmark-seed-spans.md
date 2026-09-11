# External benchmark seed-span validation

ReMemAgent's supported ALFWorld and WebShop execution bridges rely on legacy NumPy-compatible episode seeds. The accepted seed domain is therefore `0` through `2**32 - 1` (`4294967295`), inclusive.

A benchmark run does not use only the configured run seed. `BenchmarkSuiteRunner` derives one episode seed per episode. With an explicit seed, the contract is:

```text
episode_seed = run_seed + episode_index
```

When `seed=None`, the runner preserves its historical behavior and passes `episode_index` directly to the environment and policy factories. For external benchmarks this is equivalent to an implicit run seed of zero for seed-span validation:

```text
episode_seed = episode_index
```

For that reason, validating only an explicitly configured run seed is insufficient. A configuration such as `run_seed=4294967294` with three episodes would begin with a valid seed and later derive `4294967296`, which is outside the supported RNG domain. Likewise, an unseeded external run with more than `2**32` episodes would eventually cross the same boundary.

`ExternalBenchmarkSpec` validates the complete derived episode seed span before a single-run experiment can execute, whether the seed is explicit or implicit. Repeated experiments perform the same span validation for every independent run seed before constructing any benchmark environment. This fail-closed behavior prevents partially measured runs caused solely by a predictable seed overflow.

Repeated-run overlap validation remains a separate invariant: independent run seed spans must both fit the supported domain and remain disjoint from one another.

This validation does not claim complete benchmark determinism. Dataset/index state, upstream package versions, model runtime behavior, hardware/native kernels, and other external inputs still need to be pinned and recorded for controlled ALFWorld/WebShop experiments.
