# WebShop deterministic seeding

The upstream WebShop text environment uses legacy module-level random state during environment setup and task/session behavior. Its `reset()` API does not provide a modern Gym-style episode RNG contract, so ReMemAgent wraps the upstream environment with an episode-owned Python/NumPy RNG stream.

For each benchmark episode, the wrapper:

1. acquires the process-wide legacy RNG lock shared by the WebShop and ALFWorld bridges;
2. restarts an episode-owned RNG stream from the requested seed before `reset()`;
3. temporarily swaps that stream into Python's and NumPy's module-level RNGs;
4. calls the upstream operation;
5. records the advanced episode RNG state;
6. restores the caller's exact global RNG states in `finally`.

Subsequent `step()` calls continue from the RNG state produced by the preceding reset or step. They are isolated in the same way but are **not** reseeded on every action, so stochastic trajectories preserve normal RNG progression while remaining independent from unrelated host-process random-number consumption.

The factory also isolates environment-construction side effects from the caller's RNG state. This matters because upstream WebShop initialization can perform eager setup/reset behavior that mutates module-level randomness.

The lock is intentionally shared across WebShop and ALFWorld. Both bridges temporarily replace the same process-global Python and NumPy RNG states, so independent per-benchmark locks would permit cross-benchmark interleaving even though each benchmark appeared locally serialized. Episode-state initialization and restart use the same re-entrant lock as construction, reset, and step. Cross-process workers remain independently isolated by their own interpreter state.

## Seed domain

The concrete bridge accepts integer seeds from `0` through `4294967295` inclusive. That range matches the legacy NumPy `RandomState` API used by the isolation layer. Boolean values, negative integers, integers above the upper bound, and non-integer values are rejected before `gym.make()` runs. The validation is unconditional rather than dependent on NumPy import success, so benchmark configuration validity remains stable across machines.

## Evidence boundary

This is a reproducibility boundary, not a claim of complete WebShop determinism. It controls Python and NumPy legacy module-level RNG usage around construction, reset, and step, but the upstream environment also depends on installed package versions, local product/instruction data, search-index contents, native libraries, and the caller-owned policy. Those inputs must still be pinned or recorded for scientific runs.

The original WebShop environment exposes `WebAgentTextEnv-v0` through Gym and does not provide a reliable `reset(seed=...)` contract for this integration, so ReMemAgent does not fabricate one. Instead, the adapter preserves a deterministic episode RNG stream around the actual upstream API.

## Operational consequence

A repeated benchmark can create one environment per seed and execute the complete measured trajectory without allowing unrelated module-level RNG consumption in the same process to perturb that episode's Python/NumPy random stream. Stronger determinism still requires controlling the remaining external inputs rather than treating the seed boundary as sufficient scientific evidence by itself.
