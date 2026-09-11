# ALFWorld deterministic seeding

The upstream ALFWorld text environment exposes legacy code paths that can consume Python or NumPy module-level random state rather than an episode-local RNG object. ReMemAgent's concrete ALFWorld factory therefore wraps each initialized environment with an episode-owned RNG stream.

For every reset, the wrapper:

1. acquires the process-wide legacy RNG lock shared by the ALFWorld and WebShop bridges;
2. restarts the episode-owned Python and NumPy RNG streams from the benchmark episode seed;
3. temporarily swaps those states into the module-level RNGs;
4. calls the upstream `reset()`;
5. records the advanced episode states;
6. restores the caller's exact module-level RNG states in `finally`.

Every subsequent `step()` uses the same isolation boundary without reseeding. The episode-owned streams therefore advance naturally from reset through the action trajectory instead of replaying the same random draw on each step. This prevents stochastic step-time behavior from depending on unrelated random-number consumption elsewhere in the host process while avoiding permanent mutation of caller RNG state.

The lock is deliberately shared with WebShop because both integrations swap the same process-global Python and NumPy RNG objects. Separate per-benchmark locks would still allow an ALFWorld operation and a WebShop operation to overlap and corrupt one another's temporary RNG state. Episode-state initialization and restart use the same re-entrant lock, so wrapper construction cannot race with reset or step either.

## Seed domain

The concrete bridge accepts integer seeds from `0` through `4294967295` inclusive. This is the domain supported by the legacy NumPy `RandomState` API used by the isolation layer. Boolean values, negative integers, integers above that range, and non-integer values are rejected at the ReMemAgent boundary before environment construction begins. Keeping this validation independent of whether NumPy happens to be installed prevents the same experiment configuration from failing differently across runtime environments.

## Evidence boundary

This improves deterministic isolation for upstream code that uses Python's or NumPy's legacy module-level RNG APIs during reset and step. It does **not** establish complete ALFWorld determinism. TextWorld behavior, filesystem ordering, package versions, dataset contents, native-library behavior, and caller-owned model policies remain external reproducibility inputs and must be pinned or recorded for scientific runs.
