# ALFWorld deterministic seeding

The upstream ALFWorld text environment exposes legacy code paths that can consume Python or NumPy module-level random state rather than an episode-local RNG object. ReMemAgent's concrete ALFWorld factory therefore wraps each initialized environment with an episode-owned RNG stream.

For every reset, the wrapper:

1. acquires a process-local lock;
2. restarts the episode-owned Python and NumPy RNG streams from the benchmark episode seed;
3. temporarily swaps those states into the module-level RNGs;
4. calls the upstream `reset()`;
5. records the advanced episode states;
6. restores the caller's exact module-level RNG states in `finally`.

Every subsequent `step()` uses the same isolation boundary without reseeding. The episode-owned streams therefore advance naturally from reset through the action trajectory instead of replaying the same random draw on each step. This prevents stochastic step-time behavior from depending on unrelated random-number consumption elsewhere in the host process while avoiding permanent mutation of caller RNG state.

The process-local lock is intentional. ALFWorld's legacy global RNG usage cannot be safely swapped concurrently inside one process; without serialization, parallel benchmark workers could interleave global state changes and invalidate the seed contract.

## Evidence boundary

This improves deterministic isolation for upstream code that uses Python's or NumPy's legacy module-level RNG APIs during reset and step. It does **not** establish complete ALFWorld determinism. TextWorld behavior, filesystem ordering, package versions, dataset contents, native-library behavior, and caller-owned model policies remain external reproducibility inputs and must be pinned or recorded for scientific runs.
