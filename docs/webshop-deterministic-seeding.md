# WebShop deterministic seeding

The upstream WebShop text environment uses Python's module-level `random` state when creating session identifiers and selecting tasks. Its `reset()` API does not accept a Gym-style `seed` argument. The official ReMemAgent factory therefore adds a small seed-scoped boundary around the real upstream environment.

For each benchmark episode, the wrapper:

1. acquires a process-local lock;
2. saves Python's global RNG state;
3. seeds it with the requested episode seed;
4. calls the upstream `reset()`;
5. restores the previous RNG state in `finally`.

The factory also isolates environment-construction side effects from the caller's RNG state. This matters because the upstream `WebAgentTextEnv` constructor performs an eager reset and its simulator initialization mutates Python's global random state.

This gives ReMemAgent a deterministic reset boundary without permanently perturbing unrelated application code. The lock prevents concurrent benchmark workers in the same process from interleaving global RNG state changes.

## Evidence boundary

This is a reproducibility boundary, not a claim of complete WebShop determinism. The upstream environment depends on its installed package versions, local product/instruction data, search index, and caller-owned policy. The seed contract only controls Python's RNG around environment construction and explicit episode reset.

The original WebShop project documents `WebAgentTextEnv-v0` as a Gym environment and exposes `reset(session=None, instruction_text=None)`, so ReMemAgent does not pretend that a `reset(seed=...)` API exists where the benchmark does not provide one.

## Operational consequence

A repeated benchmark can now create one environment per seed, run the measured episode after the seeded reset boundary, and retain the seed in the benchmark report. If an experiment needs stronger determinism than this boundary provides, it must record and control the remaining external inputs rather than silently treating the benchmark as fully deterministic.
