# ALFWorld deterministic seeding

The upstream ALFWorld text environment currently exposes `reset()` without a seed argument and selects evaluation/training tasks through Python's module-level `random` state. ReMemAgent's concrete ALFWorld factory therefore wraps each initialized environment with a small seeded boundary.

For every reset, the wrapper:

1. acquires a process-local lock;
2. saves Python's global RNG state;
3. seeds it with the benchmark episode seed;
4. calls the upstream `reset()`;
5. restores the previous RNG state in `finally`.

This gives each ReMemAgent episode a deterministic reset seed without permanently perturbing unrelated application code. The lock is intentional: without it, concurrent benchmark workers could interleave global RNG state changes and invalidate the seed contract.

This does **not** claim that the entire ALFWorld runtime is globally deterministic. TextWorld, filesystem ordering, package versions, dataset contents, and caller-owned policies remain external reproducibility inputs. The seed boundary only covers the upstream reset-time Python RNG used for task selection.
