# Repeated benchmark seed isolation

Repeated external benchmark runs are intended to represent independent run-level random seeds. The benchmark runner derives each episode seed from the configured run seed as:

```text
episode_seed = run_seed + episode_index
```

That contract is deterministic, but distinct run seeds do not automatically imply distinct episode seeds. For example, with three episodes, run seed `0` consumes episode seeds `[0, 1, 2]` while run seed `1` consumes `[1, 2, 3]`.

ReMemAgent therefore validates the complete episode-seed ranges before repeated execution or repeated runtime preflight. If any two run-level seed ranges overlap, the request fails before caller-owned benchmark environments or model policies are constructed.

For an experiment with `N` episodes per run, choose run seeds separated by at least `N`. For example, a three-episode experiment may use `(0, 3, 6, 9)` or a more widely spaced deterministic sequence such as `(0, 100, 200, 300)`.

This validation prevents accidental RNG-seed reuse. It does **not** prove statistical independence of benchmark episodes or substitute for an experiment-design justification. Benchmark-specific environment behavior, dataset ordering, model sampling configuration, and other stochastic sources must still be recorded and controlled.

Zero-episode engineering runs consume no episode seeds, so only uniqueness of their declared run seeds is required.
