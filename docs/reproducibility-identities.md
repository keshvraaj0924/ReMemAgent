# Reproducible benchmark configuration identities

`remem.reproducibility.benchmark_configuration_digest()` computes a deterministic SHA-256 digest from a validated `BenchmarkRunConfiguration`.

The digest is built from a **versioned identity envelope**. `REPRODUCIBILITY_SCHEMA_VERSION` identifies the canonicalization contract, while the nested `configuration` object contains the benchmark name, episode/step counts, seed, trust threshold, and structured runtime provenance fields. Changing any of those inputs changes the identity.

`benchmark_configuration_payload()` exposes that canonical envelope for manifests and audit tooling without duplicating the serialization contract:

```python
from remem.benchmark import BenchmarkRunConfiguration
from remem.reproducibility import benchmark_configuration_digest

configuration = BenchmarkRunConfiguration(
    benchmark_name="alfworld",
    episode_count=100,
    max_steps=50,
    seed=7,
    environment_factory="integrations.alfworld:make_environment",
    policy_factory="integrations.alfworld:make_policy",
    success_evaluator="integrations.alfworld:evaluate_success",
)

identity = benchmark_configuration_digest(configuration)
```

The JSON encoding is canonical: keys are sorted, separators are fixed, and Unicode is encoded consistently. Versioning is deliberate: if the identity schema or canonicalization rules change in a future release, the version can distinguish the new identity contract from older artifact names instead of silently reusing the same namespace.

The digest is an **identity of configuration metadata**, not a proof that two executions are scientifically equivalent. External environment versions, model weights, hardware, dependency locks, source revision, and actual run outputs must remain part of the broader reproducibility record.
