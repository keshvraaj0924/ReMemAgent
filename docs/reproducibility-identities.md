# Reproducible benchmark configuration identities

`remem.reproducibility.benchmark_configuration_digest()` computes a deterministic SHA-256 digest from a validated `BenchmarkRunConfiguration`.

The digest uses canonical JSON: keys are sorted, separators are fixed, and Unicode is encoded consistently. Because the configuration includes the benchmark name, episode/step counts, seed, trust threshold, and structured runtime provenance fields, changing any of those inputs changes the identity.

Use the digest in experiment manifests or artifact names, for example:

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

The digest is an **identity of configuration metadata**, not a proof that two executions are scientifically equivalent. External environment versions, model weights, hardware, dependency locks, and actual run outputs must remain part of the broader reproducibility record.
