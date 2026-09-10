# Real benchmark execution gate

This document defines the minimum evidence required before ReMemAgent reports an ALFWorld or WebShop result. Adapter tests and dependency-free smoke fixtures validate software contracts only; they are not benchmark evidence.

## 1. Validate the environment boundary

Resolve the exact benchmark and policy callables first:

```text
python -m experiments.benchmark_cli \
  --benchmark <alfworld|webshop> \
  --episodes <N> \
  --max-steps <N> \
  --environment-factory <module:factory> \
  --action-policy-factory <module:factory> \
  --success-evaluator <module:evaluator> \
  --preflight
```

Then construct the real upstream environment and probe the normalized adapter:

```text
python -m experiments.benchmark_cli \
  --benchmark <alfworld|webshop> \
  --episodes <N> \
  --max-steps <N> \
  --environment-factory <module:factory> \
  --action-policy-factory <module:factory> \
  --success-evaluator <module:evaluator> \
  --runtime-preflight \
  --probe-action "<valid action>"
```

For multi-seed experiments, use `--repeated-runtime-preflight --seeds ...` before measured execution. A failed seed probe must prevent the measured run from being treated as evidence. Programmatic repeated-run callers must leave `ExternalBenchmarkSpec.seed` as `None` and provide all independent seeds through the repeated-run `seeds` argument; the framework rejects a simultaneously configured single seed rather than silently overriding provenance.

## 2. Run paired conditions when evaluating transfer

Baseline and memory-guided treatment runs must use the same benchmark, episode count, step limit, environment callable, success evaluator, transfer evaluator, and trust threshold. The policy callable is the experimental condition and is therefore allowed to differ. Use the same explicit independent seed set for both conditions.

```text
python -m experiments.paired_benchmark_cli \
  --benchmark <alfworld|webshop> \
  --episodes <N> \
  --max-steps <N> \
  --seeds "17,23,41,59,83" \
  --environment-factory <module:factory> \
  --baseline-policy-factory <module:factory> \
  --treatment-action-policy-factory <module:factory> \
  --success-evaluator <module:evaluator> \
  --minimum-trust <0..1> \
  --output artifacts/<benchmark>-paired.json \
  --manifest artifacts/<benchmark>-paired.json.manifest.json
```

The paired statistics layer aligns observations by seed and computes descriptive treatment-minus-baseline deltas. It does not pool episodes and does not turn a descriptive delta into a causal or production claim. For inferential reporting, the framework also provides an exact paired sign-flip test and a paired Cohen's *d_z* effect-size calculation; effect size is reported as undefined when fewer than two independent paired observations exist or when their sample variance is zero.

## 3. Preserve exact evidence artifacts

Every measured artifact should retain:

- benchmark configuration;
- explicit seed(s);
- callable specifications;
- trust threshold;
- runtime provenance and dependency versions;
- per-run episode metrics;
- seed-level descriptive statistics for repeated runs;
- an exact-byte SHA-256 manifest.

Do not overwrite an existing artifact unless `--overwrite` is intentional. The manifest covers the serialized bytes, including the final newline, so a modified report must fail verification.

## 4. Scientific evidence boundary

A successful quality workflow establishes software correctness for the covered tests. It does not establish benchmark effectiveness. A benchmark result is publishable only after the corresponding real environment and model configuration has executed, the requested independent seeds completed, artifacts were preserved, and the result can be reproduced from the recorded revision and dependency environment.

No benchmark improvement, significance, effect-size interpretation, or production-readiness claim should be inferred from the repository's dependency-free smoke fixture, adapter tests, preflight result, or CI status alone.
