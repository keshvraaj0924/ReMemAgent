# ReMemAgent

> **Learning when to remember, what to reconstruct, and when to forget.**

[![Research Prototype](https://img.shields.io/badge/status-research--prototype-orange)](https://github.com/keshvraaj0924/ReMemAgent)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ReMemAgent is a research framework for **adaptive, reconstructive memory in LLM agents**.

Rather than replaying retrieved experiences verbatim, ReMemAgent treats memory as evidence: retrieve it, estimate whether it transfers, reconstruct it for the current state, and reject it when it is likely to cause negative transfer.

## The idea

A conventional memory-augmented agent often behaves like:

```text
retrieve → inject → act
```

ReMemAgent makes the decision explicit:

```text
observe
   ↓
retrieve
   ↓
trust + transferability
   ↓
reconstruct
   ↓
counterfactual routing
   ├── memory-guided
   ├── hybrid
   └── self-reasoning
   ↓
act → evaluate → consolidate / retire
```

> **Core question:** Can an agent learn not only what to remember, but when a memory should influence its reasoning?

## Research focus

| Capability | Purpose |
|---|---|
| **Reconstructive memory** | Convert past experience into state-aligned guidance instead of raw replay. |
| **Trust & transferability** | Estimate whether retrieved experience is reliable outside its original context. |
| **Counterfactual routing** | Compare memory-guided reasoning against self-reasoning before committing to memory. |
| **Failure memory** | Preserve useful failure evidence and avoidance rules. |
| **Lifecycle management** | Validate, consolidate, stale, and retire memories instead of accumulating them indefinitely. |
| **Negative-transfer evaluation** | Measure when memory actively makes decisions worse. |
| **Ablation framework** | Compare memory policies under controlled synthetic conditions. |
| **Paired statistical analysis** | Evaluate matched seed deltas with exact sign-flip tests, Holm correction, and paired effect sizes. |

## Architecture

```text
                         ┌─────────────────────┐
                         │     Observation     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Memory Retrieval   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ Trust / Transferability      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Memory Reconstruction        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Counterfactual Memory Router │
                    └──────────────┬───────────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                  Memory / Hybrid     Self-reasoning
                         └─────────┬─────────┘
                                   ▼
                                Action
                                   │
                                   ▼
                           Outcome Attribution
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                    Consolidate          Retire
```

## Repository

```text
ReMemAgent/
├── remem/
│   ├── memory/          # Domain model, storage, retrieval, reconstruction
│   └── routing/         # Trust and counterfactual routing policies
├── experiments/         # Controlled research experiments and evaluation
├── tests/               # Unit and integration tests
├── docs/                # Architecture, adapters, reproducibility, research status
├── README.md
└── pyproject.toml
```

## Current status

**Active research prototype.**

The deterministic research core, external benchmark contracts, reproducibility metadata, controlled artifact persistence, readiness-evidence binding, paired seed-level statistical analysis, observability sidecars, and persisted verification attestations are implemented. Every new branch increment is expected to earn its own green Quality run before it is treated as verified.

The statistical layer supports exact paired sign-flip tests, Holm-Bonferroni correction across primary metrics, and paired Cohen's *d_z*. These are analysis primitives only; they do not constitute benchmark evidence until real matched runs are produced.

The repository does **not** claim benchmark improvements or production readiness until the corresponding real-world experiments have been executed, repeated, and reproduced. See [`docs/research-status.md`](docs/research-status.md) for the current evidence boundary, [`docs/research-experiment-protocol.md`](docs/research-experiment-protocol.md) for the canonical end-to-end experiment and claim protocol, and [`docs/strict-reproducibility.md`](docs/strict-reproducibility.md) for the fail-closed paired execution contract.

## Engineering principles

- **Explicit contracts** — typed domain objects and small composable interfaces.
- **Deterministic baselines** — research heuristics are reproducible and independently testable.
- **Learned components stay isolated** — checkpoint/model loading and inference remain outside deterministic memory heuristics.
- **Tests before claims** — behavior is covered before experimental conclusions are reported.
- **Failure is evidence** — unsuccessful experiences remain useful when they encode transferable avoidance knowledge.
- **Fail closed at research boundaries** — malformed evaluator output, provenance drift, source revision drift, readiness-evidence drift, and artifact tampering are rejected rather than guessed around.
- **Controlled complexity** — new components must justify their effect on latency, tokens, memory growth, and experimental interpretability.

## External benchmark execution

The repository exposes normalized ALFWorld/WebShop environment boundaries without forcing those third-party packages into the core dependency set. Caller-owned factories provide real environments and model policies.

Use `remem-benchmark` for single-condition runs and `remem-paired-benchmark` for matched baseline-versus-treatment experiments. Paired strict mode is intended for experiments that may become scientific evidence:

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 100 \
  --max-steps 50 \
  --seeds 11,17,29,43,71 \
  --environment-factory your_package.environments:build_webshop \
  --success-evaluator your_package.metrics:is_success \
  --baseline-policy-factory your_package.policies:build_baseline \
  --treatment-policy-factory your_package.policies:build_remem \
  --strict-reproducibility \
  --require-code-revision <REMEM_COMMIT> \
  --require-clean-working-tree \
  --require-dependency-version torch==<PINNED_VERSION> \
  --source-checkout webshop=/path/to/webshop \
  --require-source-revision webshop=<WEBSHOP_COMMIT> \
  --output artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.json.manifest.json
```

Before expensive measured inference, the same controlled contract can be exercised in readiness-only mode and persisted as independently verifiable evidence:

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 100 \
  --max-steps 50 \
  --seeds 11,17,29,43,71 \
  --environment-factory your_package.environments:build_webshop \
  --success-evaluator your_package.metrics:is_success \
  --baseline-policy-factory your_package.policies:build_baseline \
  --treatment-policy-factory your_package.policies:build_remem \
  --source-checkout webshop=/path/to/webshop \
  --require-source-revision webshop=<WEBSHOP_COMMIT> \
  --preflight-only \
  --preflight-evidence artifacts/webshop-readiness.json

remem-verify-preflight artifacts/webshop-readiness.json
```

A later measured run can require that exact readiness artifact. Fresh controlled admission must still match it before any measured episode is started:

```bash
remem-paired-benchmark \
  --benchmark webshop \
  --episodes 100 \
  --max-steps 50 \
  --seeds 11,17,29,43,71 \
  --environment-factory your_package.environments:build_webshop \
  --success-evaluator your_package.metrics:is_success \
  --baseline-policy-factory your_package.policies:build_baseline \
  --treatment-policy-factory your_package.policies:build_remem \
  --source-checkout webshop=/path/to/webshop \
  --require-source-revision webshop=<WEBSHOP_COMMIT> \
  --require-preflight-evidence artifacts/webshop-readiness.json \
  --output artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.json.manifest.json

remem-verify-benchmark \
  artifacts/webshop-paired.json \
  --manifest artifacts/webshop-paired.json.manifest.json \
  --preflight-evidence artifacts/webshop-readiness.json
```

Readiness evidence proves that the recorded runtime/source contract passed the controlled preflight represented by that artifact. It is not a benchmark result and does not establish model effectiveness. Evidence-bound measured artifacts record the readiness fingerprint in their identity-bound runtime provenance; verification requires the retained readiness JSON and rejects a missing, tampered, or unrelated object.

The placeholders are intentional. Upstream revisions, dependency versions, model checkpoints, and benchmark outcomes must come from the actual controlled execution environment; ReMemAgent does not invent them.

## Research evidence levels

The canonical protocol distinguishes four levels so engineering success cannot be mistaken for scientific evidence:

```text
E0  green engineering Quality workflow
E1  verified controlled preflight/readiness evidence
E2  verified measured benchmark artifact bundle
E3  repeated paired baseline-vs-treatment analysis across independent seeds
```

Claims should not exceed the highest retained evidence level actually achieved. Full definitions, required artifacts, negative-transfer reporting, and ablation guidance are in [`docs/research-experiment-protocol.md`](docs/research-experiment-protocol.md).

## Quality checks

The repository quality workflow runs on Python 3.11 and 3.12 and covers:

```text
pytest
ruff format --check
ruff check
mypy
pip check
compileall
wheel + source distribution builds
isolated distribution smoke tests
```

A green quality workflow is engineering evidence for the covered contracts. It is not evidence of benchmark improvement, statistical significance, or deployment readiness.

## Research lineage

ReMemAgent is an independent research implementation informed by work on agent memory, reconstructive retrieval, adaptive routing, and negative transfer. Related work guides the research questions; implementation and experimental extensions in this repository are developed independently.

## License

MIT License. See [LICENSE](LICENSE).
