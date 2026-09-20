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

## Architecture

```text
Observation
    │
    ▼
Memory Retrieval
    │
    ▼
Trust / Transferability
    │
    ▼
Memory Reconstruction
    │
    ▼
Counterfactual Memory Router
    ├── memory-guided
    ├── hybrid
    └── self-reasoning
    │
    ▼
Action → Outcome Attribution → Consolidate / Retire
```

## Implemented framework

The repository currently contains deterministic, testable foundations for:

- memory domain modeling, storage, retrieval, and deduplication;
- memory reconstruction and trust/transferability policies;
- counterfactual routing and failure-memory handling;
- consolidation and memory lifecycle management;
- synthetic negative-transfer evaluation, ablations, and metrics;
- ALFWorld/WebShop adapter boundaries;
- GRPO/verl-agent integration boundaries;
- experiment reproducibility and runtime provenance;
- CI/quality tooling and lightweight in-process observability.

These are engineering capabilities and experiment boundaries—not evidence of benchmark superiority. See [`docs/research-status.md`](docs/research-status.md) for the explicit evidence and claims policy.

## Repository

```text
ReMemAgent/
├── remem/               # Memory, routing, integrations, metrics, observability
├── experiments/         # Controlled research experiments and evaluation
├── tests/               # Unit and integration tests
├── docs/                # Architecture, contracts, reproducibility, research status
├── README.md
└── pyproject.toml
```

## Current status

**Active research prototype.**

The deterministic engineering foundation is implemented and quality-gated. The next research milestone is empirical: execute controlled synthetic and external-environment evaluations under the repository's reproducibility contract, preserve raw outputs and provenance, and only then make quantitative conclusions.

The repository does **not** currently claim state-of-the-art results, statistically significant task-success improvements, successful GRPO policy improvement, superior ALFWorld/WebShop performance, or production readiness.

## Evidence discipline

ReMemAgent distinguishes software verification from research evidence:

1. **Unit-tested behavior** verifies a software contract.
2. **Synthetic results** apply only to the recorded controlled benchmark.
3. **Environment results** require actual ALFWorld/WebShop executions with preserved configuration.
4. **Training results** require completed training plus reproducible provenance and evaluation.
5. **Research claims** require repeated runs, suitable baselines, uncertainty reporting, and reproducibility.

No benchmark number should be added to this README unless its generating revision, configuration, seeds, and raw outputs are preserved.

## Engineering principles

- **Explicit contracts** — typed domain objects and small composable interfaces.
- **Deterministic baselines** — research heuristics are reproducible and independently testable.
- **Learned components stay isolated** — model-based policies can replace heuristics without coupling them to the memory domain.
- **Tests before claims** — behavior is covered before experimental conclusions are reported.
- **Failure is evidence** — unsuccessful experiences remain useful when they encode transferable avoidance knowledge.
- **Controlled complexity** — new components must justify their effect on latency, tokens, and memory growth.

## Research documentation

Start with [`docs/research-status.md`](docs/research-status.md) for current evidence boundaries and [`docs/experiment-reporting.md`](docs/experiment-reporting.md) for the minimum evidence required before reporting quantitative results. Reproducibility, runtime provenance, policy contracts, and architecture are documented in `docs/reproducibility.md`, `docs/runtime-provenance.md`, `docs/policy-contract.md`, and `docs/ARCHITECTURE.md`.

## Research lineage

ReMemAgent is an independent research implementation inspired by reconstructive-memory research, including [MemHarness](https://github.com/KnowledgeXLab/MemHarness). Related work informs the research direction; implementation and experimental extensions in this repository are developed independently.

## License

MIT License. See [LICENSE](LICENSE).
