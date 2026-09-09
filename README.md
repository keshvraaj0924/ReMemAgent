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
                         │     Observation      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Memory Retrieval  │
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
├── docs/                # Architecture, adapters, observability, research status
├── README.md
└── pyproject.toml
```

## Current status

**Active research prototype.**

The deterministic research core, external benchmark contracts, reproducibility metadata, and paired seed-level statistical analysis are implemented. The latest branch head still requires a fresh successful GitHub quality workflow before it can be described as green.

The statistical layer now supports exact paired sign-flip tests, Holm-Bonferroni correction across the primary metrics, and paired Cohen's *d_z*. These are analysis primitives only; they do not constitute benchmark evidence until real matched runs are produced.

This is an engineering verification statement, not a scientific result. The repository does **not** claim benchmark improvements or production readiness until the corresponding real-world experiments have been executed, repeated, and reproduced.

See [`docs/research-status.md`](docs/research-status.md) for the current evidence boundary, reproducibility contract, limitations, and next milestone.

## Engineering principles