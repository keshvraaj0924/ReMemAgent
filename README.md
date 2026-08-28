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
├── README.md
└── pyproject.toml
```

## Current status

**Active research prototype.**

The project is deliberately building a deterministic, testable research baseline before introducing model-dependent training and external agent benchmarks.

The repository does **not** claim benchmark improvements or production readiness until the corresponding implementations, tests, and experiments have actually been executed and reproduced.

## Engineering principles

- **Explicit contracts** — typed domain objects and small composable interfaces.
- **Deterministic baselines** — research heuristics are reproducible and independently testable.
- **Learned components stay isolated** — model-based policies can replace heuristics without coupling them to the memory domain.
- **Tests before claims** — behavior is covered before experimental conclusions are reported.
- **Failure is evidence** — unsuccessful experiences remain useful when they encode transferable avoidance knowledge.
- **Controlled complexity** — new components must justify their effect on latency, tokens, and memory growth.

## Research lineage

ReMemAgent is an independent research implementation inspired by reconstructive-memory research, including [MemHarness](https://github.com/KnowledgeXLab/MemHarness). Related work informs the research direction; implementation and experimental extensions in this repository are developed independently.

## License

MIT License. See [LICENSE](LICENSE).
