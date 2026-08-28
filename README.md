# ReMemAgent

> **Learning when to remember, what to reconstruct, and when to forget.**

[![Status](https://img.shields.io/badge/status-research--prototype-orange)](https://github.com/keshvraaj0924/ReMemAgent)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ReMemAgent is a research framework for **adaptive, reconstructive memory in LLM agents**.

Instead of replaying retrieved experiences verbatim, ReMemAgent treats memory as evidence: retrieve it, estimate whether it transfers, reconstruct it against the current state, and reject it when it is likely to cause negative transfer.

## Why ReMemAgent?

Traditional memory-augmented agents often follow:

```text
retrieve → inject → act
```

That assumes a useful past experience remains useful when the current situation changes.

ReMemAgent follows:

```text
observe
  ↓
retrieve
  ↓
assess trust & transferability
  ↓
reconstruct for the current state
  ↓
route: memory / hybrid / self-reasoning
  ↓
act
  ↓
attribute outcome
  ↓
consolidate or retire memory
```

The central research question is:

> **Can an agent learn not only what to remember, but when a memory should influence its reasoning?**

## Core ideas

### Reconstruct, don't replay

Stored experiences are transformed into compact, state-aligned guidance instead of being copied directly into the agent context.

### Memory must earn trust

Retrieval relevance alone is insufficient. ReMemAgent tracks evidence such as empirical success, transferability, confidence, and failure history.

### Counterfactual routing

The router considers whether using memory is expected to outperform self-reasoning. Low-value or risky memories can be rejected.

### Failure memory

Failures are first-class experiences. The system can retain them as avoidance knowledge instead of silently discarding negative outcomes.

### Memory lifecycle

Memories have an explicit lifecycle rather than growing indefinitely:

```text
capture → validate → retrieve → use → evaluate → consolidate → stale → retire
```

## Architecture

```text
                         ┌──────────────────────┐
                         │      Observation     │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │  Memory Retrieval    │
                         └──────────┬───────────┘
                                    ↓
                    ┌──────────────────────────────┐
                    │ Trust / Transferability      │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │ Memory Reconstruction        │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │ Counterfactual Memory Router │
                    └──────────────┬───────────────┘
                                   ↓
                     ┌─────────────┴─────────────┐
                     ↓                           ↓
              Memory / Hybrid              Self-reasoning
                     └─────────────┬─────────────┘
                                   ↓
                                Action
                                   ↓
                           Outcome Attribution
                                   ↓
                    ┌──────────────┴──────────────┐
                    ↓                             ↓
             Consolidate                    Retire / Reject
```

## Repository structure

```text
ReMemAgent/
├── remem/
│   ├── memory/          # Memory domain, storage, retrieval and reconstruction
│   └── routing/         # Memory decision and counterfactual routing
├── experiments/         # Reproducible research experiments
├── tests/               # Unit and integration tests
├── README.md
└── pyproject.toml
```

## Research status

**Active research prototype.**

The deterministic memory engine is being built first so retrieval, reconstruction, routing, lifecycle behavior, and negative-transfer effects can be tested independently before introducing model-dependent training.

Benchmark results are intentionally not reported until the corresponding experiments are implemented and reproduced.

## Development principles

- **Readable first:** descriptive names, small functions, explicit contracts.
- **Typed interfaces:** public APIs use Python type annotations.
- **Deterministic baselines:** research heuristics remain reproducible and independently testable.
- **Learned components stay isolated:** model-based components can replace heuristics without rewriting the memory domain.
- **Tests before claims:** experimental improvements must be supported by reproducible evaluation.
- **Evidence over complexity:** every subsystem must justify its additional latency, tokens, or memory cost.

## Research lineage

ReMemAgent is an independent research implementation inspired by reconstructive-memory research, including [MemHarness](https://github.com/KnowledgeXLab/MemHarness). Related ideas are treated as research lineage; implementation and experimental extensions in this repository are developed independently.

## License

MIT License.

See [LICENSE](LICENSE) for the full text.
