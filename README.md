# ReMemAgent

> **Learning when to remember, what to reconstruct, and when to forget.**

ReMemAgent is a research framework for adaptive memory in LLM agents. It extends reconstructive-memory approaches with **counterfactual memory routing**: before relying on retrieved experience, an agent estimates whether memory is likely to improve the current decision or introduce negative transfer.

## Research thesis

Most memory-augmented agents treat retrieval as an implicit benefit: retrieve a relevant experience, inject it into context, and continue reasoning. ReMemAgent treats memory as **evidence that must earn trust**.

```text
Observation
    ↓
Adaptive Retrieval
    ↓
Memory Reconstruction
    ↓
Trust / Transferability Estimation
    ↓
Counterfactual Memory Routing
    ├── Memory-guided reasoning
    ├── Memory + self-reasoning
    └── Self-reasoning only
    ↓
Action
    ↓
Outcome Attribution
    ↓
Memory Consolidation
```

## Planned contributions

- **Counterfactual Memory Router** — estimate the expected benefit of using memory versus reasoning without it.
- **Memory Trust & Transferability** — model whether an experience transfers beyond its original context.
- **Positive + Failure Memory** — retain useful successes and high-value failures/avoidance rules.
- **Memory Consolidation** — evolve episodic experiences into validated semantic and procedural knowledge.
- **Negative Transfer Evaluation** — directly measure when memory makes an agent worse.
- **Adaptive Memory Budgeting** — learn how much memory to retrieve based on task difficulty, confidence, and expected utility.

## Research lineage

ReMemAgent is an independent research implementation inspired by recent work on reconstructive memory for agents, including [MemHarness](https://github.com/KnowledgeXLab/MemHarness). ReMemAgent will reproduce relevant baselines where practical and clearly separate reproduced components from our extensions.

## Status

🚧 **Research prototype — active development**

The first milestone focuses on a deterministic counterfactual memory engine and evaluation harness before integration with large-scale agent training.

## Roadmap

- [x] Define research hypothesis and architecture
- [ ] Implement typed memory model
- [ ] Implement trust / transferability scoring
- [ ] Implement counterfactual router
- [ ] Implement failure memory
- [ ] Implement consolidation lifecycle
- [ ] Build synthetic negative-transfer benchmark
- [ ] Add ablation framework
- [ ] Integrate ALFWorld
- [ ] Integrate GRPO training
- [ ] Report latency, token cost, memory growth, and task success

## License

MIT
