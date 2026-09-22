# Claims and evidence matrix

This document is the final guardrail between ReMemAgent's implemented software contracts and empirical research claims. It is intentionally conservative: a checked engineering capability is not evidence of improved agent performance.

## Evidence states

- **Implemented** — code exists and is covered by the repository's automated quality gates.
- **Synthetic evidence required** — execute the controlled negative-transfer benchmark under the reproducibility contract.
- **Environment evidence required** — execute the real ALFWorld/WebShop integration and preserve run artifacts.
- **Training evidence required** — execute GRPO/verl-agent training and evaluation with sufficient provenance.
- **Unsupported** — do not make the claim from the current repository state.

## Matrix

| Claim or capability | Current evidence state | Minimum evidence before reporting a result |
| --- | --- | --- |
| Core memory storage, retrieval, deduplication, reconstruction, trust/transferability, routing, failure memory, and lifecycle contracts exist | Implemented | Passing CI for the exact revision being described |
| ReMemAgent reduces negative transfer | Synthetic evidence required | Repeated seeded benchmark runs against declared baselines, raw per-run outputs, uncertainty, exact revision/configuration, and verified evidence bundle |
| ReMemAgent improves ALFWorld task success | Environment evidence required | Actual ALFWorld executions across declared seeds/baselines with environment/model versions and verified raw evidence |
| ReMemAgent improves WebShop task success | Environment evidence required | Actual WebShop executions across declared seeds/baselines with environment/model versions and verified raw evidence |
| Learned routing improves over deterministic routing | Training evidence required | Completed GRPO/verl-agent training plus held-out evaluation against deterministic baselines, checkpoints or sufficient training provenance, seeds, and verified evidence |
| ReMemAgent reduces latency or token usage | Unsupported until measured | Instrumented end-to-end runs under identical model/task/retrieval budgets; report distributions or uncertainty rather than isolated measurements |
| ReMemAgent is production ready | Unsupported | Separate operational validation covering deployment, load, reliability, security, recovery, dependency/runtime constraints, and service-level objectives |

## Required evidence chain

A quantitative table or headline claim is reportable only when its complete evidence chain can be reconstructed:

1. Record deterministic experiment identity and runtime provenance before execution.
2. Preserve raw per-seed/per-task outputs; failures and incomplete runs must remain distinguishable from numeric zero.
3. Persist aggregate `ExperimentSummary` output only from validated metrics.
4. Include the aggregate summary and every mandatory raw input in one `EvidenceContract`.
5. Build and persist the corresponding artifact manifest/evidence bundle.
6. Before consuming the aggregate for reporting, call `verify_reportable_summary(...)` against the evidence root. Schema validation of the summary alone is insufficient.
7. Record the exact repository revision, configuration, seeds, environment/model versions, baseline definitions, and metric semantics beside the reported result.

If any link in this chain is absent or fails verification, downgrade the statement to an implementation description and do not present the number as research evidence.

## Baseline discipline

Comparisons must keep task set, model, prompt contract, retrieval budget, memory budget, environment version, and random seeds fixed wherever the compared method permits. At minimum, use the supported no-memory/self-reasoning, replay-style memory, reconstructive-without-routing, and full-routing variants when testing the contribution of reconstructive memory and routing.

Ablations should isolate one mechanism at a time. Do not interpret an integration-boundary test as an environment result, or a successful training invocation as evidence of policy improvement.

## Reporting language

Prefer precise statements such as "the adapter contract passes the repository test suite" or "on the declared synthetic benchmark across N seeds, policy A changed metric X by Y with uncertainty Z." Avoid "better," "safer," "faster," "robust," or "production ready" unless the corresponding evidence in this matrix has actually been produced and verified.

## Related contracts

- `docs/research-status.md` — current implemented scope and explicitly unmade claims.
- `docs/experiment-reporting.md` — normative reporting and evidence rules.
- `docs/research-execution-checklist.md` — ordered execution and preservation checklist.
- `docs/reproducibility.md` — deterministic experiment identity.
- `docs/runtime-provenance.md` — runtime metadata requirements.
- `docs/policy-contract.md` — policy semantics and boundaries.
