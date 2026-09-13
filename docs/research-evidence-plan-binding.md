# Research evidence plan-binding verification

`remem-research-evidence verify` can optionally perform a semantic cross-check between the frozen experiment plan and the benchmark verification attestation retained in a research evidence record.

Use this check after the experiment-specific benchmark verifier has produced a plan-bound attestation and after the final evidence record has indexed the retained files. It complements exact-byte record verification; it does not replace preflight, benchmark, statistical, or scientific validation.

## Required canonical roles

The evidence record must contain these roles:

- `experiment_plan`: the persisted `ResearchExperimentPlan` used to admit measured execution;
- `verification`: the persisted JSON attestation emitted by `remem-verify-benchmark` for the measured report.

The normal evidence-record verifier first proves that every indexed file still matches its recorded path, byte count, and SHA-256. With `--require-plan-binding`, it then verifies that:

1. the frozen plan is valid and declares the same ReMemAgent revision as the evidence record;
2. the plan experiment name matches the evidence-record experiment name;
3. the verification attestation's `experiment_plan_sha256` matches the plan's canonical SHA-256;
4. `experiment_plan_file_sha256` matches the exact retained plan file bytes;
5. the attested experiment-plan name matches the frozen plan;
6. the attested ReMemAgent revision matches the frozen plan.

Any missing canonical role, malformed retained JSON, missing binding field, revision drift, name drift, canonical-plan substitution, or exact-file substitution fails closed.

## Verification command

```bash
remem-research-evidence verify \
  artifacts/research-evidence.json \
  --expected-revision <REMEM_COMMIT> \
  --require-plan-binding \
  --json
```

On success, the JSON summary retains the existing exact evidence-record identity and adds:

```json
{
  "experiment_plan_sha256": "<canonical-plan-sha256>",
  "plan_binding_verified": true
}
```

Without `--require-plan-binding`, existing evidence records and their machine-readable verification output preserve the legacy contract.

## Evidence boundary

A successful plan-binding check establishes that the exact retained frozen plan and benchmark verification attestation refer to the same declared experiment identity and ReMemAgent revision. It does **not** establish that an E3 claim is scientifically justified, that benchmark outcomes improved, that the attestation is digitally signed, or that the system is production-ready. Statistical analysis, protocol review, benchmark validity, and claim selection remain separate research responsibilities.
