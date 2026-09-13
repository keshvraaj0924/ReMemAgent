# Research evidence semantic-binding verification

`remem-research-evidence verify` can optionally perform semantic cross-checks between retained evidence artifacts that have already passed exact-byte verification. These checks close identity gaps between the frozen experiment declaration, the measured paired report, and the benchmark verification attestation.

Use these checks after the experiment-specific benchmark verifier has produced a persisted attestation and after the final evidence record has indexed the retained files. They complement exact-byte record verification; they do not replace preflight, benchmark, statistical, or scientific validation.

## Plan-binding roles

For `--require-plan-binding`, the evidence record must contain:

- `experiment_plan`: the persisted `ResearchExperimentPlan` used to admit measured execution;
- `verification`: the persisted JSON attestation emitted by `remem-verify-benchmark` for the measured report.

The normal evidence-record verifier first proves that every indexed file still matches its recorded path, byte count, and SHA-256. Plan binding then verifies that:

1. the frozen plan is valid and declares the same ReMemAgent revision as the evidence record;
2. the plan experiment name matches the evidence-record experiment name;
3. the verification attestation's `experiment_plan_sha256` matches the plan's canonical SHA-256;
4. `experiment_plan_file_sha256` matches the exact retained plan file bytes;
5. the attested experiment-plan name matches the frozen plan;
6. the attested ReMemAgent revision matches the frozen plan.

Any missing canonical role, malformed retained JSON, missing binding field, revision drift, name drift, canonical-plan substitution, or exact-file substitution fails closed.

## Report-binding roles

For `--require-report-binding`, the evidence record must contain:

- `paired_report`: the exact persisted paired benchmark report that was verified;
- `verification`: the persisted JSON attestation emitted by `remem-verify-benchmark` for that report.

Report binding cross-checks the attestation against the exact-byte identity already frozen in the evidence record. It requires:

1. the attestation `sha256` to equal the retained `paired_report` SHA-256;
2. the attestation `byte_count` to equal the retained `paired_report` byte count.

This rejects a subtle but important substitution case where both the report and verification files are individually intact but the verification attestation was produced for a different report.

## Verification command

For a fully retained plan-bound paired experiment, run both checks together:

```bash
remem-research-evidence verify \
  artifacts/research-evidence.json \
  --expected-revision <REMEM_COMMIT> \
  --require-plan-binding \
  --require-report-binding \
  --json
```

On success, the JSON summary retains the existing exact evidence-record identity and adds the bindings requested by the caller:

```json
{
  "experiment_plan_sha256": "<canonical-plan-sha256>",
  "paired_report_sha256": "<exact-retained-report-sha256>",
  "plan_binding_verified": true,
  "report_binding_verified": true
}
```

Without the binding flags, existing evidence records and their machine-readable verification output preserve the legacy contract.

## Evidence boundary

Successful plan and report binding establishes that the exact retained frozen plan, paired benchmark report, and benchmark verification attestation refer to one consistent retained evidence chain. It does **not** establish that an E3 claim is scientifically justified, that benchmark outcomes improved, that the attestation is digitally signed, or that the system is production-ready. Statistical analysis, protocol review, benchmark validity, model-policy semantic enforcement, and claim selection remain separate research responsibilities.
