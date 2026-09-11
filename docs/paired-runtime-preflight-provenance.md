# Paired runtime preflight provenance

Paired benchmark artifacts produced by `remem-paired-benchmark` now record whether the CLI's runtime preflight completed before measured execution and whether that preflight was reset-only or included a concrete probe action.

The launcher writes two string-valued runtime-provenance fields only after `run_paired_external_benchmarks_with_preflight(...)` returns successfully:

- `paired_runtime_preflight`: `completed`
- `paired_runtime_preflight_probe_action`: the exact `--probe-action` value, or `reset-only` when no action probe was requested

Because runtime provenance participates in paired experiment identity construction, this distinguishes otherwise identical paired artifacts that used different preflight probes.

This metadata is evidence about the ReMemAgent CLI execution path, not an independent attestation that an upstream benchmark installation is correct. A successful reset-only preflight validates construction and reset. A successful action probe additionally validates one concrete step through the normalized adapter. Neither constitutes benchmark evidence or establishes model effectiveness.
