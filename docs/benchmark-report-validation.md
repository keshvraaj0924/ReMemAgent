# Benchmark report validation

`remem.benchmark_validation.validate_benchmark_run_report` is a structural integrity gate for benchmark results before publication or downstream analysis.

It verifies:

- non-empty benchmark and episode identifiers;
- unique episode identifiers;
- finite trajectory and step rewards;
- non-empty observations and actions;
- contiguous episode step indices;
- non-negative retained and final memory counts;
- consistency between transfer outcomes and their derived counts;
- configuration benchmark name, seed, and episode count against the report;
- valid `minimum_trust` and positive `max_steps` values;
- that no episode exceeds the configured step limit.

The validator deliberately does **not** evaluate scientific quality, statistical significance, benchmark validity, or model effectiveness. Those require actual experimental execution and separate analysis.

The function is intentionally side-effect free and raises `ValueError` on structural corruption. The benchmark serializer now invokes this validator before converting a report to a persisted payload, and repeated-report persistence validates every constituent run before aggregation. This makes invalid in-memory reports fail closed at the artifact boundary instead of allowing malformed results to be written and discovered later.
