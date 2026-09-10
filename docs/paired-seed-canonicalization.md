# Canonical paired seed ordering

Paired baseline-versus-treatment experiments treat the requested seeds as an independent **set**, not as an execution-order control surface.

Before either runtime preflight or measured execution, ReMemAgent now validates the seed sequence and sorts it into ascending canonical order. Counterbalancing is then assigned from that canonical order:

- even canonical positions run baseline before treatment;
- odd canonical positions run treatment before baseline.

This means `(17, 11)` and `(11, 17)` produce the same per-seed AB/BA execution plan. The behavior matters because paired benchmark artifacts persist reports and comparison seeds in deterministic seed order. Without canonicalization, two scientifically equivalent seed sets could experience different warm-up, throttling, cache, or temporal-drift exposure while their persisted artifacts no longer retained the caller ordering needed to reconstruct that difference.

Canonicalization applies identically to runtime preflight and measured execution so preflight cannot reintroduce caller-order-dependent warm-up immediately before measurement.

## What this guarantees

The framework guarantees deterministic condition-first assignment for a given validated seed set, identical seed ownership across both conditions, and order-invariant paired execution when callers provide the same seeds in a different sequence.

## What this does not guarantee

Canonical ordering does not eliminate all temporal drift or external service nondeterminism. Real ALFWorld/WebShop claims still require controlled multi-seed runs, recorded runtime provenance, pinned dependencies/model checkpoints where possible, and analysis of the resulting measured artifacts. No benchmark-effectiveness result is implied by this orchestration contract.
