"""Evidence-bound execution for controlled paired external benchmarks.

This module links a previously persisted readiness artifact to a later measured run.
It performs fresh controlled admission, compares the exact admitted runtime/source
snapshots with the persisted evidence, and only then starts measured episodes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.paired_source_preflight import (
    ControlledPairedBenchmarkResult,
    preflight_controlled_paired_external_benchmarks,
    run_admitted_paired_external_benchmarks,
)
from experiments.preflight_evidence import validate_preflight_evidence_matches_admission
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutRequirement


def run_evidence_bound_paired_external_benchmarks(
    readiness_evidence: Mapping[str, Any],
    baseline_spec: ExternalBenchmarkSpec,
    treatment_spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
    *,
    runtime_requirements: RuntimeRequirements,
    source_checkout_paths: Mapping[str, Path],
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement],
    baseline_label: str = "baseline",
    treatment_label: str = "treatment",
    probe_action: str | None = None,
) -> ControlledPairedBenchmarkResult:
    """Measure only when persisted readiness matches fresh controlled admission.

    Runtime requirements, source checkouts, and paired environment readiness are
    validated first. The resulting admitted snapshots are then compared against the
    supplied readiness artifact without recollecting mutable state. Any malformed,
    tampered, or stale evidence fails before measured episodes begin.
    """

    if not isinstance(readiness_evidence, Mapping):
        raise TypeError("readiness_evidence must be a mapping")

    preflight_result = preflight_controlled_paired_external_benchmarks(
        baseline_spec,
        treatment_spec,
        seeds,
        runtime_requirements=runtime_requirements,
        source_checkout_paths=source_checkout_paths,
        source_checkout_requirements=source_checkout_requirements,
        probe_action=probe_action,
    )
    validate_preflight_evidence_matches_admission(
        readiness_evidence,
        preflight_result,
        source_checkout_requirements,
    )
    return run_admitted_paired_external_benchmarks(
        preflight_result,
        baseline_spec,
        treatment_spec,
        seeds,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
    )


__all__ = ["run_evidence_bound_paired_external_benchmarks"]
