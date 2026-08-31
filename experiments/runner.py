"""Reproducible experiment execution and result persistence."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from experiments.ablations import AblationResult, run_ablations
from experiments.reproducibility import (
    EXPERIMENT_PROTOCOL_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    ROUTING_HEURISTIC_VERSION,
    fingerprint_cases,
    fingerprint_experiment_inputs,
)
from experiments.synthetic_negative_transfer import BenchmarkCase
from remem.routing.counterfactual import CounterfactualRouter

DEFAULT_SEED = 42


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration that makes a synthetic experiment reproducible."""

    seed: int = DEFAULT_SEED
    minimum_delta: float = 0.05


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Serializable output from one reproducible ablation experiment."""

    seed: int
    minimum_delta: float
    case_ids: tuple[str, ...]
    case_fingerprint: str
    experiment_fingerprint: str
    results: tuple[AblationResult, ...]


def run_reproducible_ablation(
    cases_factory: Callable[[random.Random], Sequence[BenchmarkCase]],
    config: ExperimentConfig | None = None,
) -> ExperimentReport:
    """Run matched ablations with a dedicated seeded random generator."""

    selected_config = config or ExperimentConfig()
    if selected_config.minimum_delta < 0:
        raise ValueError("minimum_delta must be non-negative")

    generator = random.Random(selected_config.seed)
    cases = list(cases_factory(generator))
    _validate_unique_case_ids(cases)
    router = CounterfactualRouter(minimum_delta=selected_config.minimum_delta)
    results = run_ablations(cases, router)
    experiment_fingerprint = fingerprint_experiment_inputs(
        cases,
        {
            "seed": selected_config.seed,
            "minimum_delta": selected_config.minimum_delta,
        },
    )
    return ExperimentReport(
        seed=selected_config.seed,
        minimum_delta=selected_config.minimum_delta,
        case_ids=tuple(case.case_id for case in cases),
        case_fingerprint=fingerprint_cases(cases),
        experiment_fingerprint=experiment_fingerprint,
        results=tuple(results),
    )


def run_repeated_ablations(
    cases_factory: Callable[[random.Random], Sequence[BenchmarkCase]],
    seeds: Sequence[int],
    config: ExperimentConfig | None = None,
) -> tuple[ExperimentReport, ...]:
    """Run the same ablation protocol independently for each requested seed.

    Each seed receives its own ``random.Random`` instance through
    :func:`run_reproducible_ablation`. The returned reports remain separate so
    downstream analysis can compute paired or per-seed statistics without losing
    the provenance of an individual run.
    """

    selected_seeds = tuple(seeds)
    if not selected_seeds:
        raise ValueError("seeds must contain at least one seed")
    if len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("seeds must be unique")

    selected_config = config or ExperimentConfig()
    return tuple(
        run_reproducible_ablation(
            cases_factory,
            replace(selected_config, seed=seed),
        )
        for seed in selected_seeds
    )


def save_report(report: ExperimentReport, output_path: str | Path) -> Path:
    """Persist an experiment report as deterministic, human-readable JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "routing_heuristic_version": ROUTING_HEURISTIC_VERSION,
        "seed": report.seed,
        "minimum_delta": report.minimum_delta,
        "case_ids": list(report.case_ids),
        "case_fingerprint": report.case_fingerprint,
        "experiment_fingerprint": report.experiment_fingerprint,
        "results": [_serialize_result(result) for result in report.results],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def save_repeated_reports(
    reports: Sequence[ExperimentReport],
    output_path: str | Path,
) -> Path:
    """Persist multiple seed reports while preserving each run's provenance."""

    selected_reports = tuple(reports)
    _validate_unique_report_seeds(selected_reports)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "routing_heuristic_version": ROUTING_HEURISTIC_VERSION,
        "seeds": [report.seed for report in selected_reports],
        "reports": [_serialize_report(report) for report in selected_reports],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _serialize_report(report: ExperimentReport) -> dict[str, object]:
    """Convert one report to a JSON-compatible mapping."""

    return {
        "seed": report.seed,
        "minimum_delta": report.minimum_delta,
        "case_ids": list(report.case_ids),
        "case_fingerprint": report.case_fingerprint,
        "experiment_fingerprint": report.experiment_fingerprint,
        "results": [_serialize_result(result) for result in report.results],
    }


def _serialize_result(result: AblationResult) -> dict[str, object]:
    """Convert one ablation result to a JSON-compatible mapping."""

    return {
        "strategy": result.strategy.value,
        "total_cases": result.total_cases,
        "selected_memory": result.selected_memory,
        "mean_utility": result.mean_utility,
        "negative_transfer_cases": result.negative_transfer_cases,
        "selected_negative_transfer_cases": result.selected_negative_transfer_cases,
        "routing_regret": result.routing_regret,
    }


def _validate_unique_case_ids(cases: Sequence[BenchmarkCase]) -> None:
    """Reject duplicate identifiers before an experiment is executed."""

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")


def _validate_unique_report_seeds(reports: Sequence[ExperimentReport]) -> None:
    """Reject duplicate seeds so repeated-run files remain unambiguous."""

    seeds = [report.seed for report in reports]
    if len(seeds) != len(set(seeds)):
        raise ValueError("report seeds must be unique")
