"""Reproducible experiment execution and result persistence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Callable, Sequence

from experiments.ablations import AblationResult, run_ablations
from experiments.reproducibility import (
    EXPERIMENT_SCHEMA_VERSION,
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


def save_report(report: ExperimentReport, output_path: str | Path) -> Path:
    """Persist an experiment report as deterministic, human-readable JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "seed": report.seed,
        "minimum_delta": report.minimum_delta,
        "case_ids": list(report.case_ids),
        "case_fingerprint": report.case_fingerprint,
        "experiment_fingerprint": report.experiment_fingerprint,
        "results": [
            {
                "strategy": result.strategy.value,
                "total_cases": result.total_cases,
                "selected_memory": result.selected_memory,
                "mean_utility": result.mean_utility,
                "negative_transfer_cases": result.negative_transfer_cases,
                "selected_negative_transfer_cases": result.selected_negative_transfer_cases,
                "routing_regret": result.routing_regret,
            }
            for result in report.results
        ],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _validate_unique_case_ids(cases: Sequence[BenchmarkCase]) -> None:
    """Reject duplicate identifiers before an experiment is executed."""

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
