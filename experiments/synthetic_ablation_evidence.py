"""Load and verify persisted synthetic threshold-ablation evidence.

The reporting layer must not trust aggregate JSON merely because it parses. This
module reconstructs the exact benchmark inputs, reruns the deterministic
ablation, and rejects evidence whose stored outputs differ from recomputation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.synthetic_negative_transfer import BenchmarkCase, BenchmarkResult
from experiments.synthetic_threshold_ablation import ThresholdAblationResult, run_threshold_ablation

_ABLATION_NAME = "counterfactual_router_minimum_delta"
_CASE_FIELDS = {"case_id", "utility_with_memory", "utility_without_memory"}
_RESULT_FIELDS = {"minimum_delta", "benchmark_result"}
_BENCHMARK_RESULT_FIELDS = {
    "total_cases",
    "memory_selected",
    "self_reasoning_selected",
    "negative_transfer_cases",
    "selected_negative_transfer_cases",
    "avoided_negative_transfer_cases",
    "routing_regret",
    "mean_routing_regret",
    "memory_induced_negative_transfer_rate",
    "negative_transfer_avoidance_rate",
    "negative_transfer_rate",
}


@dataclass(frozen=True, slots=True)
class VerifiedThresholdAblationEvidence:
    """Typed threshold-ablation evidence verified by deterministic replay."""

    cases: tuple[BenchmarkCase, ...]
    minimum_deltas: tuple[float, ...]
    results: tuple[ThresholdAblationResult, ...]


def load_verified_threshold_ablation_evidence(
    input_path: str | Path,
) -> VerifiedThresholdAblationEvidence:
    """Load threshold evidence and fail closed unless deterministic replay matches."""

    payload = _load_json_object(Path(input_path))
    if set(payload) != {"ablation", "cases", "minimum_deltas", "results"}:
        raise ValueError("threshold ablation evidence has an unexpected schema")
    if payload["ablation"] != _ABLATION_NAME:
        raise ValueError("threshold ablation evidence has an unexpected ablation name")

    cases = _parse_cases(payload["cases"])
    minimum_deltas = _parse_thresholds(payload["minimum_deltas"])
    stored_results = _parse_results(payload["results"])
    recomputed_results = tuple(run_threshold_ablation(list(cases), list(minimum_deltas)))

    if stored_results != recomputed_results:
        raise ValueError("threshold ablation evidence does not match deterministic replay")

    return VerifiedThresholdAblationEvidence(cases, minimum_deltas, stored_results)


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object with a stable validation error boundary."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load threshold ablation evidence from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("threshold ablation evidence must contain a JSON object")
    return payload


def _parse_cases(value: Any) -> tuple[BenchmarkCase, ...]:
    """Parse the exact benchmark cases embedded in evidence."""

    if not isinstance(value, list):
        raise ValueError("threshold ablation cases must be a JSON array")
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _CASE_FIELDS:
            raise ValueError(f"threshold ablation case at index {index} has an unexpected schema")
        case_id = item["case_id"]
        with_memory = item["utility_with_memory"]
        without_memory = item["utility_without_memory"]
        if not isinstance(case_id, str):
            raise ValueError(f"threshold ablation case at index {index} has invalid case_id")
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                utility_with_memory=_finite_number(with_memory, "utility_with_memory"),
                utility_without_memory=_finite_number(without_memory, "utility_without_memory"),
            )
        )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("threshold ablation case_id values must be unique")
    return tuple(cases)


def _parse_thresholds(value: Any) -> tuple[float, ...]:
    """Parse finite, unique routing thresholds in their recorded order."""

    if not isinstance(value, list) or not value:
        raise ValueError("threshold ablation minimum_deltas must be a non-empty JSON array")
    thresholds = tuple(_finite_number(item, "minimum_delta") for item in value)
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("threshold ablation minimum_deltas must be unique")
    return thresholds


def _parse_results(value: Any) -> tuple[ThresholdAblationResult, ...]:
    """Parse stored outputs without silently accepting derived-metric drift."""

    if not isinstance(value, list):
        raise ValueError("threshold ablation results must be a JSON array")
    return tuple(_parse_result(item, index) for index, item in enumerate(value))


def _parse_result(item: Any, index: int) -> ThresholdAblationResult:
    if not isinstance(item, dict) or set(item) != _RESULT_FIELDS:
        raise ValueError(f"threshold ablation result at index {index} has an unexpected schema")
    minimum_delta = _finite_number(item["minimum_delta"], "minimum_delta")
    raw_result = item["benchmark_result"]
    if not isinstance(raw_result, dict) or set(raw_result) != _BENCHMARK_RESULT_FIELDS:
        raise ValueError(f"benchmark result at index {index} has an unexpected schema")

    integer_fields = (
        "total_cases",
        "memory_selected",
        "self_reasoning_selected",
        "negative_transfer_cases",
        "selected_negative_transfer_cases",
        "avoided_negative_transfer_cases",
    )
    integer_values: dict[str, int] = {}
    for field in integer_fields:
        field_value = raw_result[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
            raise ValueError(
                f"benchmark result field {field} at index {index} must be non-negative int"
            )
        integer_values[field] = field_value

    benchmark_result = BenchmarkResult(
        **integer_values,
        routing_regret=_finite_number(raw_result["routing_regret"], "routing_regret"),
    )
    expected_derived = {
        "mean_routing_regret": benchmark_result.mean_routing_regret,
        "memory_induced_negative_transfer_rate": benchmark_result.memory_induced_negative_transfer_rate,
        "negative_transfer_avoidance_rate": benchmark_result.negative_transfer_avoidance_rate,
        "negative_transfer_rate": benchmark_result.negative_transfer_rate,
    }
    for field, expected in expected_derived.items():
        observed = _finite_number(raw_result[field], field)
        if observed != expected:
            raise ValueError(
                f"benchmark result derived field {field} at index {index} is inconsistent"
            )
    return ThresholdAblationResult(minimum_delta=minimum_delta, benchmark_result=benchmark_result)


def _finite_number(value: Any, field_name: str) -> float:
    """Convert a JSON number to float while rejecting booleans and non-finite values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


__all__ = ["VerifiedThresholdAblationEvidence", "load_verified_threshold_ablation_evidence"]
