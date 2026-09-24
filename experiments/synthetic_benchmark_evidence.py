"""Load and verify persisted synthetic negative-transfer benchmark evidence.

Persisted benchmark aggregates are treated as untrusted input. Verification
reconstructs the exact cases and router configuration, reruns the deterministic
benchmark, and accepts the artifact only when stored and recomputed results
match exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.synthetic_negative_transfer import BenchmarkCase, BenchmarkResult, run_benchmark
from remem.routing.counterfactual import CounterfactualRouter

_BENCHMARK_NAME = "synthetic_negative_transfer"
_CASE_FIELDS = {"case_id", "utility_with_memory", "utility_without_memory"}
_ROUTER_FIELDS = {"minimum_delta"}
_RESULT_FIELDS = {
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
class VerifiedSyntheticBenchmarkEvidence:
    """Synthetic benchmark evidence accepted after deterministic replay."""

    cases: tuple[BenchmarkCase, ...]
    minimum_delta: float
    result: BenchmarkResult


def load_verified_synthetic_benchmark_evidence(
    input_path: str | Path,
) -> VerifiedSyntheticBenchmarkEvidence:
    """Load benchmark evidence and fail closed unless deterministic replay matches."""

    payload = _load_json_object(Path(input_path))
    if set(payload) != {"benchmark", "cases", "router", "result"}:
        raise ValueError("synthetic benchmark evidence has an unexpected schema")
    if payload["benchmark"] != _BENCHMARK_NAME:
        raise ValueError("synthetic benchmark evidence has an unexpected benchmark name")

    cases = _parse_cases(payload["cases"])
    minimum_delta = _parse_router(payload["router"])
    stored_result = _parse_result(payload["result"])
    recomputed_result = run_benchmark(
        list(cases), CounterfactualRouter(minimum_delta=minimum_delta)
    )
    if stored_result != recomputed_result:
        raise ValueError("synthetic benchmark evidence does not match deterministic replay")

    return VerifiedSyntheticBenchmarkEvidence(cases, minimum_delta, stored_result)


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object behind a stable validation boundary."""

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load synthetic benchmark evidence from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("synthetic benchmark evidence must contain a JSON object")
    return payload


def _parse_cases(value: Any) -> tuple[BenchmarkCase, ...]:
    """Parse exact matched cases embedded in the evidence artifact."""

    if not isinstance(value, list):
        raise ValueError("synthetic benchmark cases must be a JSON array")
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _CASE_FIELDS:
            raise ValueError(f"synthetic benchmark case at index {index} has an unexpected schema")
        case_id = item["case_id"]
        if not isinstance(case_id, str):
            raise ValueError(f"synthetic benchmark case at index {index} has invalid case_id")
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                utility_with_memory=_finite_number(
                    item["utility_with_memory"], "utility_with_memory"
                ),
                utility_without_memory=_finite_number(
                    item["utility_without_memory"], "utility_without_memory"
                ),
            )
        )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("synthetic benchmark case_id values must be unique")
    return tuple(cases)


def _parse_router(value: Any) -> float:
    """Parse the exact counterfactual router configuration used by the run."""

    if not isinstance(value, dict) or set(value) != _ROUTER_FIELDS:
        raise ValueError("synthetic benchmark router has an unexpected schema")
    return _finite_number(value["minimum_delta"], "minimum_delta")


def _parse_result(value: Any) -> BenchmarkResult:
    """Parse aggregate outputs while independently validating derived metrics."""

    if not isinstance(value, dict) or set(value) != _RESULT_FIELDS:
        raise ValueError("synthetic benchmark result has an unexpected schema")

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
        field_value = value[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
            raise ValueError(f"synthetic benchmark result field {field} must be non-negative int")
        integer_values[field] = field_value

    result = BenchmarkResult(
        **integer_values,
        routing_regret=_finite_number(value["routing_regret"], "routing_regret"),
    )
    expected_derived = {
        "mean_routing_regret": result.mean_routing_regret,
        "memory_induced_negative_transfer_rate": result.memory_induced_negative_transfer_rate,
        "negative_transfer_avoidance_rate": result.negative_transfer_avoidance_rate,
        "negative_transfer_rate": result.negative_transfer_rate,
    }
    for field, expected in expected_derived.items():
        if _finite_number(value[field], field) != expected:
            raise ValueError(f"synthetic benchmark derived field {field} is inconsistent")
    return result


def _finite_number(value: Any, field_name: str) -> float:
    """Convert a JSON number to float while rejecting booleans and non-finite values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


__all__ = [
    "VerifiedSyntheticBenchmarkEvidence",
    "load_verified_synthetic_benchmark_evidence",
]
