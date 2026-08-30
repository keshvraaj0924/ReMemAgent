"""Stable fingerprints for reproducible experiment inputs and reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from typing import TypeAlias

from experiments.synthetic_negative_transfer import BenchmarkCase

SCHEMA_VERSION = "2"
EXPERIMENT_SCHEMA_VERSION = "1"
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def fingerprint_cases(cases: Sequence[BenchmarkCase]) -> str:
    """Return a stable SHA-256 fingerprint for an ordered benchmark case set.

    Every field that can change benchmark semantics is included in the payload,
    including memory attribution and measured transfer outcomes. The ordered
    sequence remains significant so dataset permutations cannot share a
    fingerprint accidentally.
    """

    payload = [
        {
            "case_id": case.case_id,
            "utility_with_memory": case.utility_with_memory,
            "utility_without_memory": case.utility_without_memory,
            "memory_id": case.memory_id,
            "transfer_success": case.transfer_success,
        }
        for case in cases
    ]
    return _fingerprint({"schema_version": SCHEMA_VERSION, "cases": payload})


def fingerprint_experiment_inputs(
    cases: Sequence[BenchmarkCase],
    configuration: Mapping[str, JsonValue],
) -> str:
    """Return a stable fingerprint for benchmark cases and execution configuration.

    Configuration is canonicalized with sorted mapping keys while benchmark
    case order remains significant. This makes thresholds, seeds, and other
    explicitly recorded experiment settings part of the reproducibility
    identity instead of relying on undocumented runtime defaults.
    """

    return _fingerprint(
        {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "cases": [
                {
                    "case_id": case.case_id,
                    "utility_with_memory": case.utility_with_memory,
                    "utility_without_memory": case.utility_without_memory,
                    "memory_id": case.memory_id,
                    "transfer_success": case.transfer_success,
                }
                for case in cases
            ],
            "configuration": dict(configuration),
        }
    )


def _fingerprint(payload: JsonValue | dict[str, object]) -> str:
    """Hash a JSON-compatible payload using deterministic serialization."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
