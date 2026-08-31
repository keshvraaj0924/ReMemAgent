"""Stable fingerprints for reproducible experiment inputs and reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import TypeAlias

from experiments.synthetic_negative_transfer import BenchmarkCase

SCHEMA_VERSION = "2"
EXPERIMENT_SCHEMA_VERSION = "2"
EXPERIMENT_PROTOCOL_VERSION = "1"
ROUTING_HEURISTIC_VERSION = "1"
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]


def fingerprint_cases(cases: Sequence[BenchmarkCase]) -> str:
    """Return a stable SHA-256 fingerprint for an ordered benchmark case set.

    Every field that can change benchmark semantics is included in the payload,
    including memory attribution and measured transfer outcomes. The ordered
    sequence remains significant so dataset permutations cannot share a
    fingerprint accidentally.
    """

    payload: list[JsonValue] = [_case_to_json(case) for case in cases]
    return _fingerprint({"schema_version": SCHEMA_VERSION, "cases": payload})


def fingerprint_experiment_inputs(
    cases: Sequence[BenchmarkCase],
    configuration: Mapping[str, JsonValue],
) -> str:
    """Return a stable fingerprint for cases, configuration, and protocol versions.

    Configuration is canonicalized with sorted mapping keys while benchmark
    case order remains significant. Protocol and heuristic versions are part
    of the identity so changes to experiment semantics cannot silently reuse
    an old fingerprint.
    """

    normalized_configuration = dict(configuration)
    payload: dict[str, JsonValue] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "routing_heuristic_version": ROUTING_HEURISTIC_VERSION,
        "cases": [_case_to_json(case) for case in cases],
        "configuration": normalized_configuration,
    }
    return _fingerprint(payload)


def _case_to_json(case: BenchmarkCase) -> dict[str, JsonValue]:
    """Convert a benchmark case to the canonical JSON-compatible shape."""

    return {
        "case_id": case.case_id,
        "utility_with_memory": case.utility_with_memory,
        "utility_without_memory": case.utility_without_memory,
        "memory_id": case.memory_id,
        "transfer_success": case.transfer_success,
    }


def _fingerprint(payload: JsonObject) -> str:
    """Hash a JSON-compatible payload using deterministic serialization."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
