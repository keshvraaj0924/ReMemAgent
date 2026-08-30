"""Stable fingerprints for reproducible experiment inputs and reports."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Sequence

from experiments.synthetic_negative_transfer import BenchmarkCase

SCHEMA_VERSION = "2"


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
    encoded = json.dumps(
        {"schema_version": SCHEMA_VERSION, "cases": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
