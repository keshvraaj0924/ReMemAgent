"""Stable fingerprints for reproducible experiment inputs and reports."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Sequence

from experiments.synthetic_negative_transfer import BenchmarkCase

SCHEMA_VERSION = "1"


def fingerprint_cases(cases: Sequence[BenchmarkCase]) -> str:
    """Return a stable SHA-256 fingerprint for an ordered benchmark case set."""

    payload = [
        {
            "case_id": case.case_id,
            "utility_with_memory": case.utility_with_memory,
            "utility_without_memory": case.utility_without_memory,
        }
        for case in cases
    ]
    encoded = json.dumps(
        {"schema_version": SCHEMA_VERSION, "cases": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
