"""Deterministic identities for benchmark reproducibility artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

from remem.benchmark import BenchmarkRunConfiguration

REPRODUCIBILITY_SCHEMA_VERSION = 1


def benchmark_configuration_payload(
    configuration: BenchmarkRunConfiguration,
) -> dict[str, Any]:
    """Return the versioned canonical payload used for configuration identity."""

    return {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "configuration": asdict(configuration),
    }


def benchmark_configuration_digest(configuration: BenchmarkRunConfiguration) -> str:
    """Return a stable SHA-256 identity for one benchmark configuration.

    The digest is computed from a versioned canonical JSON envelope around the
    validated configuration fields. Versioning prevents future changes to the
    identity algorithm from silently creating ambiguous artifact identities.
    The digest is intended for artifact names, manifests, and audit records; it
    does not claim that two runs with the same configuration are scientifically
    equivalent.
    """

    payload = json.dumps(
        benchmark_configuration_payload(configuration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "REPRODUCIBILITY_SCHEMA_VERSION",
    "benchmark_configuration_digest",
    "benchmark_configuration_payload",
]
