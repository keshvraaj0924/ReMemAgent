"""Deterministic identities for benchmark reproducibility artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from remem.benchmark import BenchmarkRunConfiguration


def benchmark_configuration_digest(configuration: BenchmarkRunConfiguration) -> str:
    """Return a stable SHA-256 identity for one benchmark configuration.

    The digest is computed from the validated configuration fields using
    canonical JSON. It is intended for artifact names, manifests, and audit
    records; it does not claim that two runs with the same configuration are
    scientifically equivalent.
    """

    payload = json.dumps(
        asdict(configuration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["benchmark_configuration_digest"]
