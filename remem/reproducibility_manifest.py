"""Validated manifests for benchmark configuration reproducibility artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remem.benchmark import BenchmarkRunConfiguration
from remem.reproducibility import (
    REPRODUCIBILITY_SCHEMA_VERSION,
    benchmark_configuration_digest,
    benchmark_configuration_payload,
)


def benchmark_configuration_manifest(
    configuration: BenchmarkRunConfiguration,
) -> dict[str, Any]:
    """Build a self-contained, deterministic identity manifest for a run."""

    payload = benchmark_configuration_payload(configuration)
    return {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "configuration_digest": benchmark_configuration_digest(configuration),
        "payload": payload,
    }


def validate_benchmark_configuration_manifest(
    manifest: Mapping[str, Any],
    configuration: BenchmarkRunConfiguration,
) -> None:
    """Validate that a manifest exactly identifies the supplied configuration.

    The validator intentionally rejects unknown top-level keys and mismatched
    nested payloads. This makes persisted manifests fail closed instead of
    silently accepting artifacts produced under a different schema.
    """

    expected_manifest = benchmark_configuration_manifest(configuration)
    if set(manifest) != set(expected_manifest):
        raise ValueError("benchmark configuration manifest has an unexpected schema")
    if manifest.get("schema_version") != expected_manifest["schema_version"]:
        raise ValueError("benchmark configuration manifest schema version mismatch")
    if manifest.get("payload") != expected_manifest["payload"]:
        raise ValueError("benchmark configuration manifest payload mismatch")
    if manifest.get("configuration_digest") != expected_manifest["configuration_digest"]:
        raise ValueError("benchmark configuration manifest digest mismatch")


__all__ = [
    "benchmark_configuration_manifest",
    "validate_benchmark_configuration_manifest",
]
