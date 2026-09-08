"""Deterministic identities for reproducible benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from remem.benchmark import BenchmarkRunConfiguration
from remem.reproducibility import ExperimentManifest

EXPERIMENT_IDENTITY_SCHEMA_VERSION = 1
SHA256_HEX_LENGTH = 64


def build_experiment_identity(
    configuration: BenchmarkRunConfiguration,
    seeds: Sequence[int],
    runtime_provenance: Mapping[str, Any],
) -> str:
    """Return a stable SHA-256 identity for one reproducibility protocol.

    The identity binds the benchmark protocol, independent seed set, and
    runtime provenance. It is intentionally separate from the report digest:
    the report digest authenticates serialized bytes, while this identity
    identifies the execution protocol represented by those bytes.
    """

    normalized_seeds = _normalize_seeds(seeds)
    normalized_provenance = _normalize_provenance(runtime_provenance)
    manifest = ExperimentManifest(
        {
            "schema_version": EXPERIMENT_IDENTITY_SCHEMA_VERSION,
            "configuration": _configuration_payload(configuration),
            "seeds": list(normalized_seeds),
            "runtime_provenance": normalized_provenance,
        }
    )
    return manifest.sha256


def _configuration_payload(configuration: BenchmarkRunConfiguration) -> dict[str, Any]:
    """Return configuration fields that define the benchmark protocol."""

    return {
        "benchmark_name": configuration.benchmark_name,
        "episode_count": configuration.episode_count,
        "max_steps": configuration.max_steps,
        "environment_factory": configuration.environment_factory,
        "policy_factory": configuration.policy_factory,
        "success_evaluator": configuration.success_evaluator,
        "transfer_success_evaluator": configuration.transfer_success_evaluator,
        "minimum_trust": configuration.minimum_trust,
    }


def _normalize_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate and deterministically normalize an independent seed set."""

    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a sequence of integers")
    normalized = tuple(seeds)
    if not normalized:
        raise ValueError("seeds must contain at least one seed")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in normalized):
        raise TypeError("seeds must contain only integers")
    if len(normalized) != len(set(normalized)):
        raise ValueError("seeds must be unique")
    return tuple(sorted(normalized))


def _normalize_provenance(runtime_provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach JSON-compatible runtime provenance."""

    if not isinstance(runtime_provenance, Mapping):
        raise TypeError("runtime_provenance must be a mapping")
    normalized = dict(runtime_provenance)
    try:
        manifest = ExperimentManifest(normalized)
        return manifest.values
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_provenance must be JSON-compatible") from exc


def is_experiment_identity(value: str) -> bool:
    """Return whether a value is a canonical SHA-256 experiment identity."""

    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "EXPERIMENT_IDENTITY_SCHEMA_VERSION",
    "build_experiment_identity",
    "is_experiment_identity",
]
