"""Deterministic identities for reproducible benchmark artifacts."""

from __future__ import annotations

import hmac
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

    The identity binds the complete benchmark protocol, independent seed set,
    and runtime provenance. It is intentionally separate from the report digest:
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


def build_paired_experiment_identity(
    baseline_configuration: BenchmarkRunConfiguration,
    treatment_configuration: BenchmarkRunConfiguration,
    seeds: Sequence[int],
    runtime_provenance: Mapping[str, Any],
) -> str:
    """Return a stable identity binding both conditions of a paired protocol."""

    normalized_seeds = _normalize_seeds(seeds)
    normalized_provenance = _normalize_provenance(runtime_provenance)
    manifest = ExperimentManifest(
        {
            "schema_version": EXPERIMENT_IDENTITY_SCHEMA_VERSION,
            "baseline_configuration": _configuration_payload(baseline_configuration),
            "treatment_configuration": _configuration_payload(treatment_configuration),
            "seeds": list(normalized_seeds),
            "runtime_provenance": normalized_provenance,
        }
    )
    return manifest.sha256


def verify_experiment_identity(
    identity: str,
    configuration: BenchmarkRunConfiguration,
    seeds: Sequence[int],
    runtime_provenance: Mapping[str, Any],
) -> None:
    """Fail if a persisted experiment identity does not match its inputs."""

    _validate_identity(identity)
    expected_identity = build_experiment_identity(configuration, seeds, runtime_provenance)
    if not hmac.compare_digest(identity, expected_identity):
        raise ValueError("experiment identity does not match the supplied protocol metadata")


def verify_paired_experiment_identity(
    identity: str,
    baseline_configuration: BenchmarkRunConfiguration,
    treatment_configuration: BenchmarkRunConfiguration,
    seeds: Sequence[int],
    runtime_provenance: Mapping[str, Any],
) -> None:
    """Fail if a persisted paired identity does not match both condition protocols."""

    _validate_identity(identity)
    expected_identity = build_paired_experiment_identity(
        baseline_configuration,
        treatment_configuration,
        seeds,
        runtime_provenance,
    )
    if not hmac.compare_digest(identity, expected_identity):
        raise ValueError("paired experiment identity does not match the supplied protocol metadata")


def _validate_identity(identity: str) -> None:
    """Require the canonical lowercase SHA-256 representation used by artifacts."""

    if not is_experiment_identity(identity):
        raise ValueError("identity must be a canonical SHA-256 experiment identity")


def _configuration_payload(configuration: BenchmarkRunConfiguration) -> dict[str, Any]:
    """Return every configuration field that defines the benchmark protocol."""

    return {
        "benchmark_name": configuration.benchmark_name,
        "episode_count": configuration.episode_count,
        "max_steps": configuration.max_steps,
        "seed": configuration.seed,
        "environment_factory": configuration.environment_factory,
        "policy_factory": configuration.policy_factory,
        "action_policy_factory": configuration.action_policy_factory,
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
    "build_paired_experiment_identity",
    "is_experiment_identity",
    "verify_experiment_identity",
    "verify_paired_experiment_identity",
]
