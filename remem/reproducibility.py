"""Deterministic identities and canonical manifests for reproducibility artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import hmac
import json
import math
from typing import Any

from remem.benchmark import BenchmarkRunConfiguration

REPRODUCIBILITY_SCHEMA_VERSION = 1
SHA256_HEX_LENGTH = 64


class ExperimentManifest:
    """Canonical, detached JSON manifest with deterministic SHA-256 identity."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("manifest values must be a mapping")
        normalized = _normalize_json_mapping(values)
        self._canonical_json = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def values(self) -> dict[str, Any]:
        """Return a detached copy of the normalized manifest values."""

        value = json.loads(self._canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor guarantees this invariant
            raise RuntimeError("canonical manifest root is not an object")
        return value

    @property
    def sha256(self) -> str:
        """Return the SHA-256 digest of the canonical JSON representation."""

        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Return deterministic compact JSON for persistence and hashing."""

        return self._canonical_json

    @classmethod
    def from_json(cls, payload: str) -> ExperimentManifest:
        """Parse JSON and normalize it into the canonical manifest contract."""

        if not isinstance(payload, str):
            raise TypeError("manifest JSON payload must be a string")
        try:
            values = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid manifest JSON payload") from exc
        if not isinstance(values, Mapping):
            raise ValueError("manifest JSON payload must contain an object")
        return cls(values)

    def verify_sha256(self, digest: str) -> None:
        """Fail closed when ``digest`` does not identify this canonical manifest."""

        if not isinstance(digest, str):
            raise TypeError("SHA-256 digest must be a string")
        if len(digest) != SHA256_HEX_LENGTH:
            raise ValueError("SHA-256 digest must contain exactly 64 hexadecimal characters")
        if any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError("SHA-256 digest must contain only hexadecimal characters")
        if not hmac.compare_digest(digest.lower(), self.sha256):
            raise ValueError("manifest SHA-256 mismatch")


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
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_json_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and recursively detach a JSON-compatible mapping."""

    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("manifest object keys must be non-empty strings")
        normalized[key] = _normalize_json_value(value)
    return normalized


def _normalize_json_value(value: Any) -> Any:
    """Normalize one manifest value while rejecting ambiguous JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest floating-point values must be finite")
        return value
    if isinstance(value, Mapping):
        return _normalize_json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    raise TypeError("manifest values must be JSON-compatible")


__all__ = [
    "ExperimentManifest",
    "REPRODUCIBILITY_SCHEMA_VERSION",
    "benchmark_configuration_digest",
    "benchmark_configuration_payload",
]
