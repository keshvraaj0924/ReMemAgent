"""Deterministic experiment manifests for reproducible research artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from typing import Any


class ExperimentManifest:
    """Canonical, hashable description of an experiment configuration.

    The manifest intentionally stores configuration data rather than runtime
    objects. Callers can persist the canonical JSON and SHA-256 digest next to
    benchmark results to detect accidental configuration drift.
    """

    def __init__(self, values: Mapping[str, Any]) -> None:
        """Validate and retain a detached copy of JSON-compatible values."""

        if not isinstance(values, Mapping):
            raise TypeError("manifest values must be a mapping")
        self._values = _normalize_mapping(values)

    @classmethod
    def from_json(cls, payload: str) -> "ExperimentManifest":
        """Load and validate a manifest from JSON without trusting its formatting."""

        if not isinstance(payload, str):
            raise TypeError("manifest JSON must be a string")
        try:
            values = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("manifest JSON is invalid") from exc
        if not isinstance(values, Mapping):
            raise ValueError("manifest JSON must contain an object")
        return cls(values)

    @property
    def values(self) -> dict[str, Any]:
        """Return a detached canonical manifest mapping."""

        return json.loads(self.to_json())

    def to_json(self) -> str:
        """Serialize the manifest with stable ordering and separators."""

        return json.dumps(
            self._values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        """Return the SHA-256 digest of the canonical JSON representation."""

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def verify_sha256(self, expected_sha256: str) -> None:
        """Raise when an expected SHA-256 digest is malformed or does not match."""

        if not isinstance(expected_sha256, str):
            raise TypeError("expected_sha256 must be a string")
        normalized_digest = expected_sha256.strip().lower()
        if len(normalized_digest) != hashlib.sha256().digest_size * 2:
            raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
        if any(character not in "0123456789abcdef" for character in normalized_digest):
            raise ValueError("expected_sha256 must contain only hexadecimal characters")
        if not hmac.compare_digest(self.sha256, normalized_digest):
            raise ValueError(
                "manifest SHA-256 mismatch: "
                f"expected {expected_sha256}, computed {self.sha256}"
            )


def _normalize_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively validate and normalize JSON-compatible manifest data."""

    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("manifest keys must be non-empty strings")
        normalized[key] = _normalize_value(value)
    return normalized


def _normalize_value(value: Any) -> Any:
    """Validate one manifest value and return an immutable-safe copy."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("manifest floats must be finite")
        return value
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_value(item) for item in value]
    raise TypeError(
        "manifest values must contain only JSON-compatible scalars, mappings, or sequences"
    )
