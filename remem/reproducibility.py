"""Deterministic seed derivation for reproducible ReMemAgent experiments.

The research framework avoids relying on process-global random state for experiment
identity. A single validated base seed can instead derive stable, namespaced child
seeds for environments, policies, benchmarks, and training workers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

_MAX_SEED = 2**32 - 1
_SEED_DERIVATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class SeedAssignment:
    """One deterministic seed assignment recorded in an experiment manifest."""

    namespace: str
    index: int
    seed: int


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    """Serializable record of the seed inputs used by an experiment."""

    base_seed: int
    derivation_version: int
    assignments: tuple[SeedAssignment, ...]

    def to_json(self) -> str:
        """Serialize the manifest using stable ordering for artifact comparison."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> ReproducibilityManifest:
        """Deserialize and verify a reproducibility manifest artifact.

        Raises:
            TypeError: If the JSON document does not use the expected field types.
            ValueError: If JSON is invalid, required fields are missing, unknown
                fields are present, or deterministic seed verification fails.
        """
        if not isinstance(payload, str):
            raise TypeError("manifest payload must be a string")
        try:
            raw_manifest = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("manifest payload must contain valid JSON") from error
        if not isinstance(raw_manifest, dict):
            raise TypeError("manifest JSON root must be an object")

        required_fields = {"base_seed", "derivation_version", "assignments"}
        cls._validate_fields(raw_manifest, required_fields, context="manifest")
        raw_assignments = raw_manifest["assignments"]
        if not isinstance(raw_assignments, list):
            raise TypeError("manifest assignments must be a list")

        assignments = tuple(
            cls._parse_assignment(raw_assignment, position)
            for position, raw_assignment in enumerate(raw_assignments)
        )
        manifest = cls(
            base_seed=raw_manifest["base_seed"],
            derivation_version=raw_manifest["derivation_version"],
            assignments=assignments,
        )
        manifest.verify()
        return manifest

    def verify(self) -> None:
        """Verify that every recorded seed matches the declared derivation inputs.

        Raises:
            ValueError: If the derivation version is unsupported, an assignment is
                duplicated, or a recorded seed does not match deterministic
                derivation from ``base_seed``.
        """
        if isinstance(self.derivation_version, bool) or not isinstance(
            self.derivation_version, int
        ):
            raise TypeError("derivation_version must be an integer")
        if self.derivation_version != _SEED_DERIVATION_VERSION:
            raise ValueError(f"unsupported seed derivation version: {self.derivation_version}")

        config = ReproducibilityConfig(base_seed=self.base_seed)
        seen_components: set[tuple[str, int]] = set()
        for assignment in self.assignments:
            normalized_namespace = config._validate_component(
                assignment.namespace, assignment.index
            )
            identity = (normalized_namespace, assignment.index)
            if identity in seen_components:
                raise ValueError("manifest contains duplicate namespace/index assignments")
            seen_components.add(identity)

            if isinstance(assignment.seed, bool) or not isinstance(assignment.seed, int):
                raise TypeError("assignment seed must be an integer")
            if not 0 <= assignment.seed <= _MAX_SEED:
                raise ValueError(f"assignment seed must be between 0 and {_MAX_SEED}")
            expected_seed = config.derive_seed(normalized_namespace, index=assignment.index)
            if assignment.seed != expected_seed:
                raise ValueError(
                    "manifest seed mismatch for "
                    f"{normalized_namespace}[{assignment.index}]: "
                    f"expected {expected_seed}, got {assignment.seed}"
                )

    @staticmethod
    def _validate_fields(
        payload: dict[str, Any], expected_fields: set[str], *, context: str
    ) -> None:
        actual_fields = set(payload)
        missing_fields = expected_fields - actual_fields
        unknown_fields = actual_fields - expected_fields
        if missing_fields:
            raise ValueError(f"{context} is missing fields: {sorted(missing_fields)}")
        if unknown_fields:
            raise ValueError(f"{context} contains unknown fields: {sorted(unknown_fields)}")

    @classmethod
    def _parse_assignment(cls, payload: Any, position: int) -> SeedAssignment:
        if not isinstance(payload, dict):
            raise TypeError(f"assignment {position} must be an object")
        cls._validate_fields(payload, {"namespace", "index", "seed"}, context="assignment")
        return SeedAssignment(
            namespace=payload["namespace"],
            index=payload["index"],
            seed=payload["seed"],
        )


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Configuration for deterministic, namespaced experiment seeds."""

    base_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise TypeError("base_seed must be an integer")
        if not 0 <= self.base_seed <= _MAX_SEED:
            raise ValueError(f"base_seed must be between 0 and {_MAX_SEED}")

    def derive_seed(self, namespace: str, *, index: int = 0) -> int:
        """Derive a stable 32-bit seed for one experiment component."""
        normalized_namespace = self._validate_component(namespace, index)
        seed_material = (
            f"remem:v{_SEED_DERIVATION_VERSION}:{self.base_seed}:{normalized_namespace}:{index}"
        ).encode()
        digest = hashlib.blake2s(seed_material, digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def create_manifest(self, components: Iterable[tuple[str, int]]) -> ReproducibilityManifest:
        """Build a deterministic manifest for experiment RNG components.

        Duplicate component identities are rejected because silently recording the
        same RNG stream twice makes experiment configuration ambiguous.
        """
        assignments: list[SeedAssignment] = []
        seen_components: set[tuple[str, int]] = set()

        for namespace, index in components:
            normalized_namespace = self._validate_component(namespace, index)
            identity = (normalized_namespace, index)
            if identity in seen_components:
                raise ValueError("components must not contain duplicate namespace/index pairs")
            seen_components.add(identity)
            assignments.append(
                SeedAssignment(
                    namespace=normalized_namespace,
                    index=index,
                    seed=self.derive_seed(normalized_namespace, index=index),
                )
            )

        assignments.sort(key=lambda assignment: (assignment.namespace, assignment.index))
        manifest = ReproducibilityManifest(
            base_seed=self.base_seed,
            derivation_version=_SEED_DERIVATION_VERSION,
            assignments=tuple(assignments),
        )
        manifest.verify()
        return manifest

    @staticmethod
    def _validate_component(namespace: str, index: int) -> str:
        if not isinstance(namespace, str):
            raise TypeError("namespace must be a string")
        normalized_namespace = namespace.strip()
        if not normalized_namespace:
            raise ValueError("namespace must not be empty")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0:
            raise ValueError("index must be non-negative")
        return normalized_namespace


__all__ = [
    "ReproducibilityConfig",
    "ReproducibilityManifest",
    "SeedAssignment",
]
