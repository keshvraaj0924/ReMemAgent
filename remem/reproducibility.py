"""Deterministic seed derivation for reproducible ReMemAgent experiments.

The research framework avoids relying on process-global random state for experiment
identity. A single validated base seed can instead derive stable, namespaced child
seeds for environments, policies, benchmarks, and training workers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

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
            f"remem:v{_SEED_DERIVATION_VERSION}:"
            f"{self.base_seed}:{normalized_namespace}:{index}"
        ).encode()
        digest = hashlib.blake2s(seed_material, digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def create_manifest(
        self, components: Iterable[tuple[str, int]]
    ) -> ReproducibilityManifest:
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
                raise ValueError(
                    "components must not contain duplicate namespace/index pairs"
                )
            seen_components.add(identity)
            assignments.append(
                SeedAssignment(
                    namespace=normalized_namespace,
                    index=index,
                    seed=self.derive_seed(normalized_namespace, index=index),
                )
            )

        assignments.sort(key=lambda assignment: (assignment.namespace, assignment.index))
        return ReproducibilityManifest(
            base_seed=self.base_seed,
            derivation_version=_SEED_DERIVATION_VERSION,
            assignments=tuple(assignments),
        )

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
