"""Deterministic seed derivation for reproducible ReMemAgent experiments.

The research framework avoids relying on process-global random state for experiment
identity. A single validated base seed can instead derive stable, namespaced child
seeds for environments, policies, benchmarks, and training workers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_MAX_SEED = 2**32 - 1


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
        """Derive a stable 32-bit seed for one experiment component.

        Args:
            namespace: Stable component name such as ``benchmark`` or ``policy``.
            index: Non-negative replica or worker index within the namespace.

        Returns:
            A deterministic unsigned 32-bit integer suitable for common RNG APIs.
        """
        normalized_namespace = namespace.strip()
        if not normalized_namespace:
            raise ValueError("namespace must not be empty")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0:
            raise ValueError("index must be non-negative")

        seed_material = f"remem:v1:{self.base_seed}:{normalized_namespace}:{index}".encode()
        digest = hashlib.blake2s(seed_material, digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)


__all__ = ["ReproducibilityConfig"]
