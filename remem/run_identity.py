"""Stable experiment identities derived from verified reproducibility manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from remem.reproducibility import ReproducibilityManifest

_RUN_ID_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentRunIdentity:
    """Content-addressed identity for one verified experiment seed configuration."""

    version: int
    digest: str

    @classmethod
    def from_manifest(cls, manifest: ReproducibilityManifest) -> ExperimentRunIdentity:
        """Build a stable identity after verifying the manifest's seed assignments."""
        if not isinstance(manifest, ReproducibilityManifest):
            raise TypeError("manifest must be a ReproducibilityManifest")
        manifest.verify()
        material = f"remem-run:v{_RUN_ID_VERSION}:{manifest.to_json()}".encode()
        digest = hashlib.sha256(material).hexdigest()
        return cls(version=_RUN_ID_VERSION, digest=digest)

    @property
    def value(self) -> str:
        """Return the portable versioned run identifier used in artifact metadata."""
        return f"v{self.version}-{self.digest}"


__all__ = ["ExperimentRunIdentity"]
