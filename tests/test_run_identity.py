"""Tests for deterministic experiment run identities."""

import pytest

from remem.reproducibility import ReproducibilityConfig, ReproducibilityManifest, SeedAssignment
from remem.run_identity import ExperimentRunIdentity


def test_run_identity_is_stable_for_equivalent_manifests() -> None:
    config = ReproducibilityConfig(base_seed=42)
    first = config.create_manifest([("policy", 1), ("environment", 0)])
    second = config.create_manifest([("environment", 0), ("policy", 1)])
    first_identity = ExperimentRunIdentity.from_manifest(first)
    second_identity = ExperimentRunIdentity.from_manifest(second)
    assert first_identity == second_identity
    assert first_identity.value.startswith("v1-")
    assert len(first_identity.digest) == 64


def test_run_identity_changes_with_seed_configuration() -> None:
    first = ReproducibilityConfig(base_seed=42).create_manifest([("policy", 0)])
    second = ReproducibilityConfig(base_seed=43).create_manifest([("policy", 0)])
    assert ExperimentRunIdentity.from_manifest(first) != ExperimentRunIdentity.from_manifest(second)


def test_run_identity_rejects_tampered_manifest() -> None:
    manifest = ReproducibilityManifest(
        base_seed=42,
        derivation_version=1,
        assignments=(SeedAssignment(namespace="policy", index=0, seed=123),),
    )
    with pytest.raises(ValueError, match="manifest seed mismatch"):
        ExperimentRunIdentity.from_manifest(manifest)
