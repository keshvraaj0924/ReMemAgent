"""Tests for deterministic experiment seed derivation."""

import json
import random
from pathlib import Path

import pytest

from remem.reproducibility import (
    ReproducibilityConfig,
    ReproducibilityManifest,
    SeedAssignment,
)


def test_derive_seed_is_stable_for_same_component() -> None:
    config = ReproducibilityConfig(base_seed=42)
    first = config.derive_seed("benchmark", index=3)
    second = config.derive_seed("benchmark", index=3)
    assert first == second
    assert 0 <= first <= 2**32 - 1


def test_derive_seed_separates_namespaces_and_replicas() -> None:
    config = ReproducibilityConfig(base_seed=42)
    seeds = {
        config.derive_seed("benchmark", index=0),
        config.derive_seed("benchmark", index=1),
        config.derive_seed("policy", index=0),
    }
    assert len(seeds) == 3


def test_derive_seed_changes_with_base_seed() -> None:
    first = ReproducibilityConfig(base_seed=1).derive_seed("benchmark")
    second = ReproducibilityConfig(base_seed=2).derive_seed("benchmark")
    assert first != second


def test_create_random_replays_component_sequence() -> None:
    config = ReproducibilityConfig(base_seed=42)
    first = config.create_random("benchmark", index=3)
    second = config.create_random("benchmark", index=3)
    assert [first.random() for _ in range(5)] == [second.random() for _ in range(5)]


def test_create_random_isolates_component_streams() -> None:
    config = ReproducibilityConfig(base_seed=42)
    benchmark = config.create_random("benchmark")
    policy = config.create_random("policy")
    assert [benchmark.random() for _ in range(5)] != [policy.random() for _ in range(5)]


def test_create_random_does_not_mutate_global_random_state() -> None:
    random.seed(8675309)
    expected = [random.random() for _ in range(3)]
    random.seed(8675309)
    config = ReproducibilityConfig(base_seed=42)
    local_random = config.create_random("benchmark")
    _ = [local_random.random() for _ in range(10)]
    assert [random.random() for _ in range(3)] == expected


def test_manifest_is_order_independent_and_serializable() -> None:
    config = ReproducibilityConfig(base_seed=42)
    first = config.create_manifest([("policy", 1), ("environment", 0)])
    second = config.create_manifest([("environment", 0), ("policy", 1)])
    assert first == second
    first.verify()
    payload = json.loads(first.to_json())
    assert payload["base_seed"] == 42
    assert payload["derivation_version"] == 1
    assert [item["namespace"] for item in payload["assignments"]] == ["environment", "policy"]
    assert payload["assignments"][0]["seed"] == config.derive_seed("environment")


def test_manifest_json_round_trip_verifies_artifact() -> None:
    manifest = ReproducibilityConfig(base_seed=42).create_manifest(
        [("environment", 0), ("policy", 2)]
    )
    loaded = ReproducibilityManifest.from_json(manifest.to_json())
    assert loaded == manifest


def test_manifest_save_and_load_round_trip(tmp_path: Path) -> None:
    manifest = ReproducibilityConfig(base_seed=42).create_manifest(
        [("environment", 0), ("policy", 2)]
    )
    destination = tmp_path / "run" / "reproducibility.json"
    destination.parent.mkdir()
    saved_path = manifest.save(destination)
    assert saved_path == destination
    assert destination.read_text(encoding="utf-8") == f"{manifest.to_json()}\n"
    assert ReproducibilityManifest.load(destination) == manifest
    assert not list(destination.parent.glob("*.tmp"))


def test_manifest_save_verifies_before_replacing_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "reproducibility.json"
    destination.write_text("existing artifact\n", encoding="utf-8")
    invalid_manifest = ReproducibilityManifest(
        base_seed=42,
        derivation_version=1,
        assignments=(SeedAssignment(namespace="policy", index=0, seed=123),),
    )
    with pytest.raises(ValueError, match="manifest seed mismatch"):
        invalid_manifest.save(destination)
    assert destination.read_text(encoding="utf-8") == "existing artifact\n"


def test_manifest_save_requires_existing_parent_directory(tmp_path: Path) -> None:
    manifest = ReproducibilityConfig(base_seed=42).create_manifest([])
    destination = tmp_path / "missing" / "reproducibility.json"
    with pytest.raises(FileNotFoundError, match="parent directory does not exist"):
        manifest.save(destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ("not-json", ValueError, "valid JSON"),
        ("[]", TypeError, "root must be an object"),
        ('{"base_seed":42}', ValueError, "missing fields"),
        (
            '{"base_seed":42,"derivation_version":1,"assignments":[],"extra":1}',
            ValueError,
            "unknown fields",
        ),
        (
            '{"base_seed":42,"derivation_version":1,"assignments":{}}',
            TypeError,
            "assignments must be a list",
        ),
    ],
)
def test_manifest_loading_rejects_invalid_schema(
    payload: str, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        ReproducibilityManifest.from_json(payload)


def test_manifest_loading_rejects_tampered_artifact() -> None:
    manifest = ReproducibilityConfig(base_seed=42).create_manifest([("policy", 0)])
    payload = json.loads(manifest.to_json())
    payload["assignments"][0]["seed"] = 123
    with pytest.raises(ValueError, match="manifest seed mismatch"):
        ReproducibilityManifest.from_json(json.dumps(payload))


def test_manifest_normalizes_component_names() -> None:
    config = ReproducibilityConfig(base_seed=42)
    manifest = config.create_manifest([(" benchmark ", 0)])
    assert manifest.assignments[0].namespace == "benchmark"
    assert manifest.assignments[0].seed == config.derive_seed("benchmark")


def test_manifest_rejects_duplicate_component_identity() -> None:
    config = ReproducibilityConfig(base_seed=42)
    with pytest.raises(ValueError, match="duplicate namespace/index pairs"):
        config.create_manifest([("policy", 0), (" policy ", 0)])


def test_manifest_verification_rejects_tampered_seed() -> None:
    manifest = ReproducibilityManifest(
        base_seed=42,
        derivation_version=1,
        assignments=(SeedAssignment(namespace="policy", index=0, seed=123),),
    )
    with pytest.raises(ValueError, match="manifest seed mismatch"):
        manifest.verify()


def test_manifest_verification_rejects_unsupported_derivation_version() -> None:
    manifest = ReproducibilityManifest(base_seed=42, derivation_version=999, assignments=())
    with pytest.raises(ValueError, match="unsupported seed derivation version"):
        manifest.verify()


def test_manifest_verification_rejects_duplicate_assignments() -> None:
    config = ReproducibilityConfig(base_seed=42)
    seed = config.derive_seed("policy")
    manifest = ReproducibilityManifest(
        base_seed=42,
        derivation_version=1,
        assignments=(
            SeedAssignment(namespace="policy", index=0, seed=seed),
            SeedAssignment(namespace=" policy ", index=0, seed=seed),
        ),
    )
    with pytest.raises(ValueError, match="duplicate namespace/index assignments"):
        manifest.verify()


@pytest.mark.parametrize("base_seed", [-1, 2**32])
def test_config_rejects_out_of_range_base_seed(base_seed: int) -> None:
    with pytest.raises(ValueError, match="base_seed must be between"):
        ReproducibilityConfig(base_seed=base_seed)


def test_config_rejects_boolean_base_seed() -> None:
    with pytest.raises(TypeError, match="base_seed must be an integer"):
        ReproducibilityConfig(base_seed=True)


@pytest.mark.parametrize("namespace", ["", "   "])
def test_derive_seed_rejects_empty_namespace(namespace: str) -> None:
    config = ReproducibilityConfig(base_seed=42)
    with pytest.raises(ValueError, match="namespace must not be empty"):
        config.derive_seed(namespace)


def test_derive_seed_rejects_non_string_namespace() -> None:
    config = ReproducibilityConfig(base_seed=42)
    with pytest.raises(TypeError, match="namespace must be a string"):
        config.derive_seed(42)  # type: ignore[arg-type]


def test_derive_seed_rejects_invalid_replica_index() -> None:
    config = ReproducibilityConfig(base_seed=42)
    with pytest.raises(ValueError, match="index must be non-negative"):
        config.derive_seed("benchmark", index=-1)
    with pytest.raises(TypeError, match="index must be an integer"):
        config.derive_seed("benchmark", index=True)
