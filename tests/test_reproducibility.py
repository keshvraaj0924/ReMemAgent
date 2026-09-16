"""Tests for deterministic experiment seed derivation."""

import pytest

from remem.reproducibility import ReproducibilityConfig


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


def test_derive_seed_rejects_invalid_replica_index() -> None:
    config = ReproducibilityConfig(base_seed=42)

    with pytest.raises(ValueError, match="index must be non-negative"):
        config.derive_seed("benchmark", index=-1)

    with pytest.raises(TypeError, match="index must be an integer"):
        config.derive_seed("benchmark", index=True)
