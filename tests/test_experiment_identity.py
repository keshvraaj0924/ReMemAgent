from __future__ import annotations

import pytest

from experiments.experiment_identity import (
    build_experiment_identity,
    is_experiment_identity,
    verify_experiment_identity,
)
from remem.benchmark import BenchmarkRunConfiguration
from remem.reproducibility import ExperimentManifest


def _configuration() -> BenchmarkRunConfiguration:
    return BenchmarkRunConfiguration(
        benchmark_name="alfworld-eval",
        episode_count=10,
        max_steps=20,
        seed=11,
        environment_factory="adapter:make_environment",
        policy_factory="policy:make_policy",
        success_evaluator="metrics:is_success",
        transfer_success_evaluator="metrics:is_transfer_success",
        minimum_trust=0.7,
    )


def test_identity_is_stable_for_seed_order_and_mapping_order() -> None:
    first = build_experiment_identity(
        _configuration(),
        [41, 7, 19],
        {"code_revision": "abc", "dependency_versions": {"z": "2", "a": "1"}},
    )
    second = build_experiment_identity(
        _configuration(),
        [19, 41, 7],
        {"dependency_versions": {"a": "1", "z": "2"}, "code_revision": "abc"},
    )

    assert first == second
    assert is_experiment_identity(first)


def test_identity_matches_shared_canonical_manifest_digest() -> None:
    provenance = {"code_revision": "abc", "nested": {"value": 3}}
    identity = build_experiment_identity(_configuration(), [7, 19], provenance)
    expected = ExperimentManifest(
        {
            "schema_version": 1,
            "configuration": {
                "benchmark_name": "alfworld-eval",
                "episode_count": 10,
                "max_steps": 20,
                "seed": 11,
                "environment_factory": "adapter:make_environment",
                "policy_factory": "policy:make_policy",
                "success_evaluator": "metrics:is_success",
                "transfer_success_evaluator": "metrics:is_transfer_success",
                "minimum_trust": 0.7,
            },
            "seeds": [7, 19],
            "runtime_provenance": provenance,
        }
    ).sha256

    assert identity == expected


def test_identity_changes_when_protocol_changes() -> None:
    baseline = build_experiment_identity(_configuration(), [7, 19], {"code_revision": "abc"})
    changed = build_experiment_identity(
        BenchmarkRunConfiguration(
            benchmark_name="alfworld-eval",
            episode_count=11,
            max_steps=20,
            seed=11,
            environment_factory="adapter:make_environment",
            policy_factory="policy:make_policy",
            success_evaluator="metrics:is_success",
            transfer_success_evaluator="metrics:is_transfer_success",
            minimum_trust=0.7,
        ),
        [7, 19],
        {"code_revision": "abc"},
    )

    assert baseline != changed


def test_identity_changes_when_configuration_seed_changes() -> None:
    baseline = build_experiment_identity(_configuration(), [7, 19], {"code_revision": "abc"})
    changed_configuration = BenchmarkRunConfiguration(
        benchmark_name="alfworld-eval",
        episode_count=10,
        max_steps=20,
        seed=12,
        environment_factory="adapter:make_environment",
        policy_factory="policy:make_policy",
        success_evaluator="metrics:is_success",
        transfer_success_evaluator="metrics:is_transfer_success",
        minimum_trust=0.7,
    )

    assert baseline != build_experiment_identity(
        changed_configuration,
        [7, 19],
        {"code_revision": "abc"},
    )


def test_identity_changes_when_runtime_provenance_changes() -> None:
    baseline = build_experiment_identity(_configuration(), [7, 19], {"code_revision": "abc"})
    changed = build_experiment_identity(_configuration(), [7, 19], {"code_revision": "def"})

    assert baseline != changed


def test_identity_rejects_duplicate_or_missing_seeds() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_experiment_identity(_configuration(), [], {})
    with pytest.raises(ValueError, match="unique"):
        build_experiment_identity(_configuration(), [7, 7], {})


def test_identity_rejects_non_json_runtime_provenance() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        build_experiment_identity(_configuration(), [7], {"object": object()})


def test_identity_validator_rejects_noncanonical_values() -> None:
    assert not is_experiment_identity("A" * 64)
    assert not is_experiment_identity("0" * 63)
    assert not is_experiment_identity("0" * 65)
    assert not is_experiment_identity(42)  # type: ignore[arg-type]


def test_identity_verification_accepts_matching_protocol() -> None:
    provenance = {"code_revision": "abc", "dependency_versions": {"python": "3.12"}}
    identity = build_experiment_identity(_configuration(), [7, 19], provenance)

    verify_experiment_identity(identity, _configuration(), [19, 7], provenance)


def test_identity_verification_rejects_stale_identity() -> None:
    provenance = {"code_revision": "abc"}
    identity = build_experiment_identity(_configuration(), [7, 19], provenance)

    with pytest.raises(ValueError, match="does not match"):
        verify_experiment_identity(identity, _configuration(), [7, 20], provenance)


def test_identity_verification_rejects_changed_runtime_provenance() -> None:
    provenance = {"code_revision": "abc"}
    identity = build_experiment_identity(_configuration(), [7, 19], provenance)

    with pytest.raises(ValueError, match="does not match"):
        verify_experiment_identity(
            identity,
            _configuration(),
            [7, 19],
            {"code_revision": "def"},
        )


def test_identity_verification_rejects_changed_configuration() -> None:
    identity = build_experiment_identity(_configuration(), [7, 19], {})
    changed_configuration = BenchmarkRunConfiguration(
        benchmark_name="alfworld-eval",
        episode_count=11,
        max_steps=20,
        seed=11,
        environment_factory="adapter:make_environment",
        policy_factory="policy:make_policy",
        success_evaluator="metrics:is_success",
        transfer_success_evaluator="metrics:is_transfer_success",
        minimum_trust=0.7,
    )

    with pytest.raises(ValueError, match="does not match"):
        verify_experiment_identity(identity, changed_configuration, [7, 19], {})


def test_identity_verification_rejects_malformed_identity() -> None:
    with pytest.raises(ValueError, match="canonical SHA-256"):
        verify_experiment_identity("0" * 63, _configuration(), [7], {})
