from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from experiments.benchmark_report import paired_configuration_fingerprint
from experiments.experiment_identity import (
    build_paired_experiment_identity,
    verify_paired_experiment_identity,
)
from experiments.paired_artifacts import validate_persisted_paired_artifact
from remem.benchmark import BenchmarkRunConfiguration


def _configuration(policy_factory: str) -> BenchmarkRunConfiguration:
    return BenchmarkRunConfiguration(
        benchmark_name="alfworld-eval",
        episode_count=4,
        max_steps=20,
        seed=11,
        environment_factory="adapter:make_environment",
        policy_factory=policy_factory,
        success_evaluator="metrics:is_success",
        transfer_success_evaluator="metrics:is_transfer_success",
        minimum_trust=0.7,
    )


def _paired_payload() -> dict[str, object]:
    baseline = _configuration("policy:make_baseline")
    treatment = _configuration("policy:make_memory")
    seeds = [11, 17]
    runtime_provenance = {"code_revision": "abc123"}
    return {
        "seeds": seeds,
        "baseline": {"reports": [{"configuration": asdict(baseline)}]},
        "treatment": {"reports": [{"configuration": asdict(treatment)}]},
        "runtime_provenance": runtime_provenance,
        "configuration_fingerprint": paired_configuration_fingerprint(baseline, treatment),
        "experiment_identity": build_paired_experiment_identity(
            replace(baseline, seed=None),
            replace(treatment, seed=None),
            seeds,
            runtime_provenance,
        ),
    }


def test_verify_paired_experiment_identity_accepts_matching_protocol() -> None:
    baseline = _configuration("policy:make_baseline")
    treatment = _configuration("policy:make_memory")
    provenance = {"code_revision": "abc123"}
    identity = build_paired_experiment_identity(
        replace(baseline, seed=None),
        replace(treatment, seed=None),
        [11, 17],
        provenance,
    )

    verify_paired_experiment_identity(
        identity,
        replace(baseline, seed=None),
        replace(treatment, seed=None),
        [17, 11],
        provenance,
    )


def test_verify_paired_experiment_identity_rejects_changed_treatment() -> None:
    baseline = _configuration("policy:make_baseline")
    treatment = _configuration("policy:make_memory")
    identity = build_paired_experiment_identity(
        replace(baseline, seed=None),
        replace(treatment, seed=None),
        [11, 17],
        {},
    )
    changed_treatment = _configuration("policy:make_other_memory")

    with pytest.raises(ValueError, match="paired experiment identity does not match"):
        verify_paired_experiment_identity(
            identity,
            replace(baseline, seed=None),
            replace(changed_treatment, seed=None),
            [11, 17],
            {},
        )


def test_validate_persisted_paired_artifact_accepts_matching_identity() -> None:
    validate_persisted_paired_artifact(_paired_payload())


def test_validate_persisted_paired_artifact_rejects_stale_identity() -> None:
    payload = _paired_payload()
    payload["experiment_identity"] = "0" * 64

    with pytest.raises(ValueError, match="paired experiment identity does not match"):
        validate_persisted_paired_artifact(payload)


def test_validate_persisted_paired_artifact_rejects_stale_configuration_fingerprint() -> None:
    payload = _paired_payload()
    payload["configuration_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="configuration fingerprint does not match"):
        validate_persisted_paired_artifact(payload)


def test_validate_persisted_paired_artifact_allows_legacy_identityless_pair() -> None:
    payload = _paired_payload()
    payload.pop("experiment_identity")
    payload.pop("configuration_fingerprint")

    validate_persisted_paired_artifact(payload)
