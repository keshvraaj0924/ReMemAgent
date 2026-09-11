"""Regression coverage for canonical external source provenance fingerprints."""

from __future__ import annotations

import pytest

from experiments.runtime_provenance import CLEAN_STATE, DIRTY_STATE
from experiments.source_checkouts import (
    SOURCE_CHECKOUT_PROVENANCE_SCHEMA_VERSION,
    SOURCE_CHECKOUT_REQUIREMENTS_SCHEMA_VERSION,
    SourceCheckoutProvenance,
    SourceCheckoutRequirement,
    source_checkout_provenance_from_dict,
    source_checkout_provenance_sha256,
    source_checkout_provenance_to_dict,
    source_checkout_requirements_from_dict,
    source_checkout_requirements_sha256,
    source_checkout_requirements_to_dict,
)


def test_requirement_contract_round_trip_is_canonical_and_case_insensitive() -> None:
    first = {
        " WebShop ": SourceCheckoutRequirement("webshop-revision"),
        "ALFWorld": SourceCheckoutRequirement("alfworld-revision", False),
    }
    second = {
        "alfworld": SourceCheckoutRequirement("alfworld-revision", False),
        "webshop": SourceCheckoutRequirement("webshop-revision"),
    }

    payload = source_checkout_requirements_to_dict(first)
    restored = source_checkout_requirements_from_dict(payload)

    assert payload["schema_version"] == SOURCE_CHECKOUT_REQUIREMENTS_SCHEMA_VERSION
    assert tuple(payload["repositories"]) == ("alfworld", "webshop")
    assert restored == second
    assert source_checkout_requirements_sha256(first) == source_checkout_requirements_sha256(second)


def test_requirement_fingerprint_changes_when_revision_or_clean_policy_changes() -> None:
    baseline = {"webshop": SourceCheckoutRequirement("revision-a", True)}
    revision_drift = {"webshop": SourceCheckoutRequirement("revision-b", True)}
    policy_drift = {"webshop": SourceCheckoutRequirement("revision-a", False)}

    assert source_checkout_requirements_sha256(baseline) != source_checkout_requirements_sha256(
        revision_drift
    )
    assert source_checkout_requirements_sha256(baseline) != source_checkout_requirements_sha256(
        policy_drift
    )


def test_provenance_snapshot_round_trip_is_canonical_and_case_insensitive() -> None:
    first = {
        "WebShop": SourceCheckoutProvenance("webshop-revision", CLEAN_STATE),
        " alfworld ": SourceCheckoutProvenance("alfworld-revision", DIRTY_STATE),
    }
    second = {
        "alfworld": SourceCheckoutProvenance("alfworld-revision", DIRTY_STATE),
        "webshop": SourceCheckoutProvenance("webshop-revision", CLEAN_STATE),
    }

    payload = source_checkout_provenance_to_dict(first)
    restored = source_checkout_provenance_from_dict(payload)

    assert payload["schema_version"] == SOURCE_CHECKOUT_PROVENANCE_SCHEMA_VERSION
    assert tuple(payload["repositories"]) == ("alfworld", "webshop")
    assert restored == second
    assert source_checkout_provenance_sha256(first) == source_checkout_provenance_sha256(second)


def test_provenance_fingerprint_changes_with_observed_git_state() -> None:
    clean = {"webshop": SourceCheckoutProvenance("revision-a", CLEAN_STATE)}
    dirty = {"webshop": SourceCheckoutProvenance("revision-a", DIRTY_STATE)}
    other_revision = {"webshop": SourceCheckoutProvenance("revision-b", CLEAN_STATE)}

    assert source_checkout_provenance_sha256(clean) != source_checkout_provenance_sha256(dirty)
    assert source_checkout_provenance_sha256(clean) != source_checkout_provenance_sha256(
        other_revision
    )


def test_persisted_source_contract_rejects_unknown_schema_version() -> None:
    payload = source_checkout_requirements_to_dict(
        {"webshop": SourceCheckoutRequirement("revision-a")}
    )
    payload["schema_version"] = SOURCE_CHECKOUT_REQUIREMENTS_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="unsupported source checkout requirements schema version"):
        source_checkout_requirements_from_dict(payload)


def test_persisted_source_snapshot_requires_non_empty_repository_mapping() -> None:
    with pytest.raises(ValueError, match="repositories must not be empty"):
        source_checkout_provenance_from_dict(
            {
                "schema_version": SOURCE_CHECKOUT_PROVENANCE_SCHEMA_VERSION,
                "repositories": {},
            }
        )
