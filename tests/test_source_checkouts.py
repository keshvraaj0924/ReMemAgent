from __future__ import annotations

from pathlib import Path

import pytest

from experiments.runtime_provenance import CLEAN_STATE, DIRTY_STATE, UNKNOWN_VALUE
from experiments.source_checkouts import (
    SourceCheckoutProvenance,
    SourceCheckoutRequirement,
    collect_source_checkout_provenance,
    validate_source_checkout_requirements,
)


def test_collect_source_checkout_provenance_records_revision_and_tree_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import source_checkouts

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    outputs = iter(("abc123\n", "", "def456\n", " M benchmark.py\n"))
    monkeypatch.setattr(
        source_checkouts.subprocess,
        "run",
        lambda *args, **kwargs: Result(next(outputs)),
    )

    provenance = collect_source_checkout_provenance(
        {
            "WebShop": Path("/tmp/webshop"),
            "ALFWorld": Path("/tmp/alfworld"),
        }
    )

    assert tuple(provenance) == ("ALFWorld", "WebShop")
    assert provenance["WebShop"] == SourceCheckoutProvenance("abc123", CLEAN_STATE)
    assert provenance["ALFWorld"] == SourceCheckoutProvenance("def456", DIRTY_STATE)


def test_collect_source_checkout_provenance_returns_unknown_on_git_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import source_checkouts

    def fail(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(source_checkouts.subprocess, "run", fail)

    provenance = collect_source_checkout_provenance({"WebShop": Path("/missing")})

    assert provenance["WebShop"] == SourceCheckoutProvenance(UNKNOWN_VALUE, UNKNOWN_VALUE)


def test_collect_source_checkout_provenance_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import source_checkouts

    class Result:
        stdout = "abc123\n"

    monkeypatch.setattr(source_checkouts.subprocess, "run", lambda *args, **kwargs: Result())
    provenance = collect_source_checkout_provenance({"WebShop": Path("/tmp/webshop")})

    with pytest.raises(TypeError):
        provenance["WebShop"] = SourceCheckoutProvenance("changed", CLEAN_STATE)  # type: ignore[index]


def test_validate_source_checkout_requirements_accepts_exact_clean_revision() -> None:
    provenance = {
        "WebShop": SourceCheckoutProvenance("abc123", CLEAN_STATE),
    }
    requirements = {
        "webshop": SourceCheckoutRequirement("abc123", require_clean_working_tree=True),
    }

    validate_source_checkout_requirements(provenance, requirements)


def test_validate_source_checkout_requirements_rejects_revision_drift() -> None:
    provenance = {
        "WebShop": SourceCheckoutProvenance("actual", CLEAN_STATE),
    }
    requirements = {
        "WebShop": SourceCheckoutRequirement("expected"),
    }

    with pytest.raises(ValueError, match="revision mismatch"):
        validate_source_checkout_requirements(provenance, requirements)


def test_validate_source_checkout_requirements_rejects_dirty_required_checkout() -> None:
    provenance = {
        "WebShop": SourceCheckoutProvenance("abc123", DIRTY_STATE),
    }
    requirements = {
        "WebShop": SourceCheckoutRequirement("abc123", require_clean_working_tree=True),
    }

    with pytest.raises(ValueError, match="must be clean"):
        validate_source_checkout_requirements(provenance, requirements)


def test_validate_source_checkout_requirements_can_allow_dirty_checkout() -> None:
    provenance = {
        "WebShop": SourceCheckoutProvenance("abc123", DIRTY_STATE),
    }
    requirements = {
        "WebShop": SourceCheckoutRequirement("abc123", require_clean_working_tree=False),
    }

    validate_source_checkout_requirements(provenance, requirements)


def test_validate_source_checkout_requirements_rejects_missing_checkout() -> None:
    with pytest.raises(ValueError, match="was not collected"):
        validate_source_checkout_requirements(
            {},
            {"WebShop": SourceCheckoutRequirement("abc123")},
        )


def test_collect_source_checkout_provenance_rejects_duplicate_normalized_names() -> None:
    with pytest.raises(ValueError, match="unique after whitespace and case normalization"):
        collect_source_checkout_provenance(
            {
                "WebShop": Path("/tmp/one"),
                " webshop ": Path("/tmp/two"),
            }
        )


def test_source_checkout_requirement_rejects_invalid_clean_flag() -> None:
    with pytest.raises(TypeError, match="boolean"):
        SourceCheckoutRequirement("abc123", require_clean_working_tree=1)  # type: ignore[arg-type]


def test_source_checkout_provenance_serializes_detached_state() -> None:
    provenance = SourceCheckoutProvenance("abc123", CLEAN_STATE)

    payload = provenance.to_dict()
    payload["revision"] = "changed"

    assert provenance.revision == "abc123"
