from __future__ import annotations

from pathlib import Path

import pytest

from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutRequirement
from experiments.strict_reproducibility import validate_strict_paired_reproducibility


def _runtime_requirements(**overrides: object) -> RuntimeRequirements:
    values: dict[str, object] = {
        "expected_code_revision": "abc123",
        "require_clean_working_tree": True,
        "dependency_versions": {"torch": "2.6.0"},
    }
    values.update(overrides)
    return RuntimeRequirements(**values)  # type: ignore[arg-type]


def _source_requirements(*, require_clean: bool = True) -> dict[str, SourceCheckoutRequirement]:
    return {
        "webshop": SourceCheckoutRequirement(
            expected_revision="def456",
            require_clean_working_tree=require_clean,
        )
    }


def test_strict_reproducibility_accepts_complete_contract(tmp_path: Path) -> None:
    validate_strict_paired_reproducibility(
        seeds=(11, 17, 29),
        runtime_requirements=_runtime_requirements(),
        source_checkout_requirements=_source_requirements(),
        manifest_path=tmp_path / "paired.json.manifest.json",
    )


@pytest.mark.parametrize("seeds", [(), (11,)])
def test_strict_reproducibility_requires_multiple_independent_seeds(
    seeds: tuple[int, ...], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="at least 2 independent seeds"):
        validate_strict_paired_reproducibility(
            seeds=seeds,
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_runtime_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires runtime requirements"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=None,
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_exact_code_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact code revision"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(expected_code_revision=None),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_clean_framework_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="clean ReMemAgent working tree"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(require_clean_working_tree=False),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_exact_dependency_pin(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one exact dependency version"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(dependency_versions={}),
            source_checkout_requirements=_source_requirements(),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_source_checkout_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one source checkout requirement"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=None,
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_rejects_dirty_source_exception(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dirty checkout exceptions.*webshop"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=_source_requirements(require_clean=False),
            manifest_path=tmp_path / "manifest.json",
        )


def test_strict_reproducibility_requires_integrity_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integrity manifest destination"):
        validate_strict_paired_reproducibility(
            seeds=(11, 17),
            runtime_requirements=_runtime_requirements(),
            source_checkout_requirements=_source_requirements(),
            manifest_path=None,
        )
