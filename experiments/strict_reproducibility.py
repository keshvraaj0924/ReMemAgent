"""Strict reproducibility admission for publishable paired benchmark runs.

The normal benchmark CLIs intentionally remain usable for development and smoke
experiments.  This module adds an opt-in fail-closed contract for experiments
that are intended to become scientific evidence: multiple independent seeds,
an exact clean ReMemAgent revision, explicit dependency pins, exact clean
source checkouts, and an integrity manifest destination must all be declared
before any external environment is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutRequirement

MINIMUM_STRICT_SEED_COUNT = 2


def validate_strict_paired_reproducibility(
    *,
    seeds: Sequence[int],
    runtime_requirements: RuntimeRequirements | None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None,
    manifest_path: Path | None,
) -> None:
    """Validate the declaration required for a strict paired research run.

    This function validates only the *declared* experiment contract. Runtime
    provenance and source checkout observations are still collected and checked
    by the existing controlled preflight immediately before measurement.
    Keeping those responsibilities separate prevents a declaration validator
    from being mistaken for evidence that the machine actually satisfied it.
    """

    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a sequence of integers")
    if len(seeds) < MINIMUM_STRICT_SEED_COUNT:
        raise ValueError(
            "strict reproducibility requires at least "
            f"{MINIMUM_STRICT_SEED_COUNT} independent seeds"
        )

    if runtime_requirements is None:
        raise ValueError("strict reproducibility requires runtime requirements")
    if not isinstance(runtime_requirements, RuntimeRequirements):
        raise TypeError("runtime_requirements must be a RuntimeRequirements instance")
    if runtime_requirements.expected_code_revision is None:
        raise ValueError("strict reproducibility requires an exact code revision")
    if not runtime_requirements.require_clean_working_tree:
        raise ValueError("strict reproducibility requires a clean ReMemAgent working tree")
    if not runtime_requirements.dependency_versions:
        raise ValueError("strict reproducibility requires at least one exact dependency version")

    if source_checkout_requirements is None or not source_checkout_requirements:
        raise ValueError("strict reproducibility requires at least one source checkout requirement")
    if not isinstance(source_checkout_requirements, Mapping):
        raise TypeError("source_checkout_requirements must be a mapping")

    dirty_allowed_sources: list[str] = []
    for source_name, requirement in source_checkout_requirements.items():
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError("source checkout requirement names must be non-empty strings")
        if not isinstance(requirement, SourceCheckoutRequirement):
            raise TypeError(
                "source checkout requirement values must be SourceCheckoutRequirement instances"
            )
        if not requirement.require_clean_working_tree:
            dirty_allowed_sources.append(source_name)

    if dirty_allowed_sources:
        names = ", ".join(sorted(dirty_allowed_sources, key=str.casefold))
        raise ValueError(
            "strict reproducibility requires clean source checkouts; dirty checkout "
            f"exceptions were declared for: {names}"
        )

    if manifest_path is None:
        raise ValueError("strict reproducibility requires an integrity manifest destination")
    if not isinstance(manifest_path, Path):
        raise TypeError("manifest_path must be a pathlib.Path")


__all__ = ["MINIMUM_STRICT_SEED_COUNT", "validate_strict_paired_reproducibility"]
