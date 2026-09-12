"""Persistence helpers for controlled single and repeated benchmark artifacts.

Controlled execution validates runtime and optional source-checkout state before
benchmark side effects. This module preserves that exact admitted state in the
artifact and binds deterministic evidence digests into runtime provenance before
experiment identity construction.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from experiments.benchmark_report import save_benchmark_report, save_repeated_benchmark_reports
from experiments.controlled_external_benchmark import (
    ControlledExternalBenchmarkResult,
    ControlledRepeatedExternalBenchmarkResult,
)
from experiments.paired_artifacts import (
    RUNTIME_REQUIREMENTS_PROVENANCE_KEY,
    SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY,
    SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY,
    validate_persisted_runtime_requirements,
    validate_persisted_source_checkouts,
)
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import (
    SourceCheckoutProvenance,
    SourceCheckoutRequirement,
    source_checkout_provenance_sha256,
    source_checkout_provenance_to_dict,
    source_checkout_requirements_sha256,
    source_checkout_requirements_to_dict,
    validate_source_checkout_requirements,
)


def save_controlled_benchmark_result(
    result: ControlledExternalBenchmarkResult,
    output_path: Path,
    *,
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
) -> Path:
    """Persist one controlled report with exact pre-measurement admission evidence."""

    provenance, metadata = _build_controlled_evidence(
        result.runtime_provenance.to_dict(),
        runtime_requirements=runtime_requirements,
        source_checkout_provenance=result.source_checkout_provenance,
        source_checkout_requirements=source_checkout_requirements,
    )

    def writer(path: Path) -> Path:
        return save_benchmark_report(result.report, path, runtime_provenance=provenance)

    return _save_with_metadata(output_path, writer, metadata)


def save_controlled_repeated_benchmark_result(
    result: ControlledRepeatedExternalBenchmarkResult,
    output_path: Path,
    *,
    runtime_requirements: RuntimeRequirements | None = None,
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None = None,
    statistics: Mapping[str, Any] | None = None,
) -> Path:
    """Persist repeated controlled reports with their shared admitted state."""

    provenance, metadata = _build_controlled_evidence(
        result.runtime_provenance.to_dict(),
        runtime_requirements=runtime_requirements,
        source_checkout_provenance=result.source_checkout_provenance,
        source_checkout_requirements=source_checkout_requirements,
    )

    def writer(path: Path) -> Path:
        return save_repeated_benchmark_reports(
            result.reports,
            path,
            runtime_provenance=provenance,
            statistics=statistics,
        )

    return _save_with_metadata(output_path, writer, metadata)


def validate_persisted_controlled_benchmark_artifact(payload: Mapping[str, Any]) -> None:
    """Validate optional identity-bound runtime and source admission evidence."""

    validate_persisted_runtime_requirements(payload)
    validate_persisted_source_checkouts(payload)


def _build_controlled_evidence(
    runtime_provenance: Mapping[str, object],
    *,
    runtime_requirements: RuntimeRequirements | None,
    source_checkout_provenance: Mapping[str, SourceCheckoutProvenance],
    source_checkout_requirements: Mapping[str, SourceCheckoutRequirement] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return identity-bound provenance plus full persisted admission metadata."""

    provenance = dict(runtime_provenance)
    metadata: dict[str, object] = {}

    if runtime_requirements is not None:
        if not isinstance(runtime_requirements, RuntimeRequirements):
            raise TypeError("runtime_requirements must be a RuntimeRequirements instance")
        _reserve_provenance_key(provenance, RUNTIME_REQUIREMENTS_PROVENANCE_KEY)
        provenance[RUNTIME_REQUIREMENTS_PROVENANCE_KEY] = runtime_requirements.sha256
        metadata["runtime_requirements"] = runtime_requirements.to_dict()

    if source_checkout_provenance or source_checkout_requirements is not None:
        if not source_checkout_provenance:
            raise ValueError("source_checkout_provenance must not be empty")
        if source_checkout_requirements is None or not source_checkout_requirements:
            raise ValueError(
                "source_checkout_provenance and source_checkout_requirements must be provided together"
            )
        validate_source_checkout_requirements(
            source_checkout_provenance,
            source_checkout_requirements,
        )
        _reserve_provenance_key(provenance, SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY)
        _reserve_provenance_key(provenance, SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY)
        provenance[SOURCE_CHECKOUT_REQUIREMENTS_PROVENANCE_KEY] = (
            source_checkout_requirements_sha256(source_checkout_requirements)
        )
        provenance[SOURCE_CHECKOUT_SNAPSHOT_PROVENANCE_KEY] = source_checkout_provenance_sha256(
            source_checkout_provenance
        )
        metadata["source_checkout_requirements"] = source_checkout_requirements_to_dict(
            source_checkout_requirements
        )
        metadata["source_checkout_provenance"] = source_checkout_provenance_to_dict(
            source_checkout_provenance
        )

    return provenance, metadata


def _reserve_provenance_key(provenance: Mapping[str, object], key: str) -> None:
    """Reject caller provenance that attempts to spoof controlled evidence digests."""

    if key in provenance:
        raise ValueError(f"runtime_provenance reserves {key!r}")


def _save_with_metadata(
    output_path: Path,
    writer: Callable[[Path], object],
    metadata: Mapping[str, object],
) -> Path:
    """Write the identity-bearing report, attach evidence, then atomically publish it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.controlled.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        payload = json.loads(temporary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("benchmark artifact writer must produce a JSON object")
        payload.update(metadata)
        _write_json(payload, temporary_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _write_json(payload: Mapping[str, object], output_path: Path) -> None:
    """Write deterministic UTF-8 JSON used by benchmark artifact tooling."""

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "save_controlled_benchmark_result",
    "save_controlled_repeated_benchmark_result",
    "validate_persisted_controlled_benchmark_artifact",
]
