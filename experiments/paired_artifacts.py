"""Persistence adapter for complete paired benchmark execution results.

This module keeps temporal execution provenance attached to the measured paired
result instead of reconstructing it later from seed order. The lower-level
benchmark report serializer remains responsible for report/configuration
validation and experiment identity construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.benchmark_report import save_paired_benchmark_result
from experiments.paired_benchmark import PairedBenchmarkResult, PairedSeedExecution

PAIRED_EXECUTION_ORDER_PROVENANCE_KEY = "paired_execution_order_sha256"


def save_paired_execution_result(
    result: PairedBenchmarkResult,
    output_path: Path,
    *,
    runtime_provenance: Mapping[str, object] | None = None,
) -> Path:
    """Persist a complete paired result including validated temporal provenance.

    The execution-order digest is injected into runtime provenance before the
    lower-level serializer constructs the experiment identity. Consequently, a
    different temporal condition plan cannot share the same experiment identity
    even when policies, seeds, and benchmark protocol are otherwise identical.
    The full order is then written into the final artifact for human inspection
    and downstream verification.
    """

    execution_order = _validate_execution_order(result)
    provenance = dict(runtime_provenance or {})
    if PAIRED_EXECUTION_ORDER_PROVENANCE_KEY in provenance:
        raise ValueError(
            f"runtime_provenance reserves {PAIRED_EXECUTION_ORDER_PROVENANCE_KEY!r}"
        )
    provenance[PAIRED_EXECUTION_ORDER_PROVENANCE_KEY] = _execution_order_fingerprint(
        execution_order
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.paired.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_paired_benchmark_result(
            result.baseline_reports,
            result.treatment_reports,
            result.comparison,
            temporary_path,
            runtime_provenance=provenance,
        )
        payload = json.loads(temporary_path.read_text(encoding="utf-8"))
        payload["execution_order"] = [asdict(entry) for entry in execution_order]
        _write_json_file(payload, temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _validate_execution_order(
    result: PairedBenchmarkResult,
) -> tuple[PairedSeedExecution, ...]:
    """Validate that recorded temporal provenance exactly matches paired seeds."""

    execution_order = tuple(result.execution_order)
    expected_seeds = tuple(result.comparison.seeds)
    if len(execution_order) != len(expected_seeds):
        raise ValueError("execution_order must contain exactly one entry per paired seed")

    actual_seeds = tuple(entry.seed for entry in execution_order)
    if actual_seeds != expected_seeds:
        raise ValueError("execution_order seeds must exactly match comparison seeds")

    for seed_index, entry in enumerate(execution_order):
        expected_conditions = (
            ("baseline", "treatment")
            if seed_index % 2 == 0
            else ("treatment", "baseline")
        )
        actual_conditions = (entry.first_condition, entry.second_condition)
        if actual_conditions != expected_conditions:
            raise ValueError(
                "execution_order must match the deterministic counterbalanced condition plan"
            )
    return execution_order


def _execution_order_fingerprint(execution_order: tuple[PairedSeedExecution, ...]) -> str:
    """Return a deterministic digest for one validated temporal execution plan."""

    canonical_payload = json.dumps(
        [asdict(entry) for entry in execution_order],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def _write_json_file(payload: Mapping[str, Any], output_path: Path) -> None:
    """Write deterministic JSON to an already-private temporary path."""

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "PAIRED_EXECUTION_ORDER_PROVENANCE_KEY",
    "save_paired_execution_result",
]
