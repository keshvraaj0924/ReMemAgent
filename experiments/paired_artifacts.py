"""Persistence and verification for complete paired benchmark execution results.

This module keeps temporal execution provenance attached to the measured paired
result instead of reconstructing it later from seed order. The lower-level
benchmark report serializer remains responsible for report/configuration
validation and experiment identity construction.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from experiments.benchmark_report import (
    paired_configuration_fingerprint,
    save_paired_benchmark_result,
)
from experiments.experiment_identity import verify_paired_experiment_identity
from experiments.paired_benchmark import PairedBenchmarkResult, PairedSeedExecution
from remem.benchmark import BenchmarkRunConfiguration

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


def validate_persisted_paired_artifact(payload: Mapping[str, Any]) -> None:
    """Verify paired protocol identity plus optional temporal execution provenance.

    This validator is intended to run after the generic per-run artifact checks.
    It recomputes the top-level paired configuration fingerprint and experiment
    identity from the persisted condition configurations, seed set, and runtime
    provenance. Legacy paired artifacts without an experiment identity remain
    readable but do not gain an identity guarantee retroactively.
    """

    if "baseline" not in payload and "treatment" not in payload:
        return

    validate_persisted_paired_execution_provenance(payload)

    identity = payload.get("experiment_identity")
    if identity is None:
        return
    if not isinstance(identity, str):
        raise ValueError("paired experiment_identity must be a string")

    baseline_configuration = _condition_configuration(payload, "baseline")
    treatment_configuration = _condition_configuration(payload, "treatment")
    seeds = _persisted_seeds(payload)
    runtime_provenance = payload.get("runtime_provenance")
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError("paired artifact runtime_provenance must be a mapping")

    stored_fingerprint = payload.get("configuration_fingerprint")
    if not isinstance(stored_fingerprint, str):
        raise ValueError("paired artifact configuration_fingerprint must be a string")
    expected_fingerprint = paired_configuration_fingerprint(
        baseline_configuration,
        treatment_configuration,
    )
    if not hmac.compare_digest(stored_fingerprint, expected_fingerprint):
        raise ValueError(
            "paired configuration fingerprint does not match persisted condition configurations"
        )

    verify_paired_experiment_identity(
        identity,
        replace(baseline_configuration, seed=None),
        replace(treatment_configuration, seed=None),
        seeds,
        runtime_provenance,
    )


def validate_persisted_paired_execution_provenance(payload: Mapping[str, Any]) -> None:
    """Verify paired execution trace semantics and its persisted SHA-256 digest.

    Artifacts without an ``execution_order`` field are treated as legacy paired
    artifacts and are left to the generic benchmark-artifact validator. Once the
    field is present, all paired execution provenance fields are mandatory and
    validated fail-closed.
    """

    if "execution_order" not in payload:
        return

    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or any(
        not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds
    ):
        raise ValueError("paired execution artifact seeds must be a list of integers")
    if not raw_seeds:
        raise ValueError("paired execution artifact seeds must not be empty")
    if len(raw_seeds) != len(set(raw_seeds)):
        raise ValueError("paired execution artifact seeds must be unique")

    raw_execution_order = payload["execution_order"]
    if not isinstance(raw_execution_order, list):
        raise ValueError("execution_order must be a list")
    if len(raw_execution_order) != len(raw_seeds):
        raise ValueError("execution_order must contain exactly one entry per paired seed")

    normalized_entries: list[PairedSeedExecution] = []
    for seed_index, raw_entry in enumerate(raw_execution_order):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("execution_order entries must be JSON objects")
        if set(raw_entry) != {"seed", "first_condition", "second_condition"}:
            raise ValueError("execution_order entries must use the exact persisted schema")

        seed = raw_entry["seed"]
        first_condition = raw_entry["first_condition"]
        second_condition = raw_entry["second_condition"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("execution_order seed must be an integer")
        if not isinstance(first_condition, str) or not isinstance(second_condition, str):
            raise ValueError("execution_order conditions must be strings")
        if seed != raw_seeds[seed_index]:
            raise ValueError("execution_order seeds must exactly match artifact seeds")

        expected_conditions = (
            ("baseline", "treatment")
            if seed_index % 2 == 0
            else ("treatment", "baseline")
        )
        if (first_condition, second_condition) != expected_conditions:
            raise ValueError(
                "execution_order must match the deterministic counterbalanced condition plan"
            )
        normalized_entries.append(
            PairedSeedExecution(
                seed=seed,
                first_condition=first_condition,
                second_condition=second_condition,
            )
        )

    runtime_provenance = payload.get("runtime_provenance")
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError("paired execution artifact requires runtime_provenance")
    stored_digest = runtime_provenance.get(PAIRED_EXECUTION_ORDER_PROVENANCE_KEY)
    if not isinstance(stored_digest, str):
        raise ValueError(
            f"runtime_provenance requires {PAIRED_EXECUTION_ORDER_PROVENANCE_KEY!r}"
        )

    expected_digest = _execution_order_fingerprint(tuple(normalized_entries))
    if not hmac.compare_digest(stored_digest, expected_digest):
        raise ValueError("paired execution order provenance digest does not match execution_order")


def _condition_configuration(
    payload: Mapping[str, Any],
    condition_name: str,
) -> BenchmarkRunConfiguration:
    """Reconstruct one condition's reference configuration from persisted reports."""

    condition = payload.get(condition_name)
    if not isinstance(condition, Mapping):
        raise ValueError(f"paired artifact {condition_name} must be a mapping")
    reports = condition.get("reports")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes, bytearray)):
        raise ValueError(f"paired artifact {condition_name}.reports must be a sequence")
    if not reports:
        raise ValueError(f"paired artifact {condition_name}.reports must not be empty")
    first_report = reports[0]
    if not isinstance(first_report, Mapping):
        raise ValueError(f"paired artifact {condition_name}.reports[0] must be a mapping")
    configuration_payload = first_report.get("configuration")
    if not isinstance(configuration_payload, Mapping):
        raise ValueError(f"paired artifact {condition_name} requires configuration provenance")
    try:
        return BenchmarkRunConfiguration(**dict(configuration_payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"paired artifact {condition_name} contains invalid configuration provenance"
        ) from exc


def _persisted_seeds(payload: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the exact validated independent seed set stored by a paired artifact."""

    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes, bytearray)):
        raise ValueError("paired artifact seeds must be a sequence")
    seeds = tuple(raw_seeds)
    if not seeds:
        raise ValueError("paired artifact seeds must not be empty")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("paired artifact seeds must contain only integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("paired artifact seeds must be unique")
    return seeds


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
    "validate_persisted_paired_artifact",
    "validate_persisted_paired_execution_provenance",
]
