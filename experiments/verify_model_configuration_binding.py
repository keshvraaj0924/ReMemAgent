"""Verify frozen model configuration against persisted benchmark runtime provenance."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.evidence_bound_paired_cli import (
    EXPERIMENT_PARAMETERS_PROVENANCE_KEY,
    EXPERIMENT_PLAN_PROVENANCE_KEY,
    MODEL_IDENTITY_PROVENANCE_KEY,
)
from experiments.research_experiment_plan import JsonScalar, verify_research_experiment_plan


@dataclass(frozen=True, slots=True)
class ModelConfigurationBindingResult:
    """Verified model declaration and exact report identity."""

    report_byte_count: int
    report_sha256: str
    experiment_plan_sha256: str
    remem_revision: str
    model_identity: str | None
    parameters: tuple[tuple[str, JsonScalar], ...]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible attestation."""

        return {
            "report_byte_count": self.report_byte_count,
            "report_sha256": self.report_sha256,
            "experiment_plan_sha256": self.experiment_plan_sha256,
            "remem_revision": self.remem_revision,
            "model_identity": self.model_identity,
            "parameters": dict(self.parameters),
            "model_configuration_binding_verified": True,
        }


def verify_model_configuration_binding(
    report_path: str | Path,
    experiment_plan_path: str | Path,
) -> ModelConfigurationBindingResult:
    """Verify that measured runtime model metadata exactly matches the frozen plan."""

    report_bytes = Path(report_path).read_bytes()
    payload = _load_json_object(report_bytes, "benchmark report")
    runtime_provenance = payload.get("runtime_provenance")
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError("benchmark report requires runtime_provenance for model binding")

    stored_plan_sha256 = runtime_provenance.get(EXPERIMENT_PLAN_PROVENANCE_KEY)
    if not isinstance(stored_plan_sha256, str) or not stored_plan_sha256:
        raise ValueError("benchmark report is not bound to a research experiment plan")

    plan = verify_research_experiment_plan(
        experiment_plan_path,
        expected_sha256=stored_plan_sha256,
    )
    _verify_revision(runtime_provenance, plan.remem_revision)
    _verify_model_identity(runtime_provenance, plan.model_identity)
    _verify_parameters(runtime_provenance, dict(plan.parameters))

    return ModelConfigurationBindingResult(
        report_byte_count=len(report_bytes),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        experiment_plan_sha256=plan.sha256,
        remem_revision=plan.remem_revision,
        model_identity=plan.model_identity,
        parameters=plan.parameters,
    )


def _load_json_object(raw_bytes: bytes, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be a JSON object")
    return payload


def _verify_revision(runtime_provenance: Mapping[str, Any], expected_revision: str) -> None:
    recorded_revision = runtime_provenance.get("code_revision")
    if not isinstance(recorded_revision, str) or not recorded_revision:
        raise ValueError("benchmark report runtime provenance must contain code_revision")
    if not hmac.compare_digest(recorded_revision, expected_revision):
        raise ValueError("benchmark report code revision does not match research experiment plan")


def _verify_model_identity(
    runtime_provenance: Mapping[str, Any],
    expected_identity: str | None,
) -> None:
    if MODEL_IDENTITY_PROVENANCE_KEY not in runtime_provenance:
        raise ValueError("benchmark report is missing admitted model identity provenance")
    recorded_identity = runtime_provenance[MODEL_IDENTITY_PROVENANCE_KEY]
    if recorded_identity is not None and not isinstance(recorded_identity, str):
        raise TypeError("benchmark report model identity provenance must be a string or null")
    if recorded_identity != expected_identity:
        raise ValueError("benchmark report model identity does not match research experiment plan")


def _verify_parameters(
    runtime_provenance: Mapping[str, Any],
    expected_parameters: Mapping[str, JsonScalar],
) -> None:
    if EXPERIMENT_PARAMETERS_PROVENANCE_KEY not in runtime_provenance:
        raise ValueError("benchmark report is missing admitted experiment parameter provenance")
    recorded_parameters = runtime_provenance[EXPERIMENT_PARAMETERS_PROVENANCE_KEY]
    if not isinstance(recorded_parameters, Mapping):
        raise TypeError("benchmark report experiment parameter provenance must be a JSON object")

    expected_json = _canonical_parameter_json(expected_parameters)
    recorded_json = _canonical_parameter_json(recorded_parameters)
    if not hmac.compare_digest(recorded_json, expected_json):
        raise ValueError(
            "benchmark report experiment parameters do not match research experiment plan"
        )


def _canonical_parameter_json(parameters: Mapping[object, object]) -> str:
    normalized: dict[str, JsonScalar] = {}
    for name, value in parameters.items():
        if not isinstance(name, str) or not name:
            raise TypeError("experiment parameter provenance keys must be non-empty strings")
        if value is None or isinstance(value, (str, bool, int)):
            normalized_value: JsonScalar = value
        elif isinstance(value, float):
            normalized_value = value
        else:
            raise TypeError("experiment parameter provenance values must be JSON scalars")
        normalized[name] = normalized_value
    try:
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError("experiment parameter provenance values must be finite") from exc


def parse_args() -> argparse.Namespace:
    """Parse retained report and frozen-plan paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Persisted measured benchmark report")
    parser.add_argument(
        "experiment_plan", type=Path, help="Retained frozen research experiment plan"
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> int:
    """Verify one retained model-configuration binding."""

    arguments = parse_args()
    try:
        result = verify_model_configuration_binding(arguments.report, arguments.experiment_plan)
    except (OSError, TypeError, ValueError) as error:
        print(f"model configuration binding verification failed: {error}", file=sys.stderr)
        return 1

    if arguments.json_output:
        print(
            json.dumps(
                result.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    else:
        print(f"model configuration binding verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
