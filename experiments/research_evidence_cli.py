"""Freeze and verify machine-checkable research evidence records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from experiments.research_evidence_record import (
    EVIDENCE_LEVELS,
    ResearchEvidenceRecord,
    build_research_evidence_record,
    verify_research_evidence_record,
    write_research_evidence_record,
)
from experiments.research_experiment_plan import verify_research_experiment_plan

EXPERIMENT_PLAN_ROLE = "experiment_plan"
VERIFICATION_ATTESTATION_ROLE = "verification"


def parse_args() -> argparse.Namespace:
    """Parse research evidence record commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Create a no-overwrite exact-byte research evidence record",
    )
    freeze_parser.add_argument("--experiment", required=True, help="Stable experiment name")
    freeze_parser.add_argument(
        "--level",
        required=True,
        choices=sorted(EVIDENCE_LEVELS),
        help="Highest evidence level established by the retained artifacts",
    )
    freeze_parser.add_argument(
        "--revision",
        required=True,
        help="Exact 40-character ReMemAgent Git commit SHA",
    )
    freeze_parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="ROLE=PATH",
        help="Retained evidence artifact; may be provided more than once",
    )
    freeze_parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional immutable experiment note; may be provided more than once",
    )
    freeze_parser.add_argument("--output", required=True, type=Path, help="Record JSON path")

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify every artifact bound by an existing evidence record",
    )
    verify_parser.add_argument("record", type=Path, help="Persisted research evidence record")
    verify_parser.add_argument(
        "--expected-revision",
        help="Fail unless the record is bound to this exact ReMemAgent commit SHA",
    )
    verify_parser.add_argument(
        "--require-plan-binding",
        action="store_true",
        help=(
            "Require canonical experiment_plan and verification roles and prove that the "
            "retained verification attestation is bound to that exact plan and revision"
        ),
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a deterministic machine-readable verification summary",
    )
    return parser.parse_args()


def _parse_artifact_specs(specifications: list[str]) -> dict[str, Path]:
    """Parse ROLE=PATH specifications while rejecting ambiguous duplicate roles."""

    artifacts: dict[str, Path] = {}
    for specification in specifications:
        role, separator, raw_path = specification.partition("=")
        role = role.strip()
        raw_path = raw_path.strip()
        if not separator or not role or not raw_path:
            raise ValueError("artifact must use non-empty ROLE=PATH syntax")
        if role in artifacts:
            raise ValueError(f"duplicate artifact role: {role}")
        artifacts[role] = Path(raw_path)
    return artifacts


def _freeze(arguments: argparse.Namespace) -> None:
    output_path = arguments.output.resolve()
    record = build_research_evidence_record(
        experiment_name=arguments.experiment,
        evidence_level=arguments.level,
        remem_revision=arguments.revision,
        artifacts=_parse_artifact_specs(arguments.artifact),
        record_directory=output_path.parent,
        notes=arguments.note,
    )
    write_research_evidence_record(output_path, record)
    print(f"research evidence frozen: {output_path}")


def _verification_summary(
    record_path: Path,
    record: ResearchEvidenceRecord,
    *,
    plan_binding_sha256: str | None = None,
) -> dict[str, object]:
    """Build an exact-record identity summary after semantic verification succeeds."""

    raw_bytes = record_path.read_bytes()
    payload: dict[str, object] = {
        "artifact_count": len(record.artifacts),
        "evidence_level": record.evidence_level,
        "experiment_name": record.experiment_name,
        "record_byte_count": len(raw_bytes),
        "record_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "remem_revision": record.remem_revision,
        "schema_version": record.schema_version,
        "verified": True,
    }
    if plan_binding_sha256 is not None:
        payload["experiment_plan_sha256"] = plan_binding_sha256
        payload["plan_binding_verified"] = True
    return payload


def _artifact_paths_by_role(
    record_path: Path,
    record: ResearchEvidenceRecord,
) -> dict[str, Path]:
    """Resolve indexed artifact paths after exact-byte verification has succeeded."""

    root = record_path.parent.resolve()
    return {artifact.role: root / artifact.path for artifact in record.artifacts}


def _load_json_object(path: Path, description: str) -> Mapping[str, object]:
    """Load one retained JSON object with a concise fail-closed error contract."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} root must be a JSON object")
    return payload


def _required_attestation_string(
    payload: Mapping[str, object],
    field_name: str,
) -> str:
    """Return one required non-empty string from a retained verification attestation."""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"verification attestation missing required {field_name}")
    return value


def _verify_plan_binding(
    record_path: Path,
    record: ResearchEvidenceRecord,
) -> str:
    """Prove that retained plan and verification artifacts share one experiment identity."""

    artifact_paths = _artifact_paths_by_role(record_path, record)
    missing_roles = sorted(
        role
        for role in (EXPERIMENT_PLAN_ROLE, VERIFICATION_ATTESTATION_ROLE)
        if role not in artifact_paths
    )
    if missing_roles:
        raise ValueError(
            "plan-binding verification requires evidence roles: " + ", ".join(missing_roles)
        )

    plan_path = artifact_paths[EXPERIMENT_PLAN_ROLE]
    plan = verify_research_experiment_plan(
        plan_path,
        expected_revision=record.remem_revision,
    )
    if plan.experiment_name != record.experiment_name:
        raise ValueError(
            "research evidence experiment name does not match frozen experiment plan: "
            f"{record.experiment_name!r} != {plan.experiment_name!r}"
        )

    attestation_path = artifact_paths[VERIFICATION_ATTESTATION_ROLE]
    attestation = _load_json_object(attestation_path, "verification attestation")
    attested_plan_sha256 = _required_attestation_string(
        attestation,
        "experiment_plan_sha256",
    )
    attested_plan_file_sha256 = _required_attestation_string(
        attestation,
        "experiment_plan_file_sha256",
    )
    attested_plan_name = _required_attestation_string(
        attestation,
        "experiment_plan_name",
    )
    attested_plan_revision = _required_attestation_string(
        attestation,
        "experiment_plan_remem_revision",
    )

    if attested_plan_sha256 != plan.sha256:
        raise ValueError("verification attestation experiment-plan SHA-256 mismatch")
    plan_file_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if attested_plan_file_sha256 != plan_file_sha256:
        raise ValueError("verification attestation experiment-plan file SHA-256 mismatch")
    if attested_plan_name != plan.experiment_name:
        raise ValueError("verification attestation experiment-plan name mismatch")
    if attested_plan_revision != plan.remem_revision:
        raise ValueError("verification attestation experiment-plan revision mismatch")
    return plan.sha256


def _verify(arguments: argparse.Namespace) -> None:
    record_path = arguments.record.resolve()
    record = verify_research_evidence_record(record_path)
    if (
        arguments.expected_revision is not None
        and record.remem_revision != arguments.expected_revision
    ):
        raise ValueError(
            "research evidence revision mismatch: "
            f"expected {arguments.expected_revision}, recorded {record.remem_revision}"
        )
    plan_binding_sha256 = None
    if arguments.require_plan_binding:
        plan_binding_sha256 = _verify_plan_binding(record_path, record)
    if arguments.json:
        print(
            json.dumps(
                _verification_summary(
                    record_path,
                    record,
                    plan_binding_sha256=plan_binding_sha256,
                ),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return
    binding_suffix = ""
    if plan_binding_sha256 is not None:
        binding_suffix = f", plan {plan_binding_sha256}"
    print(
        "research evidence verified: "
        f"{arguments.record} ({record.evidence_level}, {len(record.artifacts)} artifacts, "
        f"revision {record.remem_revision}{binding_suffix})"
    )


def main() -> int:
    """Execute one research evidence command and return a process exit status."""

    arguments = parse_args()
    try:
        if arguments.command == "freeze":
            _freeze(arguments)
        else:
            _verify(arguments)
    except (OSError, TypeError, ValueError) as error:
        print(f"research evidence command failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
