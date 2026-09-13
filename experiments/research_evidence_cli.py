"""Freeze and verify machine-checkable research evidence records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from experiments.research_evidence_record import (
    EVIDENCE_LEVELS,
    ResearchEvidenceRecord,
    build_research_evidence_record,
    verify_research_evidence_record,
    write_research_evidence_record,
)


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
) -> dict[str, object]:
    """Build an exact-record identity summary after semantic verification succeeds."""

    raw_bytes = record_path.read_bytes()
    return {
        "artifact_count": len(record.artifacts),
        "evidence_level": record.evidence_level,
        "experiment_name": record.experiment_name,
        "record_byte_count": len(raw_bytes),
        "record_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "remem_revision": record.remem_revision,
        "schema_version": record.schema_version,
        "verified": True,
    }


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
    if arguments.json:
        print(
            json.dumps(
                _verification_summary(record_path, record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return
    print(
        "research evidence verified: "
        f"{arguments.record} ({record.evidence_level}, {len(record.artifacts)} artifacts, "
        f"revision {record.remem_revision})"
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
