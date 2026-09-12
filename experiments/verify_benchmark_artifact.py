"""Verify a persisted benchmark report against its integrity and identity contracts."""

from __future__ import annotations

import argparse
import hmac
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.benchmark_manifest import (
    load_benchmark_artifact_manifest,
    verify_benchmark_artifact,
)
from experiments.controlled_benchmark_artifacts import (
    validate_persisted_controlled_benchmark_artifact,
)
from experiments.paired_artifacts import validate_persisted_paired_artifact
from experiments.preflight_evidence import (
    PREFLIGHT_EVIDENCE_PROVENANCE_KEY,
    verify_controlled_paired_preflight_evidence,
)
from remem.benchmark_artifacts import validate_persisted_benchmark_artifact


@dataclass(frozen=True, slots=True)
class BenchmarkVerificationResult:
    """Machine-readable attestation for one successfully verified artifact."""

    schema_version: int
    byte_count: int
    sha256: str
    preflight_evidence_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "preflight_evidence_sha256": self.preflight_evidence_sha256,
        }


def parse_args() -> argparse.Namespace:
    """Parse the report and optional integrity/readiness evidence paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Persisted benchmark JSON artifact")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Integrity manifest; defaults to <report>.manifest.json",
    )
    parser.add_argument(
        "--preflight-evidence",
        type=Path,
        help=(
            "Persisted readiness evidence required to verify an evidence-bound measured artifact"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the verified artifact attestation as canonical JSON",
    )
    return parser.parse_args()


def verify_report_artifact(
    report_path: Path,
    manifest_path: Path | None = None,
    preflight_evidence_path: Path | None = None,
) -> BenchmarkVerificationResult:
    """Verify report bytes, identities, admission evidence, and readiness binding."""

    selected_manifest_path = manifest_path or report_path.with_suffix(
        report_path.suffix + ".manifest.json"
    )
    manifest = load_benchmark_artifact_manifest(selected_manifest_path)
    verify_benchmark_artifact(report_path, manifest)
    payload = _load_report_payload(report_path)
    validate_persisted_benchmark_artifact(payload)
    validate_persisted_controlled_benchmark_artifact(payload)
    validate_persisted_paired_artifact(payload)
    preflight_evidence_sha256 = _verify_preflight_evidence_binding(
        payload,
        preflight_evidence_path,
    )
    return BenchmarkVerificationResult(
        schema_version=manifest.schema_version,
        byte_count=manifest.byte_count,
        sha256=manifest.sha256,
        preflight_evidence_sha256=preflight_evidence_sha256,
    )


def _load_report_payload(report_path: Path) -> Mapping[str, Any]:
    """Load a persisted benchmark JSON object after byte-integrity verification."""

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("benchmark artifact must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark artifact root must be a JSON object")
    return payload


def _verify_preflight_evidence_binding(
    payload: Mapping[str, Any],
    preflight_evidence_path: Path | None,
) -> str | None:
    """Verify an artifact's readiness digest against the retained evidence object.

    Evidence-bound measured artifacts fail closed unless the original readiness JSON
    is supplied. Legacy artifacts without the readiness digest remain independently
    verifiable and reject an unrelated evidence argument rather than silently
    implying a binding that was never persisted.
    """

    runtime_provenance = payload.get("runtime_provenance")
    stored_digest = (
        runtime_provenance.get(PREFLIGHT_EVIDENCE_PROVENANCE_KEY)
        if isinstance(runtime_provenance, Mapping)
        else None
    )

    if stored_digest is None and preflight_evidence_path is None:
        return None
    if stored_digest is None:
        raise ValueError("benchmark artifact is not bound to preflight evidence")
    if not isinstance(stored_digest, str):
        raise ValueError("preflight evidence provenance digest must be a string")
    if preflight_evidence_path is None:
        raise ValueError(
            "evidence-bound benchmark artifact requires --preflight-evidence for verification"
        )

    evidence = _load_preflight_evidence(preflight_evidence_path)
    verify_controlled_paired_preflight_evidence(evidence)
    evidence_digest = evidence.get("evidence_sha256")
    if not isinstance(evidence_digest, str):
        raise ValueError("verified preflight evidence must contain evidence_sha256")
    if not hmac.compare_digest(stored_digest, evidence_digest):
        raise ValueError("benchmark artifact readiness digest does not match preflight evidence")
    return evidence_digest


def _load_preflight_evidence(path: Path) -> Mapping[str, Any]:
    """Load retained readiness evidence without recollecting mutable runtime state."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("preflight evidence must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("preflight evidence root must be a JSON object")
    return payload


def main() -> int:
    """Verify a benchmark report and return a process exit status."""

    arguments = parse_args()
    try:
        result = verify_report_artifact(
            arguments.report,
            arguments.manifest,
            arguments.preflight_evidence,
        )
    except (OSError, ValueError) as error:
        print(f"benchmark artifact verification failed: {error}", file=sys.stderr)
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
        # Preserve the established CLI success prefix for callers that parse it.
        # Configuration identity is still verified by verify_report_artifact when present.
        print(f"benchmark artifact integrity verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
