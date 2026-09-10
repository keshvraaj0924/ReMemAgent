"""Verify a persisted benchmark report against its integrity and identity contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.benchmark_manifest import (
    load_benchmark_artifact_manifest,
    verify_benchmark_artifact,
)
from experiments.paired_artifacts import validate_persisted_paired_artifact
from remem.benchmark_artifacts import validate_persisted_benchmark_artifact


def parse_args() -> argparse.Namespace:
    """Parse the report and optional manifest paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Persisted benchmark JSON artifact")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Integrity manifest; defaults to <report>.manifest.json",
    )
    return parser.parse_args()


def verify_report_artifact(report_path: Path, manifest_path: Path | None = None) -> None:
    """Verify report bytes, embedded identities, and paired execution provenance."""

    selected_manifest_path = manifest_path or report_path.with_suffix(
        report_path.suffix + ".manifest.json"
    )
    manifest = load_benchmark_artifact_manifest(selected_manifest_path)
    verify_benchmark_artifact(report_path, manifest)
    payload = _load_report_payload(report_path)
    validate_persisted_benchmark_artifact(payload)
    validate_persisted_paired_artifact(payload)


def _load_report_payload(report_path: Path) -> Mapping[str, Any]:
    """Load a persisted benchmark JSON object after byte-integrity verification."""

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("benchmark artifact must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark artifact root must be a JSON object")
    return payload


def main() -> int:
    """Verify a benchmark report and return a process exit status."""

    arguments = parse_args()
    try:
        verify_report_artifact(arguments.report, arguments.manifest)
    except (OSError, ValueError) as error:
        print(f"benchmark artifact verification failed: {error}", file=sys.stderr)
        return 1

    # Preserve the established CLI success prefix for callers that parse it.
    # Configuration identity is still verified by verify_report_artifact when present.
    print(f"benchmark artifact integrity verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
