"""Verify a persisted benchmark report against its exact-byte integrity manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.benchmark_manifest import (
    load_benchmark_artifact_manifest,
    verify_benchmark_artifact,
)


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
    """Verify report bytes against a persisted benchmark artifact manifest."""

    selected_manifest_path = manifest_path or report_path.with_suffix(
        report_path.suffix + ".manifest.json"
    )
    manifest = load_benchmark_artifact_manifest(selected_manifest_path)
    verify_benchmark_artifact(report_path, manifest)


def main() -> int:
    """Verify a benchmark report and return a process exit status."""

    arguments = parse_args()
    verify_report_artifact(arguments.report, arguments.manifest)
    print(f"benchmark artifact integrity verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
