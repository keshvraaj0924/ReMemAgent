"""Verify a persisted benchmark report against its exact-byte integrity manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.benchmark_manifest import (
    load_benchmark_artifact_manifest,
    verify_benchmark_artifact,
)


def parse_args() -> argparse.Namespace:
    """Parse report and manifest paths for integrity verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Persisted benchmark report JSON")
    parser.add_argument("manifest", type=Path, help="Benchmark artifact manifest JSON")
    return parser.parse_args()


def verify_report_artifact(report_path: Path, manifest_path: Path) -> None:
    """Verify the report bytes against a persisted benchmark artifact manifest."""

    manifest = load_benchmark_artifact_manifest(manifest_path)
    verify_benchmark_artifact(report_path, manifest)


def main() -> int:
    """Verify one benchmark artifact and return a process status."""

    arguments = parse_args()
    verify_report_artifact(arguments.report, arguments.manifest)
    print(f"benchmark artifact verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
