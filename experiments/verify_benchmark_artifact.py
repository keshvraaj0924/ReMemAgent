"""Verify the integrity of a persisted benchmark report artifact."""

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


def main() -> int:
    """Verify a benchmark report and return a process exit status."""

    arguments = parse_args()
    manifest_path = arguments.manifest or arguments.report.with_suffix(
        arguments.report.suffix + ".manifest.json"
    )
    manifest = load_benchmark_artifact_manifest(manifest_path)
    verify_benchmark_artifact(arguments.report, manifest)
    print(f"benchmark artifact integrity verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
