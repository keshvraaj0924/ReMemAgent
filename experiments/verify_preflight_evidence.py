"""Verify persisted paired benchmark readiness evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.preflight_evidence import verify_controlled_paired_preflight_evidence


def parse_args() -> argparse.Namespace:
    """Parse the readiness evidence path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Persisted paired preflight evidence JSON")
    return parser.parse_args()


def load_preflight_evidence(path: Path) -> Mapping[str, Any]:
    """Load a readiness evidence JSON object from disk."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("preflight evidence must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("preflight evidence root must be a JSON object")
    return payload


def verify_preflight_evidence_file(path: Path) -> None:
    """Verify persisted readiness evidence without recollecting mutable state."""

    verify_controlled_paired_preflight_evidence(load_preflight_evidence(path))


def main() -> int:
    """Verify readiness evidence and return a process exit status."""

    arguments = parse_args()
    try:
        verify_preflight_evidence_file(arguments.evidence)
    except (OSError, ValueError, TypeError) as error:
        print(f"preflight evidence verification failed: {error}", file=sys.stderr)
        return 1

    print(f"preflight evidence verified: {arguments.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
