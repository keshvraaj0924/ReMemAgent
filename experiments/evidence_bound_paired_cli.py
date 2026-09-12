"""CLI bridge that binds paired measurement to persisted readiness evidence.

The primary paired CLI predates persisted readiness evidence. This bridge preserves
its full argument contract while adding ``--require-preflight-evidence PATH`` for
measured controlled runs. The option is removed before delegating to the existing
parser, and the controlled runner is replaced only for that invocation.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import experiments.paired_benchmark_cli as paired_cli
from experiments.evidence_bound_paired_benchmark import (
    run_evidence_bound_paired_external_benchmarks,
)

READINESS_EVIDENCE_OPTION = "--require-preflight-evidence"
SOURCE_CHECKOUT_OPTION = "--source-checkout"
SOURCE_REVISION_OPTION = "--require-source-revision"


def main() -> int:
    """Run the paired CLI, optionally requiring matching persisted readiness evidence."""

    original_argv = list(sys.argv)
    try:
        evidence_path, delegated_argv = _extract_readiness_evidence_path(original_argv)
        if evidence_path is None:
            return paired_cli.main()
        if _contains_option(delegated_argv, "--preflight-only"):
            raise SystemExit(
                f"error: {READINESS_EVIDENCE_OPTION} is for measured execution, not --preflight-only"
            )
        _require_controlled_source_contract(delegated_argv)
        readiness_evidence = _load_readiness_evidence(evidence_path)
        original_runner = paired_cli.run_controlled_paired_external_benchmarks

        def evidence_bound_runner(*args: Any, **kwargs: Any):
            return run_evidence_bound_paired_external_benchmarks(
                readiness_evidence,
                *args,
                **kwargs,
            )

        paired_cli.run_controlled_paired_external_benchmarks = evidence_bound_runner
        sys.argv = delegated_argv
        try:
            return paired_cli.main()
        finally:
            paired_cli.run_controlled_paired_external_benchmarks = original_runner
    finally:
        sys.argv = original_argv


def _extract_readiness_evidence_path(argv: Sequence[str]) -> tuple[Path | None, list[str]]:
    """Remove the evidence option from argv and return its path exactly once."""

    delegated = [argv[0]] if argv else []
    evidence_path: Path | None = None
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == READINESS_EVIDENCE_OPTION:
            if evidence_path is not None:
                raise SystemExit(f"error: {READINESS_EVIDENCE_OPTION} may be specified only once")
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit(f"error: {READINESS_EVIDENCE_OPTION} requires a path")
            evidence_path = Path(argv[index + 1])
            index += 2
            continue
        prefix = f"{READINESS_EVIDENCE_OPTION}="
        if argument.startswith(prefix):
            if evidence_path is not None:
                raise SystemExit(f"error: {READINESS_EVIDENCE_OPTION} may be specified only once")
            value = argument[len(prefix) :]
            if not value:
                raise SystemExit(f"error: {READINESS_EVIDENCE_OPTION} requires a path")
            evidence_path = Path(value)
            index += 1
            continue
        delegated.append(argument)
        index += 1
    return evidence_path, delegated


def _contains_option(argv: Sequence[str], option: str) -> bool:
    """Return whether argv contains an option in separated or ``--name=value`` form."""

    prefix = f"{option}="
    return any(argument == option or argument.startswith(prefix) for argument in argv[1:])


def _require_controlled_source_contract(argv: Sequence[str]) -> None:
    """Fail closed unless evidence-bound measurement will use controlled admission."""

    if not _contains_option(argv, SOURCE_CHECKOUT_OPTION) or not _contains_option(
        argv, SOURCE_REVISION_OPTION
    ):
        raise SystemExit(
            "error: --require-preflight-evidence requires declared source checkouts "
            "and exact source revisions"
        )


def _load_readiness_evidence(path: Path) -> Mapping[str, Any]:
    """Load a readiness evidence JSON object without collecting mutable runtime state."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"error: unable to read readiness evidence {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid readiness evidence JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit("error: readiness evidence root must be a JSON object")
    return payload


__all__ = ["READINESS_EVIDENCE_OPTION", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
