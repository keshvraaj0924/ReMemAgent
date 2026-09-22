"""Verification boundary for experiment summaries used in research reports."""

from __future__ import annotations

from pathlib import Path

from remem.evidence_bundle import EvidenceBundle
from remem.evidence_paths import normalize_evidence_path
from remem.reporting import ExperimentSummary
from remem.reporting_io import load_experiment_summary


def verify_reportable_summary(
    *,
    bundle: EvidenceBundle,
    root: str | Path,
    summary_path: str | Path,
) -> ExperimentSummary:
    """Return a summary only when its complete evidence bundle verifies.

    The summary must be explicitly required by the evidence contract. Verification
    checks the complete bundle before parsing the summary, so callers cannot treat
    an orphaned or byte-modified aggregate as reportable research evidence.
    """
    if not isinstance(bundle, EvidenceBundle):
        raise TypeError("bundle must be an EvidenceBundle")

    normalized_summary_path = normalize_evidence_path(summary_path)
    if normalized_summary_path not in bundle.contract.required_paths:
        raise ValueError("summary_path must be required by the evidence contract")

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"evidence root directory does not exist: {root_path}")

    bundle.verify(root=root_path)
    return load_experiment_summary(root_path / normalized_summary_path)


__all__ = ["verify_reportable_summary"]
