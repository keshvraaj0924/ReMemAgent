import pytest

from remem.artifact_manifest import ArtifactManifest
from remem.evidence_bundle import EvidenceBundle
from remem.evidence_contract import EvidenceContract
from remem.reportable_summary import verify_reportable_summary
from remem.reporting import ExperimentSummary
from remem.reporting_io import save_experiment_summary


def _reportable_bundle(tmp_path):
    summary = ExperimentSummary.from_dict(
        {
            "counters": {"accepted": 3},
            "mean_durations": {"retrieval": 0.25},
            "counter_rates": {"acceptance": 0.75},
        }
    )
    save_experiment_summary(summary, tmp_path / "summary.json")
    (tmp_path / "raw.jsonl").write_text('{"task":"t1","success":true}\n', encoding="utf-8")
    manifest = ArtifactManifest.capture(
        run_id="run-001",
        paths=("summary.json", "raw.jsonl"),
        root=tmp_path,
    )
    contract = EvidenceContract.create(
        name="reportable-benchmark",
        required_paths=("summary.json", "raw.jsonl"),
    )
    return summary, EvidenceBundle.create(contract=contract, manifest=manifest)


def test_verify_reportable_summary_returns_validated_summary(tmp_path):
    expected, bundle = _reportable_bundle(tmp_path)

    actual = verify_reportable_summary(
        bundle=bundle,
        root=tmp_path,
        summary_path="summary.json",
    )

    assert actual == expected


def test_verify_reportable_summary_rejects_orphan_summary(tmp_path):
    _, bundle = _reportable_bundle(tmp_path)
    contract = EvidenceContract.create(name="raw-only", required_paths=("raw.jsonl",))
    raw_only_bundle = EvidenceBundle.create(contract=contract, manifest=bundle.manifest)

    with pytest.raises(ValueError, match="required by the evidence contract"):
        verify_reportable_summary(
            bundle=raw_only_bundle,
            root=tmp_path,
            summary_path="summary.json",
        )


def test_verify_reportable_summary_detects_any_required_evidence_mutation(tmp_path):
    _, bundle = _reportable_bundle(tmp_path)
    (tmp_path / "raw.jsonl").write_text('{"task":"t1","success":false}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        verify_reportable_summary(bundle=bundle, root=tmp_path, summary_path="summary.json")


def test_verify_reportable_summary_detects_summary_mutation(tmp_path):
    _, bundle = _reportable_bundle(tmp_path)
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        verify_reportable_summary(bundle=bundle, root=tmp_path, summary_path="summary.json")


def test_verify_reportable_summary_rejects_invalid_inputs(tmp_path):
    _, bundle = _reportable_bundle(tmp_path)

    with pytest.raises(TypeError, match="bundle must be an EvidenceBundle"):
        verify_reportable_summary(
            bundle=object(),  # type: ignore[arg-type]
            root=tmp_path,
            summary_path="summary.json",
        )
    with pytest.raises(ValueError):
        verify_reportable_summary(bundle=bundle, root=tmp_path, summary_path="../summary.json")
    with pytest.raises(FileNotFoundError, match="evidence root directory"):
        verify_reportable_summary(
            bundle=bundle,
            root=tmp_path / "missing",
            summary_path="summary.json",
        )
