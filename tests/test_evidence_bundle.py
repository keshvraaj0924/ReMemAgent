import json

import pytest

from remem.artifact_manifest import ArtifactManifest
from remem.evidence_bundle import EvidenceBundle
from remem.evidence_contract import EvidenceContract


def _bundle(tmp_path):
    (tmp_path / "results.json").write_text('{"score":1}\n', encoding="utf-8")
    (tmp_path / "optional.log").write_text("debug\n", encoding="utf-8")
    manifest = ArtifactManifest.capture(
        run_id="run-001",
        paths=("results.json", "optional.log"),
        root=tmp_path,
    )
    contract = EvidenceContract.create(name="benchmark", required_paths=("results.json",))
    return EvidenceBundle.create(contract=contract, manifest=manifest)


def test_bundle_round_trip_and_verify(tmp_path):
    bundle = _bundle(tmp_path)

    restored = EvidenceBundle.from_json(bundle.to_json())

    assert restored == bundle
    restored.verify(root=tmp_path)


def test_bundle_save_and_load_is_deterministic(tmp_path):
    bundle = _bundle(tmp_path)
    destination = tmp_path / "metadata" / "evidence.json"

    bundle.save(destination)
    first_bytes = destination.read_bytes()
    loaded = EvidenceBundle.load(destination)
    loaded.save(destination)

    assert loaded == bundle
    assert destination.read_bytes() == first_bytes
    assert not list(destination.parent.glob("*.tmp"))


def test_bundle_rejects_contract_missing_from_manifest(tmp_path):
    (tmp_path / "captured.json").write_text("{}", encoding="utf-8")
    manifest = ArtifactManifest.capture(run_id="run-001", paths=("captured.json",), root=tmp_path)
    contract = EvidenceContract.create(name="benchmark", required_paths=("missing.json",))

    with pytest.raises(ValueError, match="missing required paths"):
        EvidenceBundle.create(contract=contract, manifest=manifest)


def test_bundle_detects_required_evidence_mutation(tmp_path):
    bundle = _bundle(tmp_path)
    (tmp_path / "results.json").write_text('{"score":2}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        bundle.verify(root=tmp_path)


def test_bundle_ignores_optional_artifact_mutation(tmp_path):
    bundle = _bundle(tmp_path)
    (tmp_path / "optional.log").write_text("changed\n", encoding="utf-8")

    bundle.verify(root=tmp_path)


def test_bundle_rejects_unknown_and_missing_fields(tmp_path):
    payload = json.loads(_bundle(tmp_path).to_json())
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        EvidenceBundle.from_json(json.dumps(payload))

    payload.pop("unexpected")
    payload.pop("manifest")
    with pytest.raises(ValueError, match="missing fields"):
        EvidenceBundle.from_json(json.dumps(payload))


def test_bundle_rejects_invalid_nested_values(tmp_path):
    payload = json.loads(_bundle(tmp_path).to_json())
    payload["contract"] = []

    with pytest.raises(TypeError, match="contract must be an object"):
        EvidenceBundle.from_json(json.dumps(payload))


def test_bundle_rejects_invalid_root_and_schema(tmp_path):
    with pytest.raises(TypeError, match="payload must be a string"):
        EvidenceBundle.from_json(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid JSON"):
        EvidenceBundle.from_json("{")
    with pytest.raises(TypeError, match="root must be an object"):
        EvidenceBundle.from_json("[]")

    payload = json.loads(_bundle(tmp_path).to_json())
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported evidence bundle schema version"):
        EvidenceBundle.from_json(json.dumps(payload))
