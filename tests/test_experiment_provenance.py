"""Tests for experiment provenance integrity and replay checks."""

import json
from unittest.mock import patch

import pytest

from remem.experiment_provenance import ExperimentProvenance
from remem.reproducibility import ReproducibilityConfig


def _provenance() -> ExperimentProvenance:
    manifest = ReproducibilityConfig(base_seed=42).create_manifest(
        [("environment", 0), ("policy", 0)]
    )
    return ExperimentProvenance.capture(manifest)


def test_capture_binds_manifest_and_runtime() -> None:
    provenance = _provenance()
    provenance.verify()
    assert provenance.schema_version == 1
    assert provenance.run_id.startswith("v1-")
    assert len(provenance.runtime_fingerprint) == 64
    assert provenance.runtime.matches_current_runtime()


def test_json_round_trip_is_stable() -> None:
    provenance = _provenance()
    restored = ExperimentProvenance.from_json(provenance.to_json())
    assert restored == provenance
    assert restored.to_json() == provenance.to_json()


def test_save_and_load_round_trip(tmp_path) -> None:
    provenance = _provenance()
    path = tmp_path / "provenance.json"
    provenance.save(path)
    assert path.read_text(encoding="utf-8") == provenance.to_json()
    assert ExperimentProvenance.load(path) == provenance


def test_save_syncs_parent_directory_after_replace(tmp_path) -> None:
    path = tmp_path / "provenance.json"
    with patch("remem.experiment_provenance._sync_directory") as sync_directory:
        _provenance().save(path)
    sync_directory.assert_called_once_with(tmp_path)


def test_save_does_not_sync_parent_when_replace_fails(tmp_path) -> None:
    path = tmp_path / "provenance.json"
    with (
        patch("remem.experiment_provenance.os.replace", side_effect=OSError("replace failed")),
        patch("remem.experiment_provenance._sync_directory") as sync_directory,
        pytest.raises(OSError, match="replace failed"),
    ):
        _provenance().save(path)
    sync_directory.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_save_leaves_no_temporary_artifacts(tmp_path) -> None:
    path = tmp_path / "provenance.json"
    _provenance().save(path)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["provenance.json"]


def test_save_requires_existing_parent(tmp_path) -> None:
    path = tmp_path / "missing" / "provenance.json"
    with pytest.raises(FileNotFoundError, match="provenance parent does not exist"):
        _provenance().save(path)


def test_invalid_provenance_cannot_overwrite_existing_artifact(tmp_path) -> None:
    path = tmp_path / "provenance.json"
    provenance = _provenance()
    provenance.save(path)
    original_payload = path.read_text(encoding="utf-8")

    invalid = ExperimentProvenance(
        schema_version=provenance.schema_version,
        run_id="v1-" + "0" * 64,
        runtime_fingerprint=provenance.runtime_fingerprint,
        manifest=provenance.manifest,
        runtime=provenance.runtime,
    )
    with pytest.raises(ValueError, match="run_id does not match"):
        invalid.save(path)
    assert path.read_text(encoding="utf-8") == original_payload


def test_replay_compatible_for_current_runtime() -> None:
    _provenance().assert_replay_compatible()


def test_tampered_run_id_is_rejected() -> None:
    payload = json.loads(_provenance().to_json())
    payload["run_id"] = "v1-" + "0" * 64
    with pytest.raises(ValueError, match="run_id does not match"):
        ExperimentProvenance.from_json(json.dumps(payload))


def test_tampered_runtime_fingerprint_is_rejected() -> None:
    payload = json.loads(_provenance().to_json())
    payload["runtime_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="runtime_fingerprint does not match"):
        ExperimentProvenance.from_json(json.dumps(payload))


def test_tampered_manifest_seed_is_rejected() -> None:
    payload = json.loads(_provenance().to_json())
    payload["manifest"]["assignments"][0]["seed"] += 1
    with pytest.raises(ValueError, match="seed mismatch"):
        ExperimentProvenance.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ("not-json", ValueError, "valid JSON"),
        ("[]", TypeError, "root must be an object"),
        ('{"schema_version":1}', ValueError, "missing fields"),
    ],
)
def test_loading_rejects_invalid_schema(
    payload: str, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        ExperimentProvenance.from_json(payload)


def test_loading_rejects_unknown_fields() -> None:
    payload = json.loads(_provenance().to_json())
    payload["hostname"] = "not-portable"
    with pytest.raises(ValueError, match="unknown fields"):
        ExperimentProvenance.from_json(json.dumps(payload))
