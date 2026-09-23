import json

import pytest

from remem.observability import MetricsRecorder
from remem.observability_artifact import ObservabilityArtifact


def _artifact() -> ObservabilityArtifact:
    recorder = MetricsRecorder()
    recorder.increment("episodes", 3)
    recorder.increment("memory.accepted", 2)
    recorder.observe_duration("episode", 1.25)
    return ObservabilityArtifact.create(recorder.snapshot())


def test_observability_artifact_round_trip_is_deterministic() -> None:
    artifact = _artifact()

    restored = ObservabilityArtifact.from_json(artifact.to_json())

    assert restored == artifact
    assert restored.to_json() == artifact.to_json()


def test_observability_artifact_persists_and_loads(tmp_path) -> None:
    artifact = _artifact()
    destination = tmp_path / "metrics.json"

    assert artifact.save(destination) == destination

    assert ObservabilityArtifact.load(destination) == artifact
    assert not list(tmp_path.glob(".metrics.json.*.tmp"))


def test_observability_artifact_rejects_snapshot_tampering() -> None:
    payload = json.loads(_artifact().to_json())
    payload["snapshot"]["counters"]["episodes"] = 99

    with pytest.raises(ValueError, match="digest mismatch"):
        ObservabilityArtifact.from_json(json.dumps(payload))


def test_observability_artifact_rejects_schema_drift() -> None:
    payload = json.loads(_artifact().to_json())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="fields do not match schema"):
        ObservabilityArtifact.from_json(json.dumps(payload))

    payload = json.loads(_artifact().to_json())
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported observability schema version"):
        ObservabilityArtifact.from_json(json.dumps(payload))


def test_observability_artifact_rejects_malformed_payloads() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        ObservabilityArtifact.from_json("{")
    with pytest.raises(TypeError, match="JSON root"):
        ObservabilityArtifact.from_json("[]")


def test_observability_artifact_requires_existing_parent(tmp_path) -> None:
    destination = tmp_path / "missing" / "metrics.json"

    with pytest.raises(FileNotFoundError, match="parent directory does not exist"):
        _artifact().save(destination)


def test_observability_artifact_rejects_invalid_snapshot_at_creation() -> None:
    with pytest.raises(TypeError, match="MetricSnapshot"):
        ObservabilityArtifact.create({})  # type: ignore[arg-type]
