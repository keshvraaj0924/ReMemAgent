"""Tests for provenance-bound observability evidence."""

import json
from pathlib import Path

import pytest

import remem.provenanced_observability as provenanced_observability_module
from remem.experiment_provenance import ExperimentProvenance
from remem.observability import MetricsRecorder
from remem.observability_artifact import ObservabilityArtifact
from remem.provenanced_observability import ProvenancedObservability
from remem.reproducibility import ReproducibilityConfig


def _bundle() -> ProvenancedObservability:
    manifest = ReproducibilityConfig(base_seed=42).create_manifest([("environment", 0)])
    provenance = ExperimentProvenance.capture(manifest)
    recorder = MetricsRecorder()
    recorder.increment("episodes", 2)
    recorder.observe_duration("episode", 0.5)
    observability = ObservabilityArtifact.create(recorder.snapshot())
    return ProvenancedObservability.create(
        provenance=provenance,
        observability=observability,
    )


def test_provenanced_observability_round_trip_is_deterministic() -> None:
    bundle = _bundle()

    restored = ProvenancedObservability.from_json(bundle.to_json())

    assert restored == bundle
    assert restored.to_json() == bundle.to_json()
    assert restored.run_id == restored.provenance.run_id


def test_provenanced_observability_persists_and_loads_verified_bundle(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    destination = tmp_path / "telemetry.json"

    saved_path = bundle.save(destination)
    restored = ProvenancedObservability.load(destination)

    assert saved_path == destination
    assert restored == bundle
    assert destination.read_text(encoding="utf-8") == f"{bundle.to_json()}\n"
    assert list(tmp_path.glob(".telemetry.json.*.tmp")) == []


def test_provenanced_observability_save_cleans_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "telemetry.json"

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(provenanced_observability_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _bundle().save(destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".telemetry.json.*.tmp")) == []


def test_provenanced_observability_load_rejects_persisted_tampering(
    tmp_path: Path,
) -> None:
    destination = _bundle().save(tmp_path / "telemetry.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["observability"]["snapshot"]["counters"]["episodes"] = 999
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot digest mismatch"):
        ProvenancedObservability.load(destination)


def test_provenanced_observability_save_requires_existing_parent(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "missing" / "telemetry.json"

    with pytest.raises(FileNotFoundError, match="parent directory does not exist"):
        _bundle().save(destination)


def test_provenanced_observability_rejects_detached_run_id() -> None:
    payload = json.loads(_bundle().to_json())
    payload["run_id"] = "v1-detached"

    with pytest.raises(ValueError, match="run_id does not match"):
        ProvenancedObservability.from_json(json.dumps(payload))


def test_provenanced_observability_rejects_metric_tampering() -> None:
    payload = json.loads(_bundle().to_json())
    payload["observability"]["snapshot"]["counters"]["episodes"] = 999

    with pytest.raises(ValueError, match="snapshot digest mismatch"):
        ProvenancedObservability.from_json(json.dumps(payload))


def test_provenanced_observability_rejects_provenance_tampering() -> None:
    payload = json.loads(_bundle().to_json())
    payload["provenance"]["run_id"] = "v1-detached"

    with pytest.raises(ValueError, match="run_id does not match reproducibility manifest"):
        ProvenancedObservability.from_json(json.dumps(payload))


def test_provenanced_observability_rejects_bundle_digest_tampering() -> None:
    payload = json.loads(_bundle().to_json())
    payload["bundle_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="digest mismatch"):
        ProvenancedObservability.from_json(json.dumps(payload))


def test_provenanced_observability_rejects_schema_drift() -> None:
    payload = json.loads(_bundle().to_json())
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="fields do not match schema"):
        ProvenancedObservability.from_json(json.dumps(payload))


@pytest.mark.parametrize("schema_version", [True, "1"])
def test_provenanced_observability_requires_integer_schema_version(
    schema_version: object,
) -> None:
    payload = json.loads(_bundle().to_json())
    payload["schema_version"] = schema_version

    with pytest.raises(TypeError, match="schema_version must be an integer"):
        ProvenancedObservability.from_json(json.dumps(payload))


def test_provenanced_observability_rejects_unsupported_schema_version() -> None:
    payload = json.loads(_bundle().to_json())
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported provenanced observability version"):
        ProvenancedObservability.from_json(json.dumps(payload))


def test_provenanced_observability_rejects_non_string_bundle_digest() -> None:
    payload = json.loads(_bundle().to_json())
    payload["bundle_sha256"] = 42

    with pytest.raises(TypeError, match="bundle_sha256 must be a string"):
        ProvenancedObservability.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ("not-json", ValueError, "must contain valid JSON"),
        ("[]", TypeError, "JSON root must be an object"),
    ],
)
def test_provenanced_observability_rejects_malformed_payloads(
    payload: str,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ProvenancedObservability.from_json(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provenance", [], "provenance must be an object"),
        ("observability", [], "observability must be an object"),
    ],
)
def test_provenanced_observability_requires_object_components(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(_bundle().to_json())
    payload[field] = value

    with pytest.raises(TypeError, match=message):
        ProvenancedObservability.from_json(json.dumps(payload))


def test_provenanced_observability_validates_component_types() -> None:
    bundle = _bundle()
    with pytest.raises(TypeError, match="ExperimentProvenance"):
        ProvenancedObservability.create(  # type: ignore[arg-type]
            provenance={}, observability=bundle.observability
        )
    with pytest.raises(TypeError, match="ObservabilityArtifact"):
        ProvenancedObservability.create(  # type: ignore[arg-type]
            provenance=bundle.provenance, observability={}
        )
