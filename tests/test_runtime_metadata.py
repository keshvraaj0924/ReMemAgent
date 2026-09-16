"""Tests for reproducible runtime metadata artifacts."""

import json

import pytest

from remem.runtime_metadata import RuntimeMetadata


def test_capture_produces_verified_current_runtime_metadata() -> None:
    metadata = RuntimeMetadata.capture()
    metadata.verify()
    assert metadata.matches_current_runtime()
    assert metadata.schema_version == 1


def test_runtime_metadata_json_round_trip_is_stable() -> None:
    metadata = RuntimeMetadata.capture()
    restored = RuntimeMetadata.from_json(metadata.to_json())
    assert restored == metadata
    assert restored.to_json() == metadata.to_json()


def test_runtime_metadata_fingerprint_is_stable_and_sensitive() -> None:
    metadata = RuntimeMetadata.capture()
    restored = RuntimeMetadata.from_json(metadata.to_json())
    assert restored.fingerprint() == metadata.fingerprint()
    assert len(metadata.fingerprint()) == 64

    changed = RuntimeMetadata(
        schema_version=metadata.schema_version,
        python_version="0.0.0",
        python_implementation=metadata.python_implementation,
        operating_system=metadata.operating_system,
        operating_system_release=metadata.operating_system_release,
        machine=metadata.machine,
    )
    assert changed.fingerprint() != metadata.fingerprint()
    assert not changed.matches_current_runtime()


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ("not-json", ValueError, "valid JSON"),
        ("[]", TypeError, "root must be an object"),
        ('{"schema_version":1}', ValueError, "missing fields"),
    ],
)
def test_runtime_metadata_loading_rejects_invalid_schema(
    payload: str, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        RuntimeMetadata.from_json(payload)


def test_runtime_metadata_loading_rejects_unknown_fields() -> None:
    payload = json.loads(RuntimeMetadata.capture().to_json())
    payload["hostname"] = "must-not-be-recorded"
    with pytest.raises(ValueError, match="unknown fields"):
        RuntimeMetadata.from_json(json.dumps(payload))


def test_runtime_metadata_rejects_unsupported_schema_version() -> None:
    metadata = RuntimeMetadata.capture()
    invalid = RuntimeMetadata(
        schema_version=999,
        python_version=metadata.python_version,
        python_implementation=metadata.python_implementation,
        operating_system=metadata.operating_system,
        operating_system_release=metadata.operating_system_release,
        machine=metadata.machine,
    )
    with pytest.raises(ValueError, match="unsupported runtime metadata version"):
        invalid.verify()


def test_runtime_metadata_rejects_empty_required_field() -> None:
    metadata = RuntimeMetadata.capture()
    invalid = RuntimeMetadata(
        schema_version=metadata.schema_version,
        python_version=" ",
        python_implementation=metadata.python_implementation,
        operating_system=metadata.operating_system,
        operating_system_release=metadata.operating_system_release,
        machine=metadata.machine,
    )
    with pytest.raises(ValueError, match="python_version must not be empty"):
        invalid.verify()
