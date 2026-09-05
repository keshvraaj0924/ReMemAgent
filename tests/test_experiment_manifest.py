import json

import pytest

from remem.reproducibility import ExperimentManifest


def test_manifest_is_canonical_and_order_independent() -> None:
    first = ExperimentManifest({"seed": 7, "model": {"name": "demo", "temperature": 0.0}})
    second = ExperimentManifest({"model": {"temperature": 0.0, "name": "demo"}, "seed": 7})

    assert first.to_json() == second.to_json()
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_manifest_values_are_detached() -> None:
    source = {"nested": {"items": [1, 2]}}
    manifest = ExperimentManifest(source)
    source["nested"]["items"].append(3)

    assert manifest.values == {"nested": {"items": [1, 2]}}
    assert json.loads(manifest.to_json()) == manifest.values


def test_manifest_rejects_non_finite_float() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            ExperimentManifest({"metric": value})


def test_manifest_rejects_unsupported_values() -> None:
    with pytest.raises(TypeError, match="JSON-compatible"):
        ExperimentManifest({"object": object()})


def test_manifest_rejects_empty_keys() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        ExperimentManifest({"": 1})


def test_manifest_from_json_normalizes_non_canonical_formatting() -> None:
    manifest = ExperimentManifest.from_json('{ "b": 2, "a": [1, true] }')

    assert manifest.values == {"a": [1, True], "b": 2}
    assert manifest.to_json() == '{"a":[1,true],"b":2}'


def test_manifest_from_json_rejects_invalid_or_non_object_payloads() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ExperimentManifest.from_json("{not-json}")
    with pytest.raises(ValueError, match="object"):
        ExperimentManifest.from_json("[1, 2, 3]")


def test_manifest_from_json_rejects_non_string_payload() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        ExperimentManifest.from_json(123)  # type: ignore[arg-type]


def test_manifest_verify_sha256_accepts_matching_digest() -> None:
    manifest = ExperimentManifest({"seed": 11})

    manifest.verify_sha256(manifest.sha256)


def test_manifest_verify_sha256_rejects_mismatched_digest() -> None:
    manifest = ExperimentManifest({"seed": 11})

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        manifest.verify_sha256("0" * 64)


def test_manifest_verify_sha256_rejects_empty_digest() -> None:
    manifest = ExperimentManifest({"seed": 11})

    with pytest.raises(ValueError, match="non-empty string"):
        manifest.verify_sha256("")
