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
