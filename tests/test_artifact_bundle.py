from __future__ import annotations

from pathlib import Path

import pytest

from experiments.artifact_bundle import (
    PreparedArtifact,
    create_private_artifact_path,
    publish_artifact_bundle,
)


def _prepare(destination: Path, content: str) -> PreparedArtifact:
    temporary_path = create_private_artifact_path(destination)
    temporary_path.write_text(content, encoding="utf-8")
    return PreparedArtifact(temporary_path, destination)


def test_bundle_rolls_back_prior_publication_on_no_overwrite_collision(tmp_path) -> None:
    manifest = _prepare(tmp_path / "manifest.json", "new manifest\n")
    report = _prepare(tmp_path / "report.json", "new report\n")
    report.destination_path.write_text("concurrent report\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        publish_artifact_bundle((manifest, report), overwrite=False)

    assert not manifest.destination_path.exists()
    assert report.destination_path.read_text(encoding="utf-8") == "concurrent report\n"
    assert not tuple(tmp_path.glob("*.bundle.tmp"))
    assert not tuple(tmp_path.glob(".*.bundle.tmp"))


def test_bundle_overwrite_restores_previous_files_when_later_publish_fails(
    tmp_path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text("old manifest\n", encoding="utf-8")
    report_path.write_text("old report\n", encoding="utf-8")
    manifest = _prepare(manifest_path, "new manifest\n")
    report = _prepare(report_path, "new report\n")

    import experiments.artifact_bundle as artifact_bundle

    original_replace = artifact_bundle.os.replace
    replace_calls = 0

    def fail_second_publication(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        if source in {manifest.temporary_path, report.temporary_path}:
            replace_calls += 1
            if replace_calls == 2:
                raise OSError("simulated publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(artifact_bundle.os, "replace", fail_second_publication)

    with pytest.raises(OSError, match="simulated publish failure"):
        publish_artifact_bundle((manifest, report), overwrite=True)

    assert manifest_path.read_text(encoding="utf-8") == "old manifest\n"
    assert report_path.read_text(encoding="utf-8") == "old report\n"
    assert not tuple(tmp_path.glob(".*.bundle.tmp"))


def test_bundle_publishes_primary_commit_marker_last(tmp_path, monkeypatch) -> None:
    observability = _prepare(tmp_path / "observability.json", "observability\n")
    manifest = _prepare(tmp_path / "manifest.json", "manifest\n")
    report = _prepare(tmp_path / "report.json", "report\n")

    import experiments.artifact_bundle as artifact_bundle

    original_link = artifact_bundle.os.link
    published: list[Path] = []

    def record_link(source: Path, destination: Path) -> None:
        original_link(source, destination)
        published.append(destination)

    monkeypatch.setattr(artifact_bundle.os, "link", record_link)

    publish_artifact_bundle((observability, manifest, report), overwrite=False)

    assert published == [
        observability.destination_path,
        manifest.destination_path,
        report.destination_path,
    ]
    assert report.destination_path.read_text(encoding="utf-8") == "report\n"


def test_bundle_rejects_duplicate_destinations_before_publication(tmp_path) -> None:
    destination = tmp_path / "report.json"
    first = _prepare(destination, "first\n")
    second = _prepare(destination, "second\n")

    with pytest.raises(ValueError, match="duplicate artifact destination"):
        publish_artifact_bundle((first, second), overwrite=False)

    assert not destination.exists()
    first.temporary_path.unlink(missing_ok=True)
    second.temporary_path.unlink(missing_ok=True)
