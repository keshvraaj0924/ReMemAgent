from __future__ import annotations

from pathlib import Path

import pytest

from experiments.benchmark_cli import _persist_benchmark_report


def _write_report(path: Path, content: str) -> None:
    """Write deterministic test content to a private publication path."""

    path.write_text(content, encoding="utf-8")


def test_persist_benchmark_report_preserves_concurrent_artifact_by_default(tmp_path) -> None:
    output_path = tmp_path / "report.json"

    def writer(temporary_path: Path) -> None:
        _write_report(temporary_path, "new artifact\n")
        output_path.write_text("concurrent artifact\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        _persist_benchmark_report(output_path, overwrite=False, writer=writer)

    assert output_path.read_text(encoding="utf-8") == "concurrent artifact\n"
    assert not tuple(tmp_path.glob(".report.json.*.publish.tmp"))


def test_persist_benchmark_report_replaces_concurrent_artifact_when_explicit(tmp_path) -> None:
    output_path = tmp_path / "report.json"

    def writer(temporary_path: Path) -> None:
        _write_report(temporary_path, "new artifact\n")
        output_path.write_text("concurrent artifact\n", encoding="utf-8")

    persisted_path = _persist_benchmark_report(output_path, overwrite=True, writer=writer)

    assert persisted_path == output_path
    assert output_path.read_text(encoding="utf-8") == "new artifact\n"
    assert not tuple(tmp_path.glob(".report.json.*.publish.tmp"))


def test_persist_benchmark_report_cleans_private_file_when_writer_fails(tmp_path) -> None:
    output_path = tmp_path / "report.json"

    def writer(temporary_path: Path) -> None:
        _write_report(temporary_path, "partial artifact\n")
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        _persist_benchmark_report(output_path, overwrite=False, writer=writer)

    assert not output_path.exists()
    assert not tuple(tmp_path.glob(".report.json.*.publish.tmp"))
