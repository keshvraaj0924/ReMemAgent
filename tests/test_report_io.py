"""Regression tests for durable experiment report persistence."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import call

import pytest

from experiments import report_io


def test_atomic_write_json_persists_deterministic_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "report.json"

    result = report_io.atomic_write_json(output_path, {"z": 2, "a": 1})

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "z": 2\n}\n'
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX-specific")
def test_atomic_write_json_fsyncs_parent_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "report.json"
    events: list[str] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def tracked_replace(source: str | Path, destination: str | Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    def tracked_fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(report_io.os, "replace", tracked_replace)
    monkeypatch.setattr(report_io.os, "fsync", tracked_fsync)

    report_io.atomic_write_json(output_path, {"value": 1})

    assert events == ["fsync", "replace", "fsync"]


def test_atomic_write_json_cleans_temporary_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "report.json"

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(report_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        report_io.atomic_write_json(output_path, {"value": 1})

    assert not output_path.exists()
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_fsync_directory_is_noop_off_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_calls: list[object] = []
    monkeypatch.setattr(report_io.os, "name", "nt")
    monkeypatch.setattr(report_io.os, "open", lambda *args: open_calls.append(call(*args)))

    report_io._fsync_directory(tmp_path)

    assert open_calls == []
