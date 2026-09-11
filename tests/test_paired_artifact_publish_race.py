from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.paired_artifacts import save_paired_execution_result
from experiments.paired_benchmark import PairedBenchmarkResult, PairedSeedExecution


def _result() -> PairedBenchmarkResult:
    return PairedBenchmarkResult(
        baseline_reports=(),
        treatment_reports=(),
        execution_order=(
            PairedSeedExecution(
                seed=11,
                first_condition="baseline",
                second_condition="treatment",
            ),
        ),
        comparison=SimpleNamespace(seeds=(11,)),  # type: ignore[arg-type]
    )


def _install_racing_serializer(
    monkeypatch: pytest.MonkeyPatch,
    final_path,
    *,
    competing_content: str,
) -> None:
    def fake_save(
        baseline_reports,
        treatment_reports,
        comparison,
        output_path,
        *,
        runtime_provenance,
    ):
        del baseline_reports, treatment_reports, comparison
        output_path.write_text(
            json.dumps(
                {
                    "experiment_identity": "paired-test-identity",
                    "runtime_provenance": runtime_provenance,
                }
            ),
            encoding="utf-8",
        )
        final_path.write_text(competing_content, encoding="utf-8")
        return output_path

    monkeypatch.setattr("experiments.paired_artifacts.save_paired_benchmark_result", fake_save)


def test_paired_artifact_publish_preserves_file_created_during_measurement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "paired.json"
    _install_racing_serializer(
        monkeypatch,
        output_path,
        competing_content="concurrent artifact\n",
    )

    with pytest.raises(FileExistsError, match="paired benchmark artifact already exists"):
        save_paired_execution_result(_result(), output_path)

    assert output_path.read_text(encoding="utf-8") == "concurrent artifact\n"
    assert not tuple(tmp_path.glob(".paired.json.paired.*.tmp"))


def test_paired_artifact_publish_replaces_racing_file_only_with_explicit_overwrite(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "paired.json"
    _install_racing_serializer(
        monkeypatch,
        output_path,
        competing_content="concurrent artifact\n",
    )

    saved_path = save_paired_execution_result(
        _result(),
        output_path,
        overwrite=True,
    )

    persisted = json.loads(saved_path.read_text(encoding="utf-8"))
    assert persisted["execution_order"] == [
        {
            "seed": 11,
            "first_condition": "baseline",
            "second_condition": "treatment",
        }
    ]
    assert not tuple(tmp_path.glob(".paired.json.paired.*.tmp"))
