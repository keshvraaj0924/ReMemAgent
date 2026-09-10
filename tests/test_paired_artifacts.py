from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.paired_artifacts import (
    PAIRED_EXECUTION_ORDER_PROVENANCE_KEY,
    save_paired_execution_result,
)
from experiments.paired_benchmark import PairedBenchmarkResult, PairedSeedExecution


def _result(
    execution_order: tuple[PairedSeedExecution, ...] | None = None,
) -> PairedBenchmarkResult:
    return PairedBenchmarkResult(
        baseline_reports=(),
        treatment_reports=(),
        execution_order=execution_order
        or (
            PairedSeedExecution(
                seed=11,
                first_condition="baseline",
                second_condition="treatment",
            ),
            PairedSeedExecution(
                seed=17,
                first_condition="treatment",
                second_condition="baseline",
            ),
        ),
        comparison=SimpleNamespace(seeds=(11, 17)),  # type: ignore[arg-type]
    )


def test_save_paired_execution_result_persists_temporal_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_provenance: dict[str, object] = {}

    def fake_save(
        baseline_reports,
        treatment_reports,
        comparison,
        output_path,
        *,
        runtime_provenance,
    ):
        del baseline_reports, treatment_reports, comparison
        captured_provenance.update(runtime_provenance)
        output_path.write_text(
            json.dumps(
                {
                    "experiment_identity": "paired-test-identity",
                    "runtime_provenance": runtime_provenance,
                }
            ),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr("experiments.paired_artifacts.save_paired_benchmark_result", fake_save)

    output_path = save_paired_execution_result(
        _result(),
        tmp_path / "paired.json",
        runtime_provenance={"code_revision": "abc123"},
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["execution_order"] == [
        {
            "seed": 11,
            "first_condition": "baseline",
            "second_condition": "treatment",
        },
        {
            "seed": 17,
            "first_condition": "treatment",
            "second_condition": "baseline",
        },
    ]
    digest = captured_provenance[PAIRED_EXECUTION_ORDER_PROVENANCE_KEY]
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert persisted["runtime_provenance"][PAIRED_EXECUTION_ORDER_PROVENANCE_KEY] == digest
    assert persisted["runtime_provenance"]["code_revision"] == "abc123"


def test_save_paired_execution_result_rejects_seed_mismatch_before_persistence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _result(
        (
            PairedSeedExecution(11, "baseline", "treatment"),
            PairedSeedExecution(19, "treatment", "baseline"),
        )
    )
    monkeypatch.setattr(
        "experiments.paired_artifacts.save_paired_benchmark_result",
        lambda *args, **kwargs: pytest.fail("persistence must not start"),
    )

    with pytest.raises(ValueError, match="seeds must exactly match"):
        save_paired_execution_result(invalid, tmp_path / "paired.json")


def test_save_paired_execution_result_rejects_non_counterbalanced_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _result(
        (
            PairedSeedExecution(11, "baseline", "treatment"),
            PairedSeedExecution(17, "baseline", "treatment"),
        )
    )
    monkeypatch.setattr(
        "experiments.paired_artifacts.save_paired_benchmark_result",
        lambda *args, **kwargs: pytest.fail("persistence must not start"),
    )

    with pytest.raises(ValueError, match="deterministic counterbalanced"):
        save_paired_execution_result(invalid, tmp_path / "paired.json")


def test_save_paired_execution_result_rejects_reserved_provenance_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="reserves"):
        save_paired_execution_result(
            _result(),
            tmp_path / "paired.json",
            runtime_provenance={PAIRED_EXECUTION_ORDER_PROVENANCE_KEY: "caller-value"},
        )
