from __future__ import annotations

import json

import pytest

from experiments.synthetic_ablation_evidence import VerifiedThresholdAblationEvidence
from experiments.synthetic_ablation_report import (
    build_verified_threshold_ablation_report,
    main,
    write_verified_threshold_ablation_report,
)
from experiments.synthetic_negative_transfer import BenchmarkCase, BenchmarkResult
from experiments.synthetic_threshold_ablation import (
    ThresholdAblationResult,
    save_threshold_ablation_evidence,
)


def _cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase("helps", utility_with_memory=0.9, utility_without_memory=0.3),
        BenchmarkCase("hurts", utility_with_memory=0.2, utility_without_memory=0.8),
        BenchmarkCase("small_gain", utility_with_memory=0.55, utility_without_memory=0.5),
    ]


def test_verified_report_is_derived_from_replayed_evidence(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.json"
    report_path = tmp_path / "report.json"
    save_threshold_ablation_evidence(_cases(), [0.0, 0.1], evidence_path)

    written_path = write_verified_threshold_ablation_report(evidence_path, report_path)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert payload["ablation"] == "counterfactual_router_minimum_delta"
    assert payload["case_count"] == 3
    assert payload["negative_transfer_cases"] == 1
    assert [record["minimum_delta"] for record in payload["thresholds"]] == [0.0, 0.1]
    assert all(record["routing_regret"] >= 0.0 for record in payload["thresholds"])


def test_cli_writes_verified_report(tmp_path) -> None:
    evidence_path = tmp_path / "evidence.json"
    report_path = tmp_path / "report.json"
    save_threshold_ablation_evidence(_cases(), [0.0], evidence_path)

    exit_code = main(["--evidence", str(evidence_path), "--output", str(report_path)])

    assert exit_code == 0
    assert report_path.exists()


def test_report_rejects_incomparable_threshold_results() -> None:
    cases = tuple(_cases())
    inconsistent_result = BenchmarkResult(
        total_cases=3,
        memory_selected=1,
        self_reasoning_selected=2,
        negative_transfer_cases=2,
        selected_negative_transfer_cases=0,
        avoided_negative_transfer_cases=2,
        routing_regret=0.0,
    )
    evidence = VerifiedThresholdAblationEvidence(
        cases=cases,
        minimum_deltas=(0.0, 0.1),
        results=(
            ThresholdAblationResult(
                minimum_delta=0.0,
                benchmark_result=BenchmarkResult(
                    total_cases=3,
                    memory_selected=2,
                    self_reasoning_selected=1,
                    negative_transfer_cases=1,
                    selected_negative_transfer_cases=0,
                    avoided_negative_transfer_cases=1,
                    routing_regret=0.0,
                ),
            ),
            ThresholdAblationResult(minimum_delta=0.1, benchmark_result=inconsistent_result),
        ),
    )

    with pytest.raises(ValueError, match="same negative-transfer cases"):
        build_verified_threshold_ablation_report(evidence)


def test_report_rejects_empty_verified_results() -> None:
    evidence = VerifiedThresholdAblationEvidence(
        cases=tuple(_cases()),
        minimum_deltas=(),
        results=(),
    )

    with pytest.raises(ValueError, match="must contain results"):
        build_verified_threshold_ablation_report(evidence)
