import json
import math

import pytest

from experiments.synthetic_negative_transfer import BenchmarkCase
from experiments.synthetic_threshold_ablation import (
    main,
    run_threshold_ablation,
    save_threshold_ablation_evidence,
)


def test_threshold_ablation_measures_routing_tradeoff() -> None:
    cases = [
        BenchmarkCase("small-benefit", 0.65, 0.60),
        BenchmarkCase("harmful", 0.40, 0.80),
    ]

    results = run_threshold_ablation(cases, [0.0, 0.10])

    assert [result.minimum_delta for result in results] == [0.0, 0.10]
    assert results[0].benchmark_result.memory_selected == 1
    assert results[0].benchmark_result.routing_regret == 0.0
    assert results[1].benchmark_result.memory_selected == 0
    assert results[1].benchmark_result.routing_regret == pytest.approx(0.05)
    assert results[1].benchmark_result.negative_transfer_avoidance_rate == 1.0


def test_threshold_ablation_preserves_requested_order() -> None:
    cases = [BenchmarkCase("beneficial", 0.9, 0.6)]

    results = run_threshold_ablation(cases, [0.5, 0.0, 0.2])

    assert [result.minimum_delta for result in results] == [0.5, 0.0, 0.2]


@pytest.mark.parametrize("minimum_deltas", [[], [math.nan], [math.inf], [-math.inf]])
def test_threshold_ablation_rejects_invalid_thresholds(minimum_deltas: list[float]) -> None:
    with pytest.raises(ValueError):
        run_threshold_ablation([], minimum_deltas)


def test_threshold_ablation_rejects_duplicate_thresholds() -> None:
    with pytest.raises(ValueError, match="minimum_deltas must be unique"):
        run_threshold_ablation([], [0.1, 0.1])


def test_save_threshold_ablation_evidence_is_auditable_and_deterministic(tmp_path) -> None:
    cases = [
        BenchmarkCase("beneficial", 0.9, 0.6),
        BenchmarkCase("harmful", 0.4, 0.8),
    ]
    output_path = tmp_path / "ablation" / "thresholds.json"

    save_threshold_ablation_evidence(cases, [0.0, 0.5], output_path)
    first_bytes = output_path.read_bytes()
    payload = json.loads(first_bytes)

    assert payload["ablation"] == "counterfactual_router_minimum_delta"
    assert payload["minimum_deltas"] == [0.0, 0.5]
    assert payload["cases"][1]["case_id"] == "harmful"
    assert payload["results"][0]["benchmark_result"]["routing_regret"] == 0.0
    assert payload["results"][1]["benchmark_result"]["mean_routing_regret"] == pytest.approx(0.15)

    save_threshold_ablation_evidence(cases, [0.0, 0.5], output_path)
    assert output_path.read_bytes() == first_bytes
    assert not list(output_path.parent.glob(".*.tmp"))


def test_threshold_ablation_cli_uses_shared_case_loader_and_persists_evidence(tmp_path) -> None:
    cases_path = tmp_path / "cases.json"
    output_path = tmp_path / "evidence" / "thresholds.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "beneficial",
                    "utility_with_memory": 0.9,
                    "utility_without_memory": 0.6,
                },
                {
                    "case_id": "harmful",
                    "utility_with_memory": 0.4,
                    "utility_without_memory": 0.8,
                },
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
            "--minimum-delta",
            "0.0",
            "--minimum-delta",
            "0.5",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["minimum_deltas"] == [0.0, 0.5]
    assert [case["case_id"] for case in payload["cases"]] == ["beneficial", "harmful"]
    assert len(payload["results"]) == 2


def test_threshold_ablation_cli_rejects_duplicate_thresholds(tmp_path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="minimum_deltas must be unique"):
        main(
            [
                "--cases",
                str(cases_path),
                "--output",
                str(tmp_path / "out.json"),
                "--minimum-delta",
                "0.1",
                "--minimum-delta",
                "0.1",
            ]
        )
