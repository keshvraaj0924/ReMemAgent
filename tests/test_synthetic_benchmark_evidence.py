import json

import pytest

from experiments.synthetic_benchmark_evidence import load_verified_synthetic_benchmark_evidence
from experiments.synthetic_negative_transfer import BenchmarkCase, save_benchmark_evidence
from remem.routing.counterfactual import CounterfactualRouter


def _write_valid_evidence(tmp_path):
    path = tmp_path / "benchmark.json"
    cases = [
        BenchmarkCase("helps", 0.9, 0.4),
        BenchmarkCase("hurts", 0.2, 0.8),
        BenchmarkCase("small_gain", 0.61, 0.60),
    ]
    save_benchmark_evidence(cases, CounterfactualRouter(minimum_delta=0.05), path)
    return path


def test_load_verified_evidence_replays_benchmark(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)

    verified = load_verified_synthetic_benchmark_evidence(path)

    assert [case.case_id for case in verified.cases] == ["helps", "hurts", "small_gain"]
    assert verified.minimum_delta == 0.05
    assert verified.result.total_cases == 3
    assert verified.result.memory_selected == 1


def test_load_verified_evidence_rejects_tampered_aggregate(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["memory_selected"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic replay"):
        load_verified_synthetic_benchmark_evidence(path)


def test_load_verified_evidence_rejects_inconsistent_derived_metric(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["negative_transfer_rate"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="derived field negative_transfer_rate"):
        load_verified_synthetic_benchmark_evidence(path)


def test_load_verified_evidence_rejects_tampered_case(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["utility_with_memory"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic replay"):
        load_verified_synthetic_benchmark_evidence(path)


def test_load_verified_evidence_rejects_invalid_router_schema(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["router"]["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="router has an unexpected schema"):
        load_verified_synthetic_benchmark_evidence(path)


def test_load_verified_evidence_rejects_non_finite_router_threshold(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["router"]["minimum_delta"] = float("inf")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum_delta must be finite"):
        load_verified_synthetic_benchmark_evidence(path)


def test_load_verified_evidence_rejects_duplicate_case_ids(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="case_id values must be unique"):
        load_verified_synthetic_benchmark_evidence(path)


def test_load_verified_evidence_rejects_unexpected_top_level_schema(tmp_path) -> None:
    path = _write_valid_evidence(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["extra"] = "untrusted"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected schema"):
        load_verified_synthetic_benchmark_evidence(path)


def test_load_verified_evidence_rejects_malformed_json(tmp_path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="unable to load synthetic benchmark evidence"):
        load_verified_synthetic_benchmark_evidence(path)
