import json

import pytest

from experiments.synthetic_ablation_evidence import load_verified_threshold_ablation_evidence
from experiments.synthetic_negative_transfer import BenchmarkCase
from experiments.synthetic_threshold_ablation import save_threshold_ablation_evidence


def _write_valid_evidence(tmp_path):
    output_path = tmp_path / "thresholds.json"
    cases = [
        BenchmarkCase("beneficial", 0.9, 0.6),
        BenchmarkCase("harmful", 0.4, 0.8),
    ]
    save_threshold_ablation_evidence(cases, [0.0, 0.5], output_path)
    return output_path


def test_load_verified_threshold_ablation_evidence_replays_stored_results(tmp_path) -> None:
    evidence_path = _write_valid_evidence(tmp_path)

    evidence = load_verified_threshold_ablation_evidence(evidence_path)

    assert [case.case_id for case in evidence.cases] == ["beneficial", "harmful"]
    assert evidence.minimum_deltas == (0.0, 0.5)
    assert len(evidence.results) == 2
    assert evidence.results[0].benchmark_result.routing_regret == 0.0
    assert evidence.results[1].benchmark_result.mean_routing_regret == pytest.approx(0.15)


def test_verified_threshold_ablation_rejects_mutated_measured_output(tmp_path) -> None:
    evidence_path = _write_valid_evidence(tmp_path)
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["results"][0]["benchmark_result"]["routing_regret"] = 0.25
    payload["results"][0]["benchmark_result"]["mean_routing_regret"] = 0.125
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match deterministic replay"):
        load_verified_threshold_ablation_evidence(evidence_path)


def test_verified_threshold_ablation_rejects_mutated_derived_metric(tmp_path) -> None:
    evidence_path = _write_valid_evidence(tmp_path)
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["results"][0]["benchmark_result"]["negative_transfer_rate"] = 0.25
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="derived field negative_transfer_rate"):
        load_verified_threshold_ablation_evidence(evidence_path)


def test_verified_threshold_ablation_rejects_threshold_result_mismatch(tmp_path) -> None:
    evidence_path = _write_valid_evidence(tmp_path)
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["results"][0]["minimum_delta"] = 0.25
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match deterministic replay"):
        load_verified_threshold_ablation_evidence(evidence_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.update({"ablation": "wrong"}),
        lambda payload: payload.update({"minimum_deltas": [0.0, 0.0]}),
        lambda payload: payload.update({"minimum_deltas": []}),
    ],
)
def test_verified_threshold_ablation_rejects_invalid_envelope(tmp_path, mutation) -> None:
    evidence_path = _write_valid_evidence(tmp_path)
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    mutation(payload)
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_verified_threshold_ablation_evidence(evidence_path)


def test_verified_threshold_ablation_rejects_non_object_json(tmp_path) -> None:
    evidence_path = tmp_path / "thresholds.json"
    evidence_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_verified_threshold_ablation_evidence(evidence_path)
