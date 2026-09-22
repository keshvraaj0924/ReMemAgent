import json

import pytest

from remem.observability import MetricSnapshot, MetricsRecorder
from remem.reporting import CounterRateSpec, ExperimentSummary


def test_experiment_summary_derives_deterministic_evidence() -> None:
    recorder = MetricsRecorder()
    recorder.increment("retrieval.total", 4)
    recorder.increment("retrieval.accepted", 3)
    recorder.observe_duration("retrieval", 0.2)
    recorder.observe_duration("retrieval", 0.4)

    summary = ExperimentSummary.from_snapshot(
        recorder.snapshot(),
        counter_rates=(
            CounterRateSpec(
                name="retrieval.acceptance_rate",
                numerator="retrieval.accepted",
                denominator="retrieval.total",
            ),
        ),
    )

    assert summary.counters == {"retrieval.accepted": 3, "retrieval.total": 4}
    assert summary.mean_durations["retrieval"] == pytest.approx(0.3)
    assert summary.counter_rates["retrieval.acceptance_rate"] == pytest.approx(0.75)
    payload = summary.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert list(payload["counters"]) == ["retrieval.accepted", "retrieval.total"]


def test_experiment_summary_is_read_only() -> None:
    recorder = MetricsRecorder()
    recorder.increment("episodes")
    summary = ExperimentSummary.from_snapshot(recorder.snapshot())

    with pytest.raises(TypeError):
        summary.counters["episodes"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        summary.counter_rates["rate"] = 1.0  # type: ignore[index]


def test_experiment_summary_rejects_malformed_source_snapshot() -> None:
    malformed = MetricSnapshot(
        counters={},
        timing_seconds={"episode": 1.0},
        timing_counts={},
    )

    with pytest.raises(ValueError, match="same metric names"):
        ExperimentSummary.from_snapshot(malformed)


def test_experiment_summary_requires_recorded_rate_evidence() -> None:
    recorder = MetricsRecorder()
    recorder.increment("retrieval.total", 4)

    with pytest.raises(KeyError, match="retrieval.accepted"):
        ExperimentSummary.from_snapshot(
            recorder.snapshot(),
            counter_rates=(
                CounterRateSpec(
                    name="retrieval.acceptance_rate",
                    numerator="retrieval.accepted",
                    denominator="retrieval.total",
                ),
            ),
        )


def test_experiment_summary_rejects_duplicate_rate_names() -> None:
    recorder = MetricsRecorder()
    recorder.increment("accepted", 1)
    recorder.increment("total", 2)
    rate = CounterRateSpec(name="acceptance", numerator="accepted", denominator="total")

    with pytest.raises(ValueError, match="duplicate counter rate name"):
        ExperimentSummary.from_snapshot(recorder.snapshot(), counter_rates=(rate, rate))


def test_experiment_summary_rejects_invalid_rate_specifications() -> None:
    recorder = MetricsRecorder()
    recorder.increment("accepted")
    recorder.increment("total")

    with pytest.raises(ValueError, match="non-empty"):
        ExperimentSummary.from_snapshot(
            recorder.snapshot(),
            counter_rates=(CounterRateSpec(name=" ", numerator="accepted", denominator="total"),),
        )

    with pytest.raises(TypeError, match="CounterRateSpec"):
        ExperimentSummary.from_snapshot(
            recorder.snapshot(),
            counter_rates=("acceptance",),  # type: ignore[arg-type]
        )


def test_experiment_summary_rejects_non_snapshot_input() -> None:
    with pytest.raises(TypeError, match="MetricSnapshot"):
        ExperimentSummary.from_snapshot({})  # type: ignore[arg-type]


def test_experiment_summary_restores_validated_payload() -> None:
    payload = {
        "counters": {"retrieval.total": 4, "retrieval.accepted": 3},
        "mean_durations": {"retrieval": 0.3},
        "counter_rates": {"retrieval.acceptance_rate": 0.75},
    }

    summary = ExperimentSummary.from_dict(payload)

    assert summary.to_dict() == {
        "counters": {"retrieval.accepted": 3, "retrieval.total": 4},
        "mean_durations": {"retrieval": 0.3},
        "counter_rates": {"retrieval.acceptance_rate": 0.75},
    }
    with pytest.raises(TypeError):
        summary.mean_durations["retrieval"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"counters": {}, "mean_durations": {}}, ValueError),
        (
            {"counters": {"episodes": -1}, "mean_durations": {}, "counter_rates": {}},
            ValueError,
        ),
        (
            {"counters": {"episodes": True}, "mean_durations": {}, "counter_rates": {}},
            TypeError,
        ),
        (
            {"counters": {}, "mean_durations": {"episode": float("inf")}, "counter_rates": {}},
            ValueError,
        ),
        (
            {"counters": {}, "mean_durations": {}, "counter_rates": {"acceptance": 1.1}},
            ValueError,
        ),
        (
            {"counters": {}, "mean_durations": {}, "counter_rates": {" ": 0.5}},
            ValueError,
        ),
    ],
)
def test_experiment_summary_rejects_invalid_persisted_payloads(
    payload: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        ExperimentSummary.from_dict(payload)  # type: ignore[arg-type]
