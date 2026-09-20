import json

import pytest

from remem.observability import MetricSnapshot, MetricsRecorder


def test_metrics_recorder_returns_deterministic_snapshot() -> None:
    recorder = MetricsRecorder()

    recorder.increment("retrieval.accepted", 2)
    recorder.increment("retrieval.rejected")
    recorder.observe_duration("reconstruction", 0.25)
    recorder.observe_duration("reconstruction", 0.75)

    snapshot = recorder.snapshot()

    assert snapshot.counters == {
        "retrieval.accepted": 2,
        "retrieval.rejected": 1,
    }
    assert snapshot.timing_seconds == {"reconstruction": 1.0}
    assert snapshot.timing_counts == {"reconstruction": 2}


def test_metric_snapshot_is_read_only_and_detached_from_recorder() -> None:
    recorder = MetricsRecorder()
    recorder.increment("events")
    snapshot = recorder.snapshot()

    recorder.increment("events")

    assert snapshot.counters == {"events": 1}
    with pytest.raises(TypeError):
        snapshot.counters["events"] = 99  # type: ignore[index]


def test_metric_snapshot_to_dict_is_deterministic_and_json_serializable() -> None:
    snapshot = MetricSnapshot(
        counters={"z": 1, "a": 2},
        timing_seconds={"step": 1.5},
        timing_counts={"step": 3},
    )

    payload = snapshot.to_dict()

    assert list(payload["counters"]) == ["a", "z"]
    assert json.loads(json.dumps(payload)) == payload


def test_metrics_recorder_merges_worker_snapshots() -> None:
    first_worker = MetricsRecorder()
    first_worker.increment("episodes", 2)
    first_worker.observe_duration("episode", 0.4)

    second_worker = MetricsRecorder()
    second_worker.increment("episodes", 3)
    second_worker.observe_duration("episode", 0.6)
    second_worker.observe_duration("episode", 1.0)

    aggregate = MetricsRecorder()
    aggregate.merge(first_worker.snapshot())
    aggregate.merge(second_worker.snapshot())

    snapshot = aggregate.snapshot()
    assert snapshot.counters == {"episodes": 5}
    assert snapshot.timing_seconds == {"episode": 2.0}
    assert snapshot.timing_counts == {"episode": 3}


def test_metrics_recorder_rejects_malformed_snapshot() -> None:
    recorder = MetricsRecorder()

    with pytest.raises(ValueError, match="same metric names"):
        recorder.merge(
            MetricSnapshot(
                counters={},
                timing_seconds={"episode": 1.0},
                timing_counts={},
            )
        )

    with pytest.raises(ValueError, match="timing count must be positive"):
        recorder.merge(
            MetricSnapshot(
                counters={},
                timing_seconds={"episode": 1.0},
                timing_counts={"episode": 0},
            )
        )

    with pytest.raises(ValueError, match="finite and non-negative"):
        recorder.merge(
            MetricSnapshot(
                counters={},
                timing_seconds={"episode": float("nan")},
                timing_counts={"episode": 1},
            )
        )


def test_metrics_recorder_rejects_invalid_counter_updates() -> None:
    recorder = MetricsRecorder()

    with pytest.raises(ValueError, match="non-empty"):
        recorder.increment("  ")
    with pytest.raises(TypeError, match="integer"):
        recorder.increment("events", True)
    with pytest.raises(ValueError, match="positive"):
        recorder.increment("events", 0)


def test_metrics_recorder_rejects_invalid_duration() -> None:
    recorder = MetricsRecorder()

    with pytest.raises(ValueError, match="finite and non-negative"):
        recorder.observe_duration("latency", -0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        recorder.observe_duration("latency", float("nan"))


def test_metric_timer_records_elapsed_duration() -> None:
    recorder = MetricsRecorder()

    with recorder.timer("episode"):
        pass

    snapshot = recorder.snapshot()
    assert snapshot.timing_counts == {"episode": 1}
    assert snapshot.timing_seconds["episode"] >= 0.0


def test_metric_timer_rejects_reentry() -> None:
    recorder = MetricsRecorder()
    timer = recorder.timer("episode")

    with timer, pytest.raises(RuntimeError, match="entered more than once"):
        timer.__enter__()
