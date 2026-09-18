import pytest

from remem.observability import MetricsRecorder


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
