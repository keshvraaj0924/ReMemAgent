from __future__ import annotations

import pytest

from remem.observability_distributions import (
    ObservationHistogram,
    ObservationHistogramSnapshot,
    histogram_snapshot_delta,
    merge_histogram_snapshots,
)


def test_histogram_records_boundary_and_overflow_buckets() -> None:
    histogram = ObservationHistogram((0.1, 0.5, 1.0))

    for value in (0.0, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5):
        histogram.observe(value)

    snapshot = histogram.snapshot()

    assert snapshot.bucket_counts == (2, 2, 2, 1)
    assert snapshot.count == 7
    assert snapshot.total == pytest.approx(4.05)
    assert snapshot.mean == pytest.approx(4.05 / 7)


def test_histogram_snapshot_isolated_from_future_observations() -> None:
    histogram = ObservationHistogram((1.0,))
    histogram.observe(0.5)

    first = histogram.snapshot()
    histogram.observe(2.0)

    assert first.bucket_counts == (1, 0)
    assert histogram.snapshot().bucket_counts == (1, 1)


def test_histogram_snapshot_to_dict_is_deterministic() -> None:
    snapshot = ObservationHistogramSnapshot(
        upper_bounds=(0.1, 1.0),
        bucket_counts=(1, 2, 1),
        total=2.2,
    )

    assert snapshot.to_dict() == {
        "upper_bounds": [0.1, 1.0],
        "bucket_counts": [1, 2, 1],
        "count": 4,
        "total": 2.2,
        "mean": 0.55,
    }


def test_histogram_rejects_invalid_bucket_contracts() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservationHistogram((1.0, 1.0))
    with pytest.raises(ValueError, match="finite and non-negative"):
        ObservationHistogram((-1.0,))
    with pytest.raises(TypeError, match="must be numbers"):
        ObservationHistogram((True,))  # type: ignore[arg-type]


def test_histogram_rejects_invalid_observations() -> None:
    histogram = ObservationHistogram((1.0,))

    with pytest.raises(ValueError, match="finite and non-negative"):
        histogram.observe(float("inf"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        histogram.observe(-0.1)
    with pytest.raises(TypeError, match="must be a number"):
        histogram.observe(True)  # type: ignore[arg-type]


def test_merge_histogram_snapshots_adds_compatible_buckets() -> None:
    first = ObservationHistogramSnapshot((1.0, 2.0), (1, 2, 0), 3.0)
    second = ObservationHistogramSnapshot((1.0, 2.0), (2, 0, 1), 4.0)

    merged = merge_histogram_snapshots((first, second))

    assert merged.bucket_counts == (3, 2, 1)
    assert merged.count == 6
    assert merged.total == 7.0


def test_merge_histogram_snapshots_rejects_incompatible_bounds() -> None:
    first = ObservationHistogramSnapshot((1.0,), (1, 0), 0.5)
    second = ObservationHistogramSnapshot((2.0,), (1, 0), 0.5)

    with pytest.raises(ValueError, match="upper bounds must match exactly"):
        merge_histogram_snapshots((first, second))


def test_merge_histogram_snapshots_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one histogram snapshot"):
        merge_histogram_snapshots(())


def test_histogram_snapshot_delta_returns_interval_distribution() -> None:
    previous = ObservationHistogramSnapshot((1.0, 2.0), (1, 1, 0), 2.0)
    current = ObservationHistogramSnapshot((1.0, 2.0), (2, 3, 1), 7.5)

    delta = histogram_snapshot_delta(previous, current)

    assert delta.bucket_counts == (1, 2, 1)
    assert delta.count == 4
    assert delta.total == 5.5


def test_histogram_snapshot_delta_rejects_regression() -> None:
    previous = ObservationHistogramSnapshot((1.0,), (2, 0), 1.0)
    current = ObservationHistogramSnapshot((1.0,), (1, 1), 1.5)

    with pytest.raises(ValueError, match="bucket count regressed"):
        histogram_snapshot_delta(previous, current)


def test_histogram_snapshot_validates_internal_consistency() -> None:
    with pytest.raises(ValueError, match="length does not match"):
        ObservationHistogramSnapshot((1.0,), (1,), 0.5)
    with pytest.raises(ValueError, match="empty histogram must have a zero total"):
        ObservationHistogramSnapshot((1.0,), (0, 0), 0.5)
