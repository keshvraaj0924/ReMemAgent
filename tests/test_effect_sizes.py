from __future__ import annotations

from experiments.effect_sizes import paired_cohens_dz


def test_paired_cohens_dz_uses_seed_level_sample_stddev() -> None:
    effect_size = paired_cohens_dz((1.0, 2.0, 3.0))
    assert effect_size is not None
    assert effect_size == 2.0


def test_paired_cohens_dz_returns_none_for_single_seed() -> None:
    assert paired_cohens_dz((1.0,)) is None


def test_paired_cohens_dz_returns_none_for_zero_variance() -> None:
    assert paired_cohens_dz((1.0, 1.0, 1.0)) is None


def test_paired_cohens_dz_returns_none_for_floating_point_zero_variance() -> None:
    deltas = (
        0.3 - 0.2,
        0.4 - 0.3,
        0.2 - 0.1,
    )

    assert paired_cohens_dz(deltas) is None


def test_paired_cohens_dz_preserves_negative_direction() -> None:
    effect_size = paired_cohens_dz((-1.0, -2.0, -3.0))
    assert effect_size is not None
    assert effect_size < 0.0


def test_paired_cohens_dz_rejects_empty_input() -> None:
    try:
        paired_cohens_dz(())
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty paired observations must be rejected")


def test_paired_cohens_dz_rejects_non_finite_values() -> None:
    try:
        paired_cohens_dz((1.0, float("nan")))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite paired observations must be rejected")


def test_paired_cohens_dz_rejects_non_numeric_values() -> None:
    try:
        paired_cohens_dz((1.0, "2.0"))  # type: ignore[arg-type]
    except TypeError as exc:
        assert "real numeric" in str(exc)
    else:
        raise AssertionError("non-numeric paired observations must be rejected")


def test_paired_cohens_dz_rejects_boolean_values() -> None:
    try:
        paired_cohens_dz((1.0, True))  # type: ignore[arg-type]
    except TypeError as exc:
        assert "real numeric" in str(exc)
    else:
        raise AssertionError("boolean observations must be rejected")
