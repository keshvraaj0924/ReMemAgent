"""Effect-size utilities for paired benchmark comparisons.

The functions in this module operate only on seed-level observations. They are
kept separate from routing and memory heuristics so statistical interpretation
cannot influence the research policy itself.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Sequence


def paired_cohens_dz(deltas: Sequence[float]) -> float | None:
    """Return paired Cohen's *d_z* for seed-level treatment deltas.

    ``d_z`` is the mean paired difference divided by the sample standard
    deviation of the paired differences. It is undefined when fewer than two
    independent paired observations are available or when all paired
    differences are identical, because the denominator is zero. In those
    cases this function returns ``None`` rather than reporting an infinite or
    otherwise misleading effect size.

    Args:
        deltas: One treatment-minus-baseline metric delta per independent seed.

    Returns:
        The paired standardized mean difference, or ``None`` when the effect
        size is undefined from the supplied observations.

    Raises:
        ValueError: If ``deltas`` is empty or contains a non-finite value.
        TypeError: If an observation is not a real numeric value.
    """

    normalized_deltas = tuple(deltas)
    if not normalized_deltas:
        raise ValueError("deltas must contain at least one paired observation")
    if any(not isinstance(delta, (int, float)) or isinstance(delta, bool) for delta in normalized_deltas):
        raise TypeError("deltas must contain real numeric values")
    if any(not isfinite(float(delta)) for delta in normalized_deltas):
        raise ValueError("deltas must contain only finite values")
    if len(normalized_deltas) < 2:
        return None

    mean_delta = sum(normalized_deltas) / len(normalized_deltas)
    sample_variance = sum(
        (delta - mean_delta) ** 2 for delta in normalized_deltas
    ) / (len(normalized_deltas) - 1)
    sample_stddev = sqrt(sample_variance)
    if sample_stddev == 0.0:
        return None
    return mean_delta / sample_stddev


__all__ = ["paired_cohens_dz"]
