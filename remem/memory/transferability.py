"""Statistical transferability metrics for memory research experiments.

The metrics in this module are descriptive only. They do not change routing
behavior and therefore remain separate from the deterministic trust heuristic
and any future learned transferability model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .types import MemoryRecord


DEFAULT_CONFIDENCE_Z = 1.96


@dataclass(frozen=True, slots=True)
class TransferabilityMetrics:
    """Observed transfer performance for one memory."""

    memory_id: str
    attempts: int
    successes: int
    failures: int
    empirical_rate: float
    lower_confidence_bound: float


def measure_transferability(
    memory: MemoryRecord,
    *,
    confidence_z: float = DEFAULT_CONFIDENCE_Z,
) -> TransferabilityMetrics:
    """Measure observed transfer success and its Wilson lower bound.

    The lower bound is intentionally conservative for memories with few
    transfer observations. A memory with zero observations receives the
    neutral empirical rate of ``0.5`` and a zero lower bound rather than an
    unsupported claim of reliability.
    """

    if confidence_z < 0.0:
        raise ValueError("confidence_z must be non-negative")

    attempts = memory.transfer_attempts
    successes = memory.transfer_successes
    failures = attempts - successes
    if successes < 0 or failures < 0:
        raise ValueError("transfer counts must be non-negative and consistent")

    empirical_rate = successes / attempts if attempts else 0.5
    lower_bound = _wilson_lower_bound(successes, attempts, confidence_z)
    return TransferabilityMetrics(
        memory_id=memory.memory_id,
        attempts=attempts,
        successes=successes,
        failures=failures,
        empirical_rate=empirical_rate,
        lower_confidence_bound=lower_bound,
    )


def _wilson_lower_bound(successes: int, attempts: int, z: float) -> float:
    """Return the lower Wilson interval endpoint for a Bernoulli rate."""

    if attempts == 0:
        return 0.0
    denominator = 1.0 + (z * z / attempts)
    center = (successes / attempts) + (z * z / (2.0 * attempts))
    margin = z * sqrt(
        (successes * (attempts - successes) / (attempts**3))
        + (z * z / (4.0 * attempts**2))
    )
    return max(0.0, (center - margin) / denominator)
