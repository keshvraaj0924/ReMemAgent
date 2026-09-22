"""Validated research-reporting artifacts built from observability evidence."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

from remem.observability import MetricSnapshot


@dataclass(frozen=True, slots=True)
class CounterRateSpec:
    """Describe one counter ratio to derive for an experiment summary."""

    name: str
    numerator: str
    denominator: str


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Immutable derived metrics for one validated observability snapshot.

    The summary contains only measurements derivable from recorded evidence. It
    intentionally carries no interpretation such as benchmark wins or production
    readiness, which belong in separately reviewed research documentation.
    """

    counters: Mapping[str, int]
    mean_durations: Mapping[str, float]
    counter_rates: Mapping[str, float]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MetricSnapshot,
        *,
        counter_rates: tuple[CounterRateSpec, ...] = (),
    ) -> "ExperimentSummary":
        """Build a deterministic summary after validating all source evidence."""

        if not isinstance(snapshot, MetricSnapshot):
            raise TypeError("snapshot must be a MetricSnapshot")

        # This validates the complete snapshot before any derived value is emitted.
        mean_durations = snapshot.mean_durations()
        derived_rates: dict[str, float] = {}
        for specification in counter_rates:
            if not isinstance(specification, CounterRateSpec):
                raise TypeError("counter_rates must contain CounterRateSpec values")
            summary_name = _validate_summary_name(specification.name)
            if summary_name in derived_rates:
                raise ValueError(f"duplicate counter rate name: {summary_name}")
            derived_rates[summary_name] = snapshot.counter_rate(
                specification.numerator,
                specification.denominator,
            )

        return cls(
            counters=MappingProxyType(dict(sorted(snapshot.counters.items()))),
            mean_durations=MappingProxyType(dict(mean_durations)),
            counter_rates=MappingProxyType(dict(sorted(derived_rates.items()))),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentSummary":
        """Restore and validate a persisted experiment-summary payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("experiment summary payload must be a mapping")
        expected_keys = {"counters", "mean_durations", "counter_rates"}
        if set(payload) != expected_keys:
            raise ValueError("experiment summary payload has unexpected or missing fields")

        counters = _validate_counters(payload["counters"])
        mean_durations = _validate_float_metrics(payload["mean_durations"], "mean duration")
        counter_rates = _validate_float_metrics(
            payload["counter_rates"], "counter rate", minimum=0.0, maximum=1.0
        )
        return cls(
            counters=MappingProxyType(dict(sorted(counters.items()))),
            mean_durations=MappingProxyType(dict(sorted(mean_durations.items()))),
            counter_rates=MappingProxyType(dict(sorted(counter_rates.items()))),
        )

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        """Return a deterministic JSON-serializable reporting payload."""

        return {
            "counters": dict(sorted(self.counters.items())),
            "mean_durations": dict(sorted(self.mean_durations.items())),
            "counter_rates": dict(sorted(self.counter_rates.items())),
        }


def _validate_summary_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("summary metric name must be a string")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("summary metric name must be non-empty")
    return normalized_name


def _validate_counters(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("counters must be a mapping")
    validated: dict[str, int] = {}
    for raw_name, raw_value in value.items():
        name = _validate_summary_name(raw_name)
        if name in validated:
            raise ValueError(f"duplicate normalized metric name: {name}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(f"counter {name!r} must be an integer")
        if raw_value < 0:
            raise ValueError(f"counter {name!r} must be non-negative")
        validated[name] = raw_value
    return validated


def _validate_float_metrics(
    value: Any,
    kind: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{kind}s must be a mapping")
    validated: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = _validate_summary_name(raw_name)
        if name in validated:
            raise ValueError(f"duplicate normalized metric name: {name}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{kind} {name!r} must be numeric")
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{kind} {name!r} must be finite")
        if numeric_value < minimum or (maximum is not None and numeric_value > maximum):
            raise ValueError(f"{kind} {name!r} is outside the valid range")
        validated[name] = numeric_value
    return validated


__all__ = ["CounterRateSpec", "ExperimentSummary"]
