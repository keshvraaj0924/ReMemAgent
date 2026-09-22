"""Validated research-reporting artifacts built from observability evidence."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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


__all__ = ["CounterRateSpec", "ExperimentSummary"]
