"""Execution lifecycle for opt-in benchmark observability artifacts.

The benchmark CLI needs one collector shared by aggregate timing and optional
fixed-bucket duration distributions. This module owns that lifecycle so CLI
argument handling does not need to know collector implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.benchmark_distribution_config import BenchmarkDistributionConfig
from remem.observability import ObservationCollector, ObservationSnapshot
from remem.observability_distribution_artifacts import DistributionObservationSnapshot
from remem.observability_distribution_collector import DistributionObservationCollector


@dataclass(frozen=True, slots=True)
class BenchmarkObservabilitySnapshots:
    """Frozen aggregate and distribution snapshots for artifact publication."""

    observation_snapshot: ObservationSnapshot | None
    distribution_snapshot: DistributionObservationSnapshot | None


@dataclass(frozen=True, slots=True)
class BenchmarkObservabilitySession:
    """Validated observability configuration for one measured benchmark run."""

    observability_output_path: Path | None = None
    distribution_config: BenchmarkDistributionConfig | None = None

    def create_collector(self) -> ObservationCollector | None:
        """Create the single collector that must be supplied to the benchmark runner."""

        if self.distribution_config is not None:
            return self.distribution_config.create_collector()
        if self.observability_output_path is not None:
            return ObservationCollector()
        return None

    def freeze(
        self,
        collector: ObservationCollector | None,
    ) -> BenchmarkObservabilitySnapshots:
        """Freeze configured evidence after measured execution.

        Distribution collection implies a distribution-aware collector. Aggregate
        snapshots are persisted only when their output path was explicitly
        requested, even though a distribution-aware collector also maintains the
        aggregate timing internally.
        """

        if collector is None:
            if self.observability_output_path is not None or self.distribution_config is not None:
                raise ValueError("configured benchmark observability requires a collector")
            return BenchmarkObservabilitySnapshots(None, None)

        observation_snapshot = (
            collector.snapshot() if self.observability_output_path is not None else None
        )
        distribution_snapshot: DistributionObservationSnapshot | None = None
        if self.distribution_config is not None:
            if not isinstance(collector, DistributionObservationCollector):
                raise TypeError(
                    "configured benchmark distributions require a DistributionObservationCollector"
                )
            distribution_snapshot = self.distribution_config.snapshot(collector)

        return BenchmarkObservabilitySnapshots(
            observation_snapshot=observation_snapshot,
            distribution_snapshot=distribution_snapshot,
        )


__all__ = [
    "BenchmarkObservabilitySession",
    "BenchmarkObservabilitySnapshots",
]
