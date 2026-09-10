"""Dependency-free export boundary for observation snapshots.

Vendor SDKs remain caller-owned. ReMemAgent exports validated snapshots through a
small protocol so OpenTelemetry, Prometheus push gateways, logging systems, or
custom research infrastructure can be integrated without adding those SDKs to
the core package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from remem.observability import ObservationSnapshot


class ObservationExporter(Protocol):
    """Protocol implemented by external observation backends."""

    def export(self, snapshot: ObservationSnapshot) -> None:
        """Export one validated observation snapshot."""


ObservationPayloadCallback = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class CallbackObservationExporter:
    """Adapt a caller-owned payload callback to the observation exporter protocol."""

    callback: ObservationPayloadCallback

    def export(self, snapshot: ObservationSnapshot) -> None:
        """Export a detached JSON-compatible snapshot payload through the callback."""

        if not callable(self.callback):
            raise TypeError("callback must be callable")
        self.callback(snapshot.to_dict())


@dataclass(frozen=True, slots=True)
class CompositeObservationExporter:
    """Fan out one snapshot to multiple exporters in deterministic order."""

    exporters: tuple[ObservationExporter, ...]

    def __init__(self, exporters: Sequence[ObservationExporter]) -> None:
        selected_exporters = tuple(exporters)
        if not selected_exporters:
            raise ValueError("exporters must contain at least one exporter")
        if any(not callable(getattr(exporter, "export", None)) for exporter in selected_exporters):
            raise TypeError("each exporter must provide a callable export method")
        object.__setattr__(self, "exporters", selected_exporters)

    def export(self, snapshot: ObservationSnapshot) -> None:
        """Export sequentially and preserve the first backend failure."""

        for exporter in self.exporters:
            exporter.export(snapshot)


def export_observation_snapshot(
    snapshot: ObservationSnapshot,
    exporters: Sequence[ObservationExporter],
) -> None:
    """Export one snapshot through an explicit ordered exporter sequence."""

    CompositeObservationExporter(exporters).export(snapshot)


__all__ = [
    "CallbackObservationExporter",
    "CompositeObservationExporter",
    "ObservationExporter",
    "ObservationPayloadCallback",
    "export_observation_snapshot",
]
