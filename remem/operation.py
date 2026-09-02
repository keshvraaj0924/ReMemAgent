"""Context manager for timing and outcome accounting of one operation."""

from __future__ import annotations

from types import TracebackType
from typing import Self

from .observability import ObservationCollector


class ObservationOperation:
    """Record elapsed time and a mutually exclusive operation outcome."""

    def __init__(self, collector: ObservationCollector, name: str) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("operation name must not be empty")
        self._collector = collector
        self._name = normalized_name
        self._started_at: float | None = None

    def __enter__(self) -> Self:
        from time import monotonic

        self._started_at = monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        from time import monotonic

        if self._started_at is None:
            return
        self._collector.observe_duration(self._name, monotonic() - self._started_at)
        self._collector.record_outcome(self._name, succeeded=exc_type is None)


__all__ = ["ObservationOperation"]
