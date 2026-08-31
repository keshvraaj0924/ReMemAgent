"""Factories that bind external benchmark environments to ReMemAgent adapters.

The concrete benchmark packages remain caller-owned. This module only resolves a
factory and wraps the resulting environment with the appropriate normalized
adapter, making the boundary explicit and testable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from remem.environments import AlfWorldAdapter, EnvironmentAdapter, WebShopAdapter
from remem.integrations.loading import resolve_callable

RawEnvironmentFactory = Callable[[int], Any]
SUPPORTED_BENCHMARKS = ("alfworld", "webshop")


def _benchmark_family(benchmark_name: str) -> str:
    """Return the supported benchmark family from a benchmark identifier."""

    normalized_name = benchmark_name.strip().lower()
    for benchmark in SUPPORTED_BENCHMARKS:
        if normalized_name == benchmark or normalized_name.startswith(f"{benchmark}-"):
            return benchmark
    raise ValueError(f"unsupported benchmark: {benchmark_name!r}")


class BenchmarkEnvironmentFactory:
    """Create normalized environments from a caller-owned raw environment factory."""

    def __init__(self, benchmark_name: str, raw_factory: RawEnvironmentFactory) -> None:
        """Initialize a factory for one supported benchmark family."""

        self._benchmark_family = _benchmark_family(benchmark_name)
        self._raw_factory = raw_factory

    def __call__(self, seed: int) -> EnvironmentAdapter:
        """Create and adapt one environment using the supplied episode seed."""

        environment = self._raw_factory(seed)
        if self._benchmark_family == "alfworld":
            return AlfWorldAdapter(environment)
        return WebShopAdapter(environment)


def load_benchmark_environment_factory(
    benchmark_name: str,
    specification: str,
) -> BenchmarkEnvironmentFactory:
    """Resolve a raw ``module:attribute`` factory and bind its benchmark adapter."""

    value = resolve_environment_factory(specification)
    return BenchmarkEnvironmentFactory(benchmark_name, value)


def resolve_environment_factory(specification: str) -> RawEnvironmentFactory:
    """Resolve a callable environment factory from ``module:attribute`` notation."""

    return resolve_callable(specification)  # type: ignore[return-value]
