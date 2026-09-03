"""Factories that bind external benchmark environments to ReMemAgent adapters.

The concrete benchmark packages remain caller-owned. This module only resolves a
factory and wraps the resulting environment with the appropriate normalized
adapter, making the boundary explicit and testable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

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


def _validate_seed(seed: int) -> None:
    """Reject seed values that cannot satisfy the deterministic factory contract."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


class BenchmarkEnvironmentFactory:
    """Create normalized environments from a caller-owned raw environment factory."""

    def __init__(self, benchmark_name: str, raw_factory: RawEnvironmentFactory) -> None:
        """Initialize a factory for one supported benchmark family."""

        self._benchmark_family = _benchmark_family(benchmark_name)
        if not callable(raw_factory):
            raise TypeError("raw_factory must be callable")
        self._raw_factory = raw_factory

    def __call__(self, seed: int) -> EnvironmentAdapter:
        """Create and adapt one environment using the supplied episode seed."""

        _validate_seed(seed)
        try:
            environment = self._raw_factory(seed)
        except Exception as exc:
            raise RuntimeError(
                f"failed to create {self._benchmark_family} environment for seed {seed}"
            ) from exc

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

    return cast(RawEnvironmentFactory, resolve_callable(specification))
