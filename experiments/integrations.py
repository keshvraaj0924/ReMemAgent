"""Concrete bridges from external benchmark factories to ReMemAgent contracts.

The integration layer owns only adaptation. Benchmark-specific packages, model
checkpoints, prompts, and task configuration remain caller-owned and are loaded
through explicit callable specifications.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from remem.environments.alfworld import AlfWorldAdapter
from remem.environments.base import EnvironmentAdapter
from remem.environments.webshop import WebShopAdapter

EnvironmentFactory = Callable[[int], Any]
AdapterFactory = Callable[[Any], EnvironmentAdapter]


def build_environment_factory(
    factory: EnvironmentFactory,
    *,
    benchmark: str,
) -> Callable[[int], EnvironmentAdapter]:
    """Adapt an external environment factory to the normalized environment API.

    ``benchmark`` selects only the protocol adapter; the supplied factory still
    controls construction of the real benchmark environment.
    """

    adapter_factory = _adapter_factory_for(benchmark)

    def create_environment(seed: int) -> EnvironmentAdapter:
        environment = factory(seed)
        return adapter_factory(environment)

    return create_environment


def _adapter_factory_for(benchmark: str) -> AdapterFactory:
    """Return the adapter constructor for a supported external benchmark."""

    normalized_name = benchmark.strip().lower()
    if normalized_name == "alfworld":
        return AlfWorldAdapter
    if normalized_name == "webshop":
        return WebShopAdapter
    raise ValueError(f"unsupported benchmark '{benchmark}'")
