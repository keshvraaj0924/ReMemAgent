"""Executable boundary for caller-owned external benchmark integrations.

The framework does not install ALFWorld, WebShop, model SDKs, or checkpoints.
Instead, this module resolves explicit ``module:attribute`` specifications and
hands the resulting factories to the normalized benchmark runner. This keeps
third-party dependencies outside the core package while making the experiment
entrypoint reproducible and testable.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from remem.benchmark import (
    BenchmarkRunReport,
    BenchmarkSuiteRunner,
    EnvironmentFactory,
    PolicyFactory,
)
from remem.memory.attribution import TransferSuccessEvaluator
from remem.services import SuccessEvaluator


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkSpec:
    """Fully qualified callables required to execute one external benchmark."""

    benchmark_name: str
    episode_count: int
    max_steps: int
    environment_factory: str
    policy_factory: str
    success_evaluator: str
    transfer_success_evaluator: str | None = None
    seed: int | None = None


def resolve_callable(specification: str) -> Callable[..., Any]:
    """Resolve a callable from ``module:attribute`` notation."""

    module_name, separator, attribute_path = specification.partition(":")
    if not separator or not module_name.strip() or not attribute_path.strip():
        raise ValueError(f"invalid callable specification: {specification!r}")

    module = importlib.import_module(module_name.strip())
    value: Any = module
    for attribute_name in attribute_path.split("."):
        if not attribute_name.strip():
            raise ValueError(f"invalid callable specification: {specification!r}")
        try:
            value = getattr(value, attribute_name)
        except AttributeError as exc:
            raise ValueError(f"callable attribute not found: {specification!r}") from exc

    if not callable(value):
        raise TypeError(f"resolved value is not callable: {specification!r}")
    return cast(Callable[..., Any], value)


def run_external_benchmark(
    spec: ExternalBenchmarkSpec,
    *,
    runner: BenchmarkSuiteRunner | None = None,
) -> BenchmarkRunReport:
    """Execute an external benchmark through the normalized ReMemAgent runner.

    The supplied environment and policy factories own all third-party setup.
    ReMemAgent only enforces the normalized factory signatures and benchmark
    lifecycle, so no external dependency is silently imported or fabricated.
    """

    selected_runner = runner or BenchmarkSuiteRunner()
    environment_factory = cast(EnvironmentFactory, resolve_callable(spec.environment_factory))
    policy_factory = cast(PolicyFactory, resolve_callable(spec.policy_factory))
    success_evaluator = cast(SuccessEvaluator, resolve_callable(spec.success_evaluator))
    transfer_success_evaluator = (
        cast(TransferSuccessEvaluator, resolve_callable(spec.transfer_success_evaluator))
        if spec.transfer_success_evaluator is not None
        else None
    )
    return selected_runner.run(
        benchmark_name=spec.benchmark_name,
        episode_count=spec.episode_count,
        max_steps=spec.max_steps,
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        success_evaluator=success_evaluator,
        transfer_success_evaluator=transfer_success_evaluator,
        seed=spec.seed,
    )
