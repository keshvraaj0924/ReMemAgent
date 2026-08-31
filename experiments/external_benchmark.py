"""Executable boundary for caller-owned external benchmark integrations.

The framework does not install ALFWorld, WebShop, model SDKs, or checkpoints.
Instead, this module resolves explicit ``module:attribute`` specifications and
hands the resulting factories to the normalized benchmark runner. This keeps
third-party dependencies outside the core package while making the experiment
entrypoint reproducible and testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from remem.benchmark import BenchmarkRunReport, BenchmarkSuiteRunner, PolicyFactory
from remem.integrations.benchmarks import load_benchmark_environment_factory
from remem.integrations.loading import resolve_callable, split_callable_specification
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

    def __post_init__(self) -> None:
        """Reject invalid experiment configuration before resolving dependencies."""

        if not self.benchmark_name.strip():
            raise ValueError("benchmark_name must not be empty")
        if self.episode_count < 0:
            raise ValueError("episode_count must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        for field_name in (
            "environment_factory",
            "policy_factory",
            "success_evaluator",
        ):
            _validate_callable_specification(field_name, getattr(self, field_name))
        if self.transfer_success_evaluator is not None:
            _validate_callable_specification(
                "transfer_success_evaluator", self.transfer_success_evaluator
            )


def run_external_benchmark(
    spec: ExternalBenchmarkSpec,
    *,
    runner: BenchmarkSuiteRunner | None = None,
) -> BenchmarkRunReport:
    """Execute an external benchmark through the normalized ReMemAgent runner.

    The supplied environment factory constructs the real third-party benchmark
    environment and is wrapped here with the benchmark-specific adapter. The
    policy and evaluator factories remain caller-owned so model and reward
    semantics are never fabricated by the framework.
    """

    selected_runner = runner or BenchmarkSuiteRunner()
    environment_factory = load_benchmark_environment_factory(
        spec.benchmark_name,
        spec.environment_factory,
    )
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


def _validate_callable_specification(field_name: str, specification: str) -> None:
    """Validate that a configured callable field uses explicit import notation."""

    try:
        split_callable_specification(specification)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use module:attribute notation") from exc


__all__ = ["ExternalBenchmarkSpec", "resolve_callable", "run_external_benchmark"]
