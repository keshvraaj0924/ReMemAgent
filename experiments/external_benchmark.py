"""Executable boundary for caller-owned external benchmark integrations.

The framework does not install ALFWorld, WebShop, model SDKs, or checkpoints.
Instead, this module resolves explicit ``module:attribute`` specifications and
hands the resulting factories to the normalized benchmark runner. This keeps
third-party dependencies outside the core package while making the experiment
entrypoint reproducible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from remem.benchmark import (
    BenchmarkRunConfiguration,
    BenchmarkRunReport,
    BenchmarkSuiteRunner,
    PolicyFactory,
)
from remem.integrations.benchmarks import load_benchmark_environment_factory
from remem.integrations.loading import resolve_callable, split_callable_specification
from remem.integrations.policies import (
    ActionPolicyFactory,
    build_memory_guided_policy_factory,
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
    policy_factory: str | None
    success_evaluator: str
    transfer_success_evaluator: str | None = None
    seed: int | None = None
    action_policy_factory: str | None = None
    minimum_trust: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid experiment configuration before resolving dependencies."""

        if not self.benchmark_name.strip():
            raise ValueError("benchmark_name must not be empty")
        if self.episode_count < 0:
            raise ValueError("episode_count must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.policy_factory is None and self.action_policy_factory is None:
            raise ValueError("one of policy_factory or action_policy_factory is required")
        if self.policy_factory is not None and self.action_policy_factory is not None:
            raise ValueError("policy_factory and action_policy_factory are mutually exclusive")
        for field_name in ("environment_factory", "success_evaluator"):
            _validate_callable_specification(field_name, getattr(self, field_name))
        if self.policy_factory is not None:
            _validate_callable_specification("policy_factory", self.policy_factory)
        if self.action_policy_factory is not None:
            _validate_callable_specification(
                "action_policy_factory", self.action_policy_factory
            )
        if self.transfer_success_evaluator is not None:
            _validate_callable_specification(
                "transfer_success_evaluator", self.transfer_success_evaluator
            )
        if not 0.0 <= self.minimum_trust <= 1.0:
            raise ValueError("minimum_trust must be between 0 and 1")


def run_external_benchmark(
    spec: ExternalBenchmarkSpec,
    *,
    runner: BenchmarkSuiteRunner | None = None,
) -> BenchmarkRunReport:
    """Execute an external benchmark through the normalized ReMemAgent runner.

    The supplied environment factory constructs the real third-party benchmark
    environment and is wrapped here with the benchmark-specific adapter. The
    policy can either be a complete caller-owned ``PolicyFactory`` or a raw
    action-policy factory, in which case ReMemAgent composes memory guidance
    around that learned component. Model loading, tokenization, inference, and
    action decoding remain caller-owned.
    """

    selected_runner = runner or BenchmarkSuiteRunner()
    environment_factory = load_benchmark_environment_factory(
        spec.benchmark_name,
        spec.environment_factory,
    )
    policy_factory = _resolve_policy_factory(spec)
    success_evaluator = cast(SuccessEvaluator, resolve_callable(spec.success_evaluator))
    transfer_success_evaluator = (
        cast(TransferSuccessEvaluator, resolve_callable(spec.transfer_success_evaluator))
        if spec.transfer_success_evaluator is not None
        else None
    )
    configuration = BenchmarkRunConfiguration(
        benchmark_name=spec.benchmark_name.strip(),
        episode_count=spec.episode_count,
        max_steps=spec.max_steps,
        seed=spec.seed,
        environment_factory=spec.environment_factory,
        policy_factory=spec.policy_factory,
        success_evaluator=spec.success_evaluator,
        transfer_success_evaluator=spec.transfer_success_evaluator,
        action_policy_factory=spec.action_policy_factory,
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
        configuration=configuration,
    )


def _resolve_policy_factory(spec: ExternalBenchmarkSpec) -> PolicyFactory:
    """Resolve either a complete policy or compose one from an action policy."""

    if spec.action_policy_factory is not None:
        action_policy_factory = cast(ActionPolicyFactory, resolve_callable(spec.action_policy_factory))
        return build_memory_guided_policy_factory(
            action_policy_factory,
            minimum_trust=spec.minimum_trust,
        )
    if spec.policy_factory is None:
        raise ValueError("policy_factory is required when action_policy_factory is absent")
    return cast(PolicyFactory, resolve_callable(spec.policy_factory))


def _validate_callable_specification(field_name: str, specification: str) -> None:
    """Validate that a configured callable field uses explicit import notation."""

    try:
        split_callable_specification(specification)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use module:attribute notation") from exc


__all__ = ["ExternalBenchmarkSpec", "resolve_callable", "run_external_benchmark"]
