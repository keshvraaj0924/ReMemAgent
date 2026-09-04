"""Executable boundary for caller-owned external benchmark integrations.

The framework does not install ALFWorld, WebShop, model SDKs, or checkpoints.
Instead, this module resolves explicit ``module:attribute`` specifications and
hands the resulting factories to the normalized benchmark runner. This keeps
third-party dependencies outside the core package while making the experiment
entrypoint reproducible and testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import cast

from remem.benchmark import (
    BenchmarkRunConfiguration,
    BenchmarkRunReport,
    BenchmarkSuiteRunner,
    PolicyFactory,
)
from remem.environments import EnvironmentContractReport, validate_environment_contract
from remem.integrations.benchmarks import load_benchmark_environment_factory
from remem.integrations.loading import resolve_callable, split_callable_specification
from remem.integrations.policies import (
    ActionPolicyFactory,
    build_memory_guided_policy_factory,
    validate_policy_contract,
)
from remem.memory.attribution import TransferSuccessEvaluator
from remem.memory.store import MemoryStore
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

        if not isinstance(self.benchmark_name, str) or not self.benchmark_name.strip():
            raise ValueError("benchmark_name must be a non-empty string")
        _validate_non_negative_integer("episode_count", self.episode_count)
        _validate_positive_integer("max_steps", self.max_steps)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if self.policy_factory is None and self.action_policy_factory is None:
            raise ValueError("one of policy_factory or action_policy_factory is required")
        if self.policy_factory is not None and self.action_policy_factory is not None:
            raise ValueError("policy_factory and action_policy_factory are mutually exclusive")
        for field_name in ("environment_factory", "success_evaluator"):
            _validate_callable_specification(field_name, getattr(self, field_name))
        if self.policy_factory is not None:
            _validate_callable_specification("policy_factory", self.policy_factory)
        if self.action_policy_factory is not None:
            _validate_callable_specification("action_policy_factory", self.action_policy_factory)
        if self.transfer_success_evaluator is not None:
            _validate_callable_specification("transfer_success_evaluator", self.transfer_success_evaluator)
        if isinstance(self.minimum_trust, bool) or not isinstance(self.minimum_trust, (int, float)):
            raise TypeError("minimum_trust must be a number between 0 and 1")
        if not isfinite(float(self.minimum_trust)):
            raise ValueError("minimum_trust must be finite")
        if not 0.0 <= self.minimum_trust <= 1.0:
            raise ValueError("minimum_trust must be between 0 and 1")


def validate_seed_sequence(seeds: Sequence[int]) -> tuple[int, ...]:
    """Normalize and validate an independent integer seed sequence."""

    selected_seeds = tuple(seeds)
    if not selected_seeds:
        raise ValueError("seeds must contain at least one seed")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in selected_seeds):
        raise TypeError("seeds must contain only integers")
    if len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("seeds must be unique")
    return selected_seeds


def validate_external_benchmark(spec: ExternalBenchmarkSpec) -> None:
    """Resolve every configured callable without constructing an environment."""

    load_benchmark_environment_factory(spec.benchmark_name, spec.environment_factory)
    resolve_callable(spec.success_evaluator)
    if spec.transfer_success_evaluator is not None:
        resolve_callable(spec.transfer_success_evaluator)
    if spec.policy_factory is not None:
        resolve_callable(spec.policy_factory)
    elif spec.action_policy_factory is not None:
        resolve_callable(spec.action_policy_factory)
    else:
        raise ValueError("one of policy_factory or action_policy_factory is required")


def validate_external_benchmark_runtime(
    spec: ExternalBenchmarkSpec,
    *,
    probe_action: str | None = None,
) -> EnvironmentContractReport:
    """Probe the real configured environment and policy before execution.

    The framework creates a temporary environment for contract validation and
    closes it as part of the environment probe. Policy validation only consumes
    the captured initial observation, so it does not need the environment to
    remain alive. Probe output is never benchmark data.
    """

    validate_external_benchmark(spec)
    environment_factory = load_benchmark_environment_factory(spec.benchmark_name, spec.environment_factory)
    probe_seed = 0 if spec.seed is None else spec.seed
    environment = environment_factory(probe_seed)
    environment_report = validate_environment_contract(
        environment,
        probe_action=probe_action,
        close_environment=True,
    )
    policy_factory = _resolve_policy_factory(spec)
    validate_policy_contract(
        policy_factory,
        seed=probe_seed,
        observation=environment_report.initial_observation,
        store=MemoryStore(),
    )
    return environment_report


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
    environment_factory = load_benchmark_environment_factory(spec.benchmark_name, spec.environment_factory)
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
        policy_factory=spec.policy_factory or spec.action_policy_factory,
        success_evaluator=spec.success_evaluator,
        transfer_success_evaluator=spec.transfer_success_evaluator,
        minimum_trust=spec.minimum_trust,
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


def run_repeated_external_benchmarks(
    spec: ExternalBenchmarkSpec,
    seeds: Sequence[int],
) -> tuple[BenchmarkRunReport, ...]:
    """Execute the same external benchmark independently for each requested seed."""

    selected_seeds = validate_seed_sequence(seeds)
    return tuple(run_external_benchmark(replace(spec, seed=seed)) for seed in selected_seeds)


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


def _validate_non_negative_integer(field_name: str, value: object) -> None:
    """Require an exact integer value greater than or equal to zero."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_positive_integer(field_name: str, value: object) -> None:
    """Require an exact integer value greater than zero."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _close_environment_after_failure(environment: object) -> None:
    """Attempt cleanup without replacing the active preflight exception."""

    try:
        _close_environment(environment)
    except BaseException:
        return


def _close_environment(environment: object) -> None:
    """Close an environment when it exposes a callable cleanup method."""

    close = getattr(environment, "close", None)
    if callable(close):
        close()
