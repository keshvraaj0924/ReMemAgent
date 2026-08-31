"""Run real benchmark suites from externally supplied factory callables.

The benchmark dependencies and model implementations remain outside this package.
A callable specification in ``module:attribute`` form provides a small, explicit
integration boundary that can execute the real ALFWorld/WebShop setup without
vendoring their dependency trees into ReMemAgent.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from remem.benchmark import BenchmarkRunReport, BenchmarkSuiteRunner
from remem.environments.base import EnvironmentAdapter
from remem.execution import Policy
from remem.memory.attribution import TransferSuccessEvaluator
from remem.memory.store import MemoryStore
from remem.services import SuccessEvaluator

CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def load_callable(specification: str) -> Callable[..., Any]:
    """Load a callable from an explicit ``module:attribute`` specification."""

    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise ValueError(
            "callable specification must use the form 'module:attribute'"
        )

    module = importlib.import_module(module_name.strip())
    target: Any = module
    for component in attribute_name.strip().split("."):
        if not component:
            raise ValueError("callable attribute path must not contain empty components")
        try:
            target = getattr(target, component)
        except AttributeError as exc:
            raise AttributeError(
                f"callable '{specification}' does not expose '{component}'"
            ) from exc

    if not callable(target):
        raise TypeError(f"callable '{specification}' resolved to a non-callable object")
    return target


def load_typed_callable(specification: str) -> CallableT:
    """Load a callable while preserving the caller's static callable contract."""

    return load_callable(specification)  # type: ignore[return-value]


def run_external_benchmark(
    *,
    benchmark_name: str,
    episode_count: int,
    max_steps: int,
    environment_factory: Callable[[int], EnvironmentAdapter],
    policy_factory: Callable[[int, MemoryStore], Policy],
    success_evaluator: SuccessEvaluator,
    store: MemoryStore | None = None,
    reset_kwargs: dict[str, Any] | None = None,
    transfer_success_evaluator: TransferSuccessEvaluator | None = None,
) -> BenchmarkRunReport:
    """Execute a benchmark using real caller-owned environments and policies.

    This function intentionally performs no synthetic scoring or dependency
    discovery. The supplied factories own benchmark setup, model inference,
    checkpoint selection, and environment configuration.
    """

    return BenchmarkSuiteRunner().run(
        benchmark_name=benchmark_name,
        episode_count=episode_count,
        max_steps=max_steps,
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        success_evaluator=success_evaluator,
        store=store,
        reset_kwargs=reset_kwargs,
        transfer_success_evaluator=transfer_success_evaluator,
    )
