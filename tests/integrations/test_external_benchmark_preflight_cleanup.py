"""Tests for external benchmark preflight resource cleanup."""

from __future__ import annotations

from typing import Any

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec, validate_external_benchmark_runtime


class ClosableEnvironment:
    """Minimal environment double that records deterministic cleanup."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _make_spec() -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="alfworld",
        episode_count=1,
        max_steps=1,
        environment_factory="tests.fixtures:environment_factory",
        policy_factory="tests.fixtures:policy_factory",
        success_evaluator="tests.fixtures:success_evaluator",
    )


def test_external_preflight_closes_environment_when_environment_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = ClosableEnvironment()

    monkeypatch.setattr("experiments.external_benchmark.validate_external_benchmark", lambda _spec: None)
    monkeypatch.setattr(
        "experiments.external_benchmark.load_benchmark_environment_factory",
        lambda _name, _spec: lambda _seed: environment,
    )

    def fail_validation(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("environment validation failed")

    monkeypatch.setattr("experiments.external_benchmark.validate_environment_contract", fail_validation)

    with pytest.raises(RuntimeError, match="environment validation failed"):
        validate_external_benchmark_runtime(_make_spec())

    assert environment.closed is True


def test_external_preflight_closes_environment_when_policy_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = ClosableEnvironment()

    monkeypatch.setattr("experiments.external_benchmark.validate_external_benchmark", lambda _spec: None)
    monkeypatch.setattr(
        "experiments.external_benchmark.load_benchmark_environment_factory",
        lambda _name, _spec: lambda _seed: environment,
    )
    monkeypatch.setattr(
        "experiments.external_benchmark.validate_environment_contract",
        lambda *_args, **_kwargs: type("Report", (), {"initial_observation": "state"})(),
    )
    monkeypatch.setattr(
        "experiments.external_benchmark._resolve_policy_factory",
        lambda _spec: object(),
    )

    def fail_validation(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("policy validation failed")

    monkeypatch.setattr("experiments.external_benchmark.validate_policy_contract", fail_validation)

    with pytest.raises(RuntimeError, match="policy validation failed"):
        validate_external_benchmark_runtime(_make_spec())

    assert environment.closed is True
