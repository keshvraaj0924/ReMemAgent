"""Executable, auditable evaluation pipeline for supported text benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.environment_evaluation_evidence import (
    VerifiedEnvironmentEvidence,
    load_verified_environment_evidence,
)
from experiments.environment_evaluation_report import save_environment_evaluation_report
from experiments.integrations import EnvironmentFactory, build_environment_factory
from experiments.runtime_provenance import RuntimeProvenance
from remem.environment_evaluation import PolicyFactory, evaluate_environment_seeds


@dataclass(frozen=True, slots=True)
class EnvironmentEvaluationRequest:
    """Immutable inputs required to execute one reproducible benchmark evaluation."""

    benchmark_name: str
    seeds: tuple[int, ...]
    max_steps: int
    policy_name: str
    policy_configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.benchmark_name.strip():
            raise ValueError("benchmark_name must be non-empty")
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self.policy_name.strip():
            raise ValueError("policy_name must be non-empty")


def build_environment_evaluation_request(
    *,
    benchmark_name: str,
    seeds: Iterable[int],
    max_steps: int,
    policy_name: str,
    policy_configuration: Mapping[str, Any] | None = None,
) -> EnvironmentEvaluationRequest:
    """Materialize caller inputs before any benchmark environment is constructed."""

    return EnvironmentEvaluationRequest(
        benchmark_name=benchmark_name.strip(),
        seeds=tuple(seeds),
        max_steps=max_steps,
        policy_name=policy_name.strip(),
        policy_configuration=dict(policy_configuration or {}),
    )


def run_environment_evaluation_pipeline(
    *,
    request: EnvironmentEvaluationRequest,
    external_environment_factory: EnvironmentFactory,
    policy_factory: PolicyFactory,
    provenance: RuntimeProvenance,
    destination: str | Path,
) -> VerifiedEnvironmentEvidence:
    """Execute, persist, reload, and verify one environment evaluation.

    Returning only the verified reconstruction prevents callers from accidentally
    treating an unverified in-memory aggregate or partially written report as
    durable research evidence.
    """

    environment_factory = build_environment_factory(
        external_environment_factory,
        benchmark=request.benchmark_name,
    )
    evaluation = evaluate_environment_seeds(
        environment_factory,
        policy_factory,
        seeds=request.seeds,
        max_steps=request.max_steps,
    )
    report_path = save_environment_evaluation_report(
        destination,
        evaluation,
        benchmark_name=request.benchmark_name,
        policy_name=request.policy_name,
        policy_configuration=request.policy_configuration,
        max_steps=request.max_steps,
        provenance=provenance,
    )
    verified = load_verified_environment_evidence(report_path)
    _verify_request_binding(request, verified)
    return verified


def _verify_request_binding(
    request: EnvironmentEvaluationRequest,
    evidence: VerifiedEnvironmentEvidence,
) -> None:
    """Ensure persisted evidence remains bound to the exact execution request."""

    if evidence.benchmark_name != request.benchmark_name:
        raise ValueError("verified benchmark_name does not match execution request")
    if evidence.policy_name != request.policy_name:
        raise ValueError("verified policy_name does not match execution request")
    if dict(evidence.policy_configuration) != dict(request.policy_configuration):
        raise ValueError("verified policy_configuration does not match execution request")
    if evidence.max_steps != request.max_steps:
        raise ValueError("verified max_steps does not match execution request")
    evidence_seeds = tuple(episode.seed for episode in evidence.evaluation.episodes)
    if evidence_seeds != request.seeds:
        raise ValueError("verified seeds do not match execution request")


__all__ = [
    "EnvironmentEvaluationRequest",
    "build_environment_evaluation_request",
    "run_environment_evaluation_pipeline",
]
