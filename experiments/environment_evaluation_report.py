"""Auditable persistence for external-environment evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.report_io import atomic_write_json
from remem.environment_evaluation import EnvironmentEvaluation
from remem.execution import EpisodeResult


def build_environment_evaluation_report(
    evaluation: EnvironmentEvaluation,
    *,
    benchmark_name: str,
    max_steps: int,
) -> dict[str, Any]:
    """Build a deterministic report containing aggregates and raw transitions.

    The report intentionally preserves episode-level observations, actions, rewards,
    termination state, and environment metadata so aggregate metrics remain auditable.
    Environment ``info`` values must be JSON serializable; persistence fails closed
    rather than silently dropping benchmark metadata.
    """

    if not isinstance(evaluation, EnvironmentEvaluation):
        raise TypeError("evaluation must be an EnvironmentEvaluation")
    if not isinstance(benchmark_name, str) or not benchmark_name.strip():
        raise ValueError("benchmark_name must be a non-empty string")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    return {
        "schema_version": 1,
        "benchmark_name": benchmark_name.strip(),
        "configuration": {
            "max_steps": max_steps,
            "seeds": [episode.seed for episode in evaluation.episodes],
        },
        "aggregates": {
            "episode_count": evaluation.episode_count,
            "mean_reward": evaluation.mean_reward,
            "mean_steps": evaluation.mean_steps,
            "termination_rate": evaluation.termination_rate,
            "truncation_rate": evaluation.truncation_rate,
        },
        "episodes": [
            {
                "seed": seeded_episode.seed,
                "result": _episode_payload(seeded_episode.result),
            }
            for seeded_episode in evaluation.episodes
        ],
    }


def save_environment_evaluation_report(
    destination: str | Path,
    evaluation: EnvironmentEvaluation,
    *,
    benchmark_name: str,
    max_steps: int,
) -> Path:
    """Atomically persist a complete external-environment evaluation report."""

    report = build_environment_evaluation_report(
        evaluation,
        benchmark_name=benchmark_name,
        max_steps=max_steps,
    )
    return atomic_write_json(destination, report)


def _episode_payload(result: EpisodeResult) -> dict[str, Any]:
    """Convert one immutable episode result into JSON-ready research evidence."""

    return {
        "initial_observation": result.initial_observation,
        "total_reward": result.total_reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "step_count": result.step_count,
        "steps": [
            {
                "step_index": step.step_index,
                "observation": step.observation,
                "action": step.action,
                "result": {
                    "observation": step.result.observation,
                    "reward": step.result.reward,
                    "terminated": step.result.terminated,
                    "truncated": step.result.truncated,
                    "info": dict(step.result.info),
                },
            }
            for step in result.steps
        ],
    }


__all__ = [
    "build_environment_evaluation_report",
    "save_environment_evaluation_report",
]
