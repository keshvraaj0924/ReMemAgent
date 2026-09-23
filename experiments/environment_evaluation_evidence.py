"""Fail-closed loading of persisted external-environment evaluation evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

from experiments.environment_evaluation_report import ENVIRONMENT_EVALUATION_REPORT_SCHEMA_VERSION
from experiments.runtime_provenance import RuntimeProvenance
from remem.environment_evaluation import EnvironmentEvaluation, SeededEpisodeResult
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep

_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_name",
        "policy",
        "configuration",
        "provenance",
        "provenance_fingerprint",
        "aggregates",
        "episodes",
    }
)


@dataclass(frozen=True, slots=True)
class VerifiedEnvironmentEvidence:
    """Typed environment evidence reconstructed from a verified persisted report."""

    benchmark_name: str
    policy_name: str
    policy_configuration: Mapping[str, Any]
    max_steps: int
    provenance: RuntimeProvenance
    evaluation: EnvironmentEvaluation


def load_verified_environment_evidence(path: str | Path) -> VerifiedEnvironmentEvidence:
    """Load an environment report and verify all replayable derived evidence."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("environment evaluation report must be a JSON object")
    _require_exact_fields(payload, _REPORT_FIELDS, "report")
    if payload["schema_version"] != ENVIRONMENT_EVALUATION_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported environment evaluation report schema_version")

    benchmark_name = _non_empty_string(payload["benchmark_name"], "benchmark_name")
    policy = _mapping(payload["policy"], "policy")
    _require_exact_fields(policy, frozenset({"name", "configuration"}), "policy")
    policy_name = _non_empty_string(policy["name"], "policy.name")
    policy_configuration = _mapping(policy["configuration"], "policy.configuration")

    configuration = _mapping(payload["configuration"], "configuration")
    _require_exact_fields(configuration, frozenset({"max_steps", "seeds"}), "configuration")
    max_steps = configuration["max_steps"]
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("configuration.max_steps must be a positive integer")

    provenance = RuntimeProvenance.from_dict(_mapping(payload["provenance"], "provenance"))
    if payload["provenance_fingerprint"] != provenance.fingerprint():
        raise ValueError("provenance_fingerprint does not match provenance")

    episodes_payload = payload["episodes"]
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise ValueError("episodes must be a non-empty list")
    evaluation = EnvironmentEvaluation(episodes=tuple(_episode(item) for item in episodes_payload))

    seeds = configuration["seeds"]
    if not isinstance(seeds, list) or seeds != [episode.seed for episode in evaluation.episodes]:
        raise ValueError("configuration.seeds does not match episode seeds")
    if any(episode.result.step_count > max_steps for episode in evaluation.episodes):
        raise ValueError("episode step_count exceeds configuration.max_steps")

    _verify_aggregates(_mapping(payload["aggregates"], "aggregates"), evaluation)
    return VerifiedEnvironmentEvidence(
        benchmark_name=benchmark_name,
        policy_name=policy_name,
        policy_configuration=dict(policy_configuration),
        max_steps=max_steps,
        provenance=provenance,
        evaluation=evaluation,
    )


def _episode(payload: object) -> SeededEpisodeResult:
    episode = _mapping(payload, "episode")
    _require_exact_fields(episode, frozenset({"seed", "result"}), "episode")
    seed = episode["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("episode.seed must be an integer")
    result_payload = _mapping(episode["result"], "episode.result")
    _require_exact_fields(
        result_payload,
        frozenset(
            {
                "initial_observation",
                "total_reward",
                "terminated",
                "truncated",
                "step_count",
                "steps",
            }
        ),
        "episode.result",
    )
    steps_payload = result_payload["steps"]
    if not isinstance(steps_payload, list):
        raise TypeError("episode.result.steps must be a list")
    steps = tuple(_step(item) for item in steps_payload)
    if result_payload["step_count"] != len(steps):
        raise ValueError("episode.result.step_count does not match steps")
    total_reward = _finite_number(result_payload["total_reward"], "episode.result.total_reward")
    if not isclose(
        total_reward, sum(step.result.reward for step in steps), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("episode total_reward does not match transition rewards")
    return SeededEpisodeResult(
        seed=seed,
        result=EpisodeResult(
            initial_observation=_string(
                result_payload["initial_observation"], "initial_observation"
            ),
            steps=steps,
            total_reward=total_reward,
            terminated=_boolean(result_payload["terminated"], "terminated"),
            truncated=_boolean(result_payload["truncated"], "truncated"),
        ),
    )


def _step(payload: object) -> EpisodeStep:
    step = _mapping(payload, "step")
    _require_exact_fields(
        step, frozenset({"step_index", "observation", "action", "result"}), "step"
    )
    result = _mapping(step["result"], "step.result")
    _require_exact_fields(
        result,
        frozenset({"observation", "reward", "terminated", "truncated", "info"}),
        "step.result",
    )
    return EpisodeStep(
        step_index=step["step_index"],
        observation=_string(step["observation"], "step.observation"),
        action=_non_empty_string(step["action"], "step.action"),
        result=StepResult(
            observation=_string(result["observation"], "step.result.observation"),
            reward=_finite_number(result["reward"], "step.result.reward"),
            terminated=_boolean(result["terminated"], "step.result.terminated"),
            truncated=_boolean(result["truncated"], "step.result.truncated"),
            info=dict(_mapping(result["info"], "step.result.info")),
        ),
    )


def _verify_aggregates(aggregates: Mapping[str, Any], evaluation: EnvironmentEvaluation) -> None:
    expected = {
        "episode_count": evaluation.episode_count,
        "mean_reward": evaluation.mean_reward,
        "mean_steps": evaluation.mean_steps,
        "termination_rate": evaluation.termination_rate,
        "truncation_rate": evaluation.truncation_rate,
    }
    _require_exact_fields(aggregates, frozenset(expected), "aggregates")
    for name, expected_value in expected.items():
        actual = aggregates[name]
        if name == "episode_count":
            if actual != expected_value:
                raise ValueError("aggregates.episode_count does not match episodes")
            continue
        actual_value = _finite_number(actual, f"aggregates.{name}")
        if not isclose(actual_value, float(expected_value), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"aggregates.{name} does not match episodes")


def _require_exact_fields(payload: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _non_empty_string(value: object, field_name: str) -> str:
    text = _string(value, field_name)
    if not text.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return text.strip()


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


__all__ = ["VerifiedEnvironmentEvidence", "load_verified_environment_evidence"]
