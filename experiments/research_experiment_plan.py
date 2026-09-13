"""Versioned, deterministic pre-measurement research experiment plans."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ResearchExperimentPlan:
    """Immutable protocol identity declared before measured benchmark execution."""

    schema_version: int
    experiment_name: str
    remem_revision: str
    benchmark_name: str
    episode_count: int
    max_steps: int
    seeds: tuple[int, ...]
    environment_factory: str
    success_evaluator: str
    source_revisions: tuple[tuple[str, str], ...]
    dependency_versions: tuple[tuple[str, str], ...]
    parameters: tuple[tuple[str, JsonScalar], ...]
    baseline_policy_factory: str | None = None
    baseline_action_policy_factory: str | None = None
    treatment_policy_factory: str | None = None
    treatment_action_policy_factory: str | None = None
    transfer_success_evaluator: str | None = None
    minimum_trust: float = 0.0
    baseline_label: str = "baseline"
    treatment_label: str = "treatment"
    model_identity: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def sha256(self) -> str:
        """Return the SHA-256 identity of the canonical plan bytes."""

        payload = canonical_research_experiment_plan_json(self).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic JSON representation of this plan."""

        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "remem_revision": self.remem_revision,
            "benchmark_name": self.benchmark_name,
            "episode_count": self.episode_count,
            "max_steps": self.max_steps,
            "seeds": list(self.seeds),
            "environment_factory": self.environment_factory,
            "success_evaluator": self.success_evaluator,
            "source_revisions": dict(self.source_revisions),
            "dependency_versions": dict(self.dependency_versions),
            "parameters": dict(self.parameters),
            "baseline_policy_factory": self.baseline_policy_factory,
            "baseline_action_policy_factory": self.baseline_action_policy_factory,
            "treatment_policy_factory": self.treatment_policy_factory,
            "treatment_action_policy_factory": self.treatment_action_policy_factory,
            "transfer_success_evaluator": self.transfer_success_evaluator,
            "minimum_trust": self.minimum_trust,
            "baseline_label": self.baseline_label,
            "treatment_label": self.treatment_label,
            "model_identity": self.model_identity,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchExperimentPlan:
        """Strictly parse one persisted research experiment plan."""

        expected_keys = {
            "schema_version",
            "experiment_name",
            "remem_revision",
            "benchmark_name",
            "episode_count",
            "max_steps",
            "seeds",
            "environment_factory",
            "success_evaluator",
            "source_revisions",
            "dependency_versions",
            "parameters",
            "baseline_policy_factory",
            "baseline_action_policy_factory",
            "treatment_policy_factory",
            "treatment_action_policy_factory",
            "transfer_success_evaluator",
            "minimum_trust",
            "baseline_label",
            "treatment_label",
            "model_identity",
            "notes",
        }
        if set(payload) != expected_keys:
            raise ValueError("research experiment plan has unexpected or missing fields")
        if payload["schema_version"] != RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported research experiment plan schema: {payload['schema_version']!r}"
            )
        return build_research_experiment_plan(
            experiment_name=payload["experiment_name"],
            remem_revision=payload["remem_revision"],
            benchmark_name=payload["benchmark_name"],
            episode_count=payload["episode_count"],
            max_steps=payload["max_steps"],
            seeds=_require_sequence(payload["seeds"], "seeds"),
            environment_factory=payload["environment_factory"],
            success_evaluator=payload["success_evaluator"],
            source_revisions=_require_mapping(payload["source_revisions"], "source_revisions"),
            dependency_versions=_require_mapping(
                payload["dependency_versions"], "dependency_versions"
            ),
            parameters=_require_mapping(payload["parameters"], "parameters"),
            baseline_policy_factory=payload["baseline_policy_factory"],
            baseline_action_policy_factory=payload["baseline_action_policy_factory"],
            treatment_policy_factory=payload["treatment_policy_factory"],
            treatment_action_policy_factory=payload["treatment_action_policy_factory"],
            transfer_success_evaluator=payload["transfer_success_evaluator"],
            minimum_trust=payload["minimum_trust"],
            baseline_label=payload["baseline_label"],
            treatment_label=payload["treatment_label"],
            model_identity=payload["model_identity"],
            notes=_require_sequence(payload["notes"], "notes"),
        )


def build_research_experiment_plan(
    *,
    experiment_name: object,
    remem_revision: object,
    benchmark_name: object,
    episode_count: object,
    max_steps: object,
    seeds: Sequence[object],
    environment_factory: object,
    success_evaluator: object,
    source_revisions: Mapping[object, object],
    dependency_versions: Mapping[object, object],
    parameters: Mapping[object, object] | None = None,
    baseline_policy_factory: object = None,
    baseline_action_policy_factory: object = None,
    treatment_policy_factory: object = None,
    treatment_action_policy_factory: object = None,
    transfer_success_evaluator: object = None,
    minimum_trust: object = 0.0,
    baseline_label: object = "baseline",
    treatment_label: object = "treatment",
    model_identity: object = None,
    notes: Sequence[object] = (),
) -> ResearchExperimentPlan:
    """Build and validate a complete pre-measurement experiment declaration."""

    baseline_policy, baseline_action_policy = _validated_policy_factory_pair(
        policy_factory=baseline_policy_factory,
        action_policy_factory=baseline_action_policy_factory,
        condition="baseline",
    )
    treatment_policy, treatment_action_policy = _validated_policy_factory_pair(
        policy_factory=treatment_policy_factory,
        action_policy_factory=treatment_action_policy_factory,
        condition="treatment",
    )
    validated_transfer_evaluator = _optional_callable_spec(
        transfer_success_evaluator,
        "transfer_success_evaluator",
    )
    validated_model_identity = None
    if model_identity is not None:
        validated_model_identity = _non_empty_string(model_identity, "model_identity")
    return ResearchExperimentPlan(
        schema_version=RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION,
        experiment_name=_non_empty_string(experiment_name, "experiment_name"),
        remem_revision=_revision(remem_revision, "remem_revision"),
        benchmark_name=_non_empty_string(benchmark_name, "benchmark_name"),
        episode_count=_positive_integer(episode_count, "episode_count"),
        max_steps=_positive_integer(max_steps, "max_steps"),
        seeds=_validated_seeds(seeds),
        environment_factory=_callable_spec(environment_factory, "environment_factory"),
        success_evaluator=_callable_spec(success_evaluator, "success_evaluator"),
        source_revisions=_validated_revision_mapping(source_revisions, "source_revisions"),
        dependency_versions=_validated_string_mapping(dependency_versions, "dependency_versions"),
        parameters=_validated_parameters(parameters or {}),
        baseline_policy_factory=baseline_policy,
        baseline_action_policy_factory=baseline_action_policy,
        treatment_policy_factory=treatment_policy,
        treatment_action_policy_factory=treatment_action_policy,
        transfer_success_evaluator=validated_transfer_evaluator,
        minimum_trust=_validated_probability(minimum_trust, "minimum_trust"),
        baseline_label=_non_empty_string(baseline_label, "baseline_label"),
        treatment_label=_non_empty_string(treatment_label, "treatment_label"),
        model_identity=validated_model_identity,
        notes=tuple(_non_empty_string(note, "note") for note in notes),
    )


def canonical_research_experiment_plan_json(plan: ResearchExperimentPlan) -> str:
    """Serialize a plan to canonical newline-terminated JSON."""

    return (
        json.dumps(
            plan.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_research_experiment_plan(path: str | Path, plan: ResearchExperimentPlan) -> None:
    """Atomically persist a canonical plan without replacing an existing declaration."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_research_experiment_plan_json(plan)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"research experiment plan already exists: {destination}"
                ) from exc
            temporary_path.unlink()
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def load_research_experiment_plan(path: str | Path) -> ResearchExperimentPlan:
    """Load and strictly validate a persisted plan."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("research experiment plan must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("research experiment plan root must be a JSON object")
    return ResearchExperimentPlan.from_dict(payload)


def verify_research_experiment_plan(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_revision: str | None = None,
) -> ResearchExperimentPlan:
    """Verify a persisted plan against optional predeclared identities."""

    plan = load_research_experiment_plan(path)
    if expected_revision is not None:
        validated_revision = _revision(expected_revision, "expected_revision")
        if plan.remem_revision != validated_revision:
            raise ValueError(
                "research experiment plan revision mismatch: "
                f"expected {validated_revision}, recorded {plan.remem_revision}"
            )
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
        if not hmac.compare_digest(plan.sha256, expected_sha256.lower()):
            raise ValueError("research experiment plan SHA-256 mismatch")
    return plan


def _validated_policy_factory_pair(
    *,
    policy_factory: object,
    action_policy_factory: object,
    condition: str,
) -> tuple[str | None, str | None]:
    if policy_factory is None and action_policy_factory is None:
        raise ValueError(
            f"{condition} requires exactly one of policy_factory or action_policy_factory"
        )
    if policy_factory is not None and action_policy_factory is not None:
        raise ValueError(
            f"{condition} policy_factory and action_policy_factory are mutually exclusive"
        )
    if policy_factory is not None:
        return _callable_spec(policy_factory, f"{condition}_policy_factory"), None
    return None, _callable_spec(action_policy_factory, f"{condition}_action_policy_factory")


def _validated_seeds(values: Sequence[object]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("seeds must be a sequence of integers")
    normalized: list[int] = []
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seeds must contain only integers")
        normalized.append(seed)
    if len(normalized) < 2:
        raise ValueError("research paired experiments require at least two independent seeds")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique")
    return tuple(normalized)


def _validated_revision_mapping(
    values: Mapping[object, object], field_name: str
) -> tuple[tuple[str, str], ...]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one pinned source revision")
    normalized = [
        (_non_empty_string(name, f"{field_name} key"), _revision(value, field_name))
        for name, value in values.items()
    ]
    _require_unique_casefolded_names((name for name, _ in normalized), field_name)
    return tuple(sorted(normalized, key=lambda item: item[0].casefold()))


def _validated_string_mapping(
    values: Mapping[object, object], field_name: str
) -> tuple[tuple[str, str], ...]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one exact dependency version")
    normalized = [
        (
            _non_empty_string(name, f"{field_name} key"),
            _non_empty_string(value, f"{field_name} value"),
        )
        for name, value in values.items()
    ]
    _require_unique_casefolded_names((name for name, _ in normalized), field_name)
    return tuple(sorted(normalized, key=lambda item: item[0].casefold()))


def _validated_parameters(values: Mapping[object, object]) -> tuple[tuple[str, JsonScalar], ...]:
    normalized: list[tuple[str, JsonScalar]] = []
    for name, value in values.items():
        validated_name = _non_empty_string(name, "parameter name")
        if value is None or isinstance(value, (str, bool, int)):
            validated_value: JsonScalar = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("research experiment parameters must be finite")
            validated_value = value
        else:
            raise TypeError("research experiment parameters must use JSON scalar values")
        normalized.append((validated_name, validated_value))
    _require_unique_casefolded_names((name for name, _ in normalized), "parameters")
    return tuple(sorted(normalized, key=lambda item: item[0].casefold()))


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validated_probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number between 0 and 1")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return probability


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _callable_spec(value: object, field_name: str) -> str:
    specification = _non_empty_string(value, field_name)
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise ValueError(f"{field_name} must use module:attribute syntax")
    return specification


def _optional_callable_spec(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _callable_spec(value, field_name)


def _revision(value: object, field_name: str) -> str:
    revision = _non_empty_string(value, field_name)
    if _COMMIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{field_name} must be a lowercase 40-character Git commit SHA")
    return revision


def _require_unique_casefolded_names(values: Iterable[str], field_name: str) -> None:
    names = tuple(values)
    normalized = {name.casefold() for name in names}
    if len(names) != len(normalized):
        raise ValueError(f"{field_name} names must be unique ignoring case")


def _require_mapping(value: object, field_name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list")
    return value


__all__ = [
    "RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION",
    "ResearchExperimentPlan",
    "build_research_experiment_plan",
    "canonical_research_experiment_plan_json",
    "load_research_experiment_plan",
    "verify_research_experiment_plan",
    "write_research_experiment_plan",
]
