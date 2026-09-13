from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.research_experiment_plan import (
    RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION,
    ResearchExperimentPlan,
    build_research_experiment_plan,
    canonical_research_experiment_plan_json,
    load_research_experiment_plan,
    verify_research_experiment_plan,
    write_research_experiment_plan,
)

REMEM_REVISION = "1" * 40
WEBSHOP_REVISION = "2" * 40


def _build_plan() -> ResearchExperimentPlan:
    return build_research_experiment_plan(
        experiment_name="webshop-memory-study",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=100,
        max_steps=50,
        seeds=(11, 17, 29, 43, 71),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": WEBSHOP_REVISION},
        dependency_versions={"torch": "2.6.0", "transformers": "4.51.0"},
        baseline_policy_factory="study.policies:build_baseline",
        treatment_action_policy_factory="study.policies:build_remem_action",
        transfer_success_evaluator="study.metrics:is_transfer_success",
        minimum_trust=0.6,
        baseline_label="baseline",
        treatment_label="remem",
        parameters={"temperature": 0.0, "do_sample": False},
        model_identity="provider/model@checkpoint",
        notes=("primary metric: success rate",),
    )


def test_plan_is_canonical_and_mapping_order_independent() -> None:
    first = _build_plan()
    second = build_research_experiment_plan(
        experiment_name="webshop-memory-study",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=100,
        max_steps=50,
        seeds=(11, 17, 29, 43, 71),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": WEBSHOP_REVISION},
        dependency_versions={"transformers": "4.51.0", "torch": "2.6.0"},
        baseline_policy_factory="study.policies:build_baseline",
        treatment_action_policy_factory="study.policies:build_remem_action",
        transfer_success_evaluator="study.metrics:is_transfer_success",
        minimum_trust=0.6,
        baseline_label="baseline",
        treatment_label="remem",
        parameters={"do_sample": False, "temperature": 0.0},
        model_identity="provider/model@checkpoint",
        notes=("primary metric: success rate",),
    )

    assert first.schema_version == RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION
    assert canonical_research_experiment_plan_json(
        first
    ) == canonical_research_experiment_plan_json(second)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_plan_round_trip_and_identity_verification(tmp_path: Path) -> None:
    path = tmp_path / "experiment-plan.json"
    plan = _build_plan()

    write_research_experiment_plan(path, plan)

    assert load_research_experiment_plan(path) == plan
    assert (
        verify_research_experiment_plan(
            path,
            expected_sha256=plan.sha256.upper(),
            expected_revision=REMEM_REVISION,
        )
        == plan
    )


def test_writer_refuses_to_overwrite_and_cleans_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "experiment-plan.json"
    plan = _build_plan()
    write_research_experiment_plan(path, plan)
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        write_research_experiment_plan(path, plan)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_paired_plan_requires_multiple_unique_integer_seeds() -> None:
    arguments = dict(
        experiment_name="invalid-seeds",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=10,
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": WEBSHOP_REVISION},
        dependency_versions={"torch": "2.6.0"},
        baseline_policy_factory="study.policies:baseline",
        treatment_policy_factory="study.policies:treatment",
    )

    with pytest.raises(ValueError, match="at least two"):
        build_research_experiment_plan(seeds=(11,), **arguments)
    with pytest.raises(ValueError, match="unique"):
        build_research_experiment_plan(seeds=(11, 11), **arguments)
    with pytest.raises(TypeError, match="only integers"):
        build_research_experiment_plan(seeds=(11, True), **arguments)


def test_plan_requires_exactly_one_policy_mode_per_condition() -> None:
    common = dict(
        experiment_name="policy-mode",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=10,
        seeds=(11, 17),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": WEBSHOP_REVISION},
        dependency_versions={"torch": "2.6.0"},
        treatment_policy_factory="study.policies:treatment",
    )

    with pytest.raises(ValueError, match="baseline requires exactly one"):
        build_research_experiment_plan(**common)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_research_experiment_plan(
            baseline_policy_factory="study.policies:baseline",
            baseline_action_policy_factory="study.policies:baseline_action",
            **common,
        )

    action_plan = build_research_experiment_plan(
        baseline_action_policy_factory="study.policies:baseline_action",
        **common,
    )
    assert action_plan.baseline_policy_factory is None
    assert action_plan.baseline_action_policy_factory == "study.policies:baseline_action"


def test_plan_requires_pinned_sources_and_dependencies() -> None:
    common = dict(
        experiment_name="missing-pin",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=10,
        seeds=(11, 17),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        baseline_policy_factory="study.policies:baseline",
        treatment_policy_factory="study.policies:treatment",
    )
    with pytest.raises(ValueError, match="pinned source revision"):
        build_research_experiment_plan(
            source_revisions={},
            dependency_versions={"torch": "2.6.0"},
            **common,
        )

    with pytest.raises(ValueError, match="exact dependency version"):
        build_research_experiment_plan(
            source_revisions={"webshop": WEBSHOP_REVISION},
            dependency_versions={},
            **common,
        )


def test_plan_validates_trust_and_optional_transfer_evaluator() -> None:
    common = dict(
        experiment_name="trust-contract",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=10,
        seeds=(11, 17),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": WEBSHOP_REVISION},
        dependency_versions={"torch": "2.6.0"},
        baseline_policy_factory="study.policies:baseline",
        treatment_policy_factory="study.policies:treatment",
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_research_experiment_plan(minimum_trust=1.1, **common)
    with pytest.raises(ValueError, match="module:attribute"):
        build_research_experiment_plan(
            transfer_success_evaluator="missing_separator",
            **common,
        )


def test_plan_rejects_non_finite_or_structured_parameters() -> None:
    common = dict(
        experiment_name="invalid-parameter",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=10,
        seeds=(11, 17),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": WEBSHOP_REVISION},
        dependency_versions={"torch": "2.6.0"},
        baseline_policy_factory="study.policies:baseline",
        treatment_policy_factory="study.policies:treatment",
    )

    with pytest.raises(ValueError, match="finite"):
        build_research_experiment_plan(parameters={"temperature": float("nan")}, **common)
    with pytest.raises(TypeError, match="JSON scalar"):
        build_research_experiment_plan(parameters={"stop": ["done"]}, **common)


def test_load_rejects_unknown_fields_and_digest_substitution(tmp_path: Path) -> None:
    path = tmp_path / "experiment-plan.json"
    plan = _build_plan()
    payload = plan.to_dict()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        load_research_experiment_plan(path)

    path.write_text(canonical_research_experiment_plan_json(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_research_experiment_plan(path, expected_sha256="0" * 64)
