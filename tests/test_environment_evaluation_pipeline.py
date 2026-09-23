"""Tests for the executable external-environment evaluation pipeline."""

from __future__ import annotations

from experiments.environment_evaluation_pipeline import (
    build_environment_evaluation_request,
    run_environment_evaluation_pipeline,
)
from experiments.runtime_provenance import (
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    RuntimeProvenance,
    dependency_fingerprint,
)


class _Environment:
    """Deterministic protocol-compatible environment used at the integration boundary."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.closed = False

    def reset(self) -> str:
        return f"task-{self.seed}"

    def step(self, action: str):
        return f"done-{self.seed}", float(self.seed), True, {"action": action}

    def close(self) -> None:
        self.closed = True


def _provenance() -> RuntimeProvenance:
    dependencies = {"rememagent": "0.1.0"}
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision="test-revision",
        working_tree_state="clean",
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.1.0",
        dependency_fingerprint=dependency_fingerprint(dependencies),
        dependency_versions=dependencies,
    )


def test_pipeline_executes_persists_and_verifies_evidence(tmp_path) -> None:
    created: list[_Environment] = []

    def environment_factory(seed: int) -> _Environment:
        environment = _Environment(seed)
        created.append(environment)
        return environment

    request = build_environment_evaluation_request(
        benchmark_name="alfworld",
        seeds=(2, 5),
        max_steps=3,
        policy_name="test-policy",
        policy_configuration={"temperature": 0},
    )
    destination = tmp_path / "evaluation.json"

    evidence = run_environment_evaluation_pipeline(
        request=request,
        external_environment_factory=environment_factory,
        policy_factory=lambda seed: lambda observation, history: "look",
        provenance=_provenance(),
        destination=destination,
    )

    assert destination.is_file()
    assert evidence.benchmark_name == "alfworld"
    assert evidence.policy_name == "test-policy"
    assert tuple(episode.seed for episode in evidence.evaluation.episodes) == (2, 5)
    assert evidence.evaluation.mean_reward == 3.5
    assert all(environment.closed for environment in created)


def test_request_materializes_seed_generator_before_execution() -> None:
    request = build_environment_evaluation_request(
        benchmark_name="webshop",
        seeds=(seed for seed in (1, 3)),
        max_steps=2,
        policy_name="policy",
    )

    assert request.seeds == (1, 3)
    assert request.policy_configuration == {}
