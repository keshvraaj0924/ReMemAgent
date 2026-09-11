"""Regression coverage for source-checkout admission in external preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

import experiments.external_preflight as external_preflight
from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement
from tests.test_external_benchmark import CLOSED_SEEDS


def _build_spec() -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="webshop-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def test_source_checkout_mismatch_prevents_environment_construction(monkeypatch) -> None:
    CLOSED_SEEDS.clear()
    checkout_paths = {"WebShop": Path("/benchmarks/webshop")}
    requirements = {
        "WebShop": SourceCheckoutRequirement(expected_revision="required-revision")
    }
    observed = {
        "WebShop": SourceCheckoutProvenance(
            revision="actual-revision",
            working_tree_state="clean",
        )
    }

    monkeypatch.setattr(
        external_preflight,
        "collect_source_checkout_provenance",
        lambda repositories: observed,
    )

    with pytest.raises(ValueError, match="source checkout revision mismatch"):
        external_preflight.validate_repeated_external_benchmark_runtime(
            _build_spec(),
            (11, 17),
            source_checkout_paths=checkout_paths,
            source_checkout_requirements=requirements,
        )

    assert CLOSED_SEEDS == []


def test_source_checkout_contract_is_validated_once_before_seed_probes(monkeypatch) -> None:
    CLOSED_SEEDS.clear()
    events: list[str] = []
    checkout_paths = {"WebShop": Path("/benchmarks/webshop")}
    requirements = {
        "WebShop": SourceCheckoutRequirement(expected_revision="expected-revision")
    }
    observed = {
        "WebShop": SourceCheckoutProvenance(
            revision="expected-revision",
            working_tree_state="clean",
        )
    }

    def collect_source_state(repositories):
        events.append("collect")
        assert repositories is checkout_paths
        return observed

    def validate_source_state(provenance, declared_requirements) -> None:
        events.append("validate")
        assert provenance is observed
        assert declared_requirements is requirements

    monkeypatch.setattr(
        external_preflight,
        "collect_source_checkout_provenance",
        collect_source_state,
    )
    monkeypatch.setattr(
        external_preflight,
        "validate_source_checkout_requirements",
        validate_source_state,
    )

    reports = external_preflight.validate_repeated_external_benchmark_runtime(
        _build_spec(),
        (11, 17),
        source_checkout_paths=checkout_paths,
        source_checkout_requirements=requirements,
    )

    assert events == ["collect", "validate"]
    assert [report.initial_observation for report in reports] == ["seed-11", "seed-17"]
    assert CLOSED_SEEDS == [11, 17]


def test_source_checkout_paths_and_requirements_must_be_declared_together() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        external_preflight.validate_repeated_external_benchmark_runtime(
            _build_spec(),
            (11,),
            source_checkout_paths={"WebShop": Path("/benchmarks/webshop")},
        )

    with pytest.raises(ValueError, match="must be provided together"):
        external_preflight.validate_repeated_external_benchmark_runtime(
            _build_spec(),
            (11,),
            source_checkout_requirements={
                "WebShop": SourceCheckoutRequirement(expected_revision="expected-revision")
            },
        )


def test_empty_source_checkout_requirement_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        external_preflight.validate_repeated_external_benchmark_runtime(
            _build_spec(),
            (11,),
            source_checkout_paths={},
            source_checkout_requirements={},
        )


def test_measured_repeated_execution_forwards_source_checkout_contract(monkeypatch) -> None:
    checkout_paths = {"WebShop": Path("/benchmarks/webshop")}
    requirements = {
        "WebShop": SourceCheckoutRequirement(expected_revision="expected-revision")
    }
    captured: dict[str, object] = {}

    def fake_preflight(*args, **kwargs):
        captured["paths"] = kwargs.get("source_checkout_paths")
        captured["requirements"] = kwargs.get("source_checkout_requirements")
        return ()

    monkeypatch.setattr(
        external_preflight,
        "validate_repeated_external_benchmark_runtime",
        fake_preflight,
    )
    monkeypatch.setattr(
        external_preflight,
        "run_repeated_external_benchmarks",
        lambda *args, **kwargs: (),
    )

    external_preflight.run_repeated_external_benchmarks_with_preflight(
        _build_spec(),
        (11, 17),
        source_checkout_paths=checkout_paths,
        source_checkout_requirements=requirements,
    )

    assert captured["paths"] is checkout_paths
    assert captured["requirements"] is requirements


def test_single_run_source_checkout_failure_blocks_environment_probe(monkeypatch) -> None:
    CLOSED_SEEDS.clear()
    checkout_paths = {"WebShop": Path("/benchmarks/webshop")}
    requirements = {"WebShop": SourceCheckoutRequirement(expected_revision="required")}
    observed = {
        "WebShop": SourceCheckoutProvenance(
            revision="actual",
            working_tree_state="clean",
        )
    }

    monkeypatch.setattr(
        external_preflight,
        "collect_source_checkout_provenance",
        lambda repositories: observed,
    )

    with pytest.raises(ValueError, match="source checkout revision mismatch"):
        external_preflight.validate_controlled_external_benchmark_runtime(
            _build_spec(),
            source_checkout_paths=checkout_paths,
            source_checkout_requirements=requirements,
        )

    assert CLOSED_SEEDS == []
