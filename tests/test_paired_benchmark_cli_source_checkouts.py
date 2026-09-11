from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.paired_benchmark_cli as paired_cli
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutProvenance


def _source_arguments(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "source_checkout": ["webshop=/opt/webshop", "alfworld=/opt/alfworld"],
        "require_source_revision": ["webshop=abc123", "alfworld=def456"],
        "allow_dirty_source_checkout": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_source_checkout_contract_requires_matching_named_sources() -> None:
    arguments = _source_arguments(
        source_checkout=["webshop=/opt/webshop"],
        require_source_revision=["alfworld=def456"],
    )

    with pytest.raises(ValueError, match="must declare the same names"):
        paired_cli._build_source_checkout_contract(arguments)


def test_build_source_checkout_contract_is_case_insensitive_and_clean_by_default() -> None:
    arguments = _source_arguments(
        source_checkout=["WebShop=/opt/webshop"],
        require_source_revision=["webshop=abc123"],
    )

    paths, requirements = paired_cli._build_source_checkout_contract(arguments)

    assert paths == {"WebShop": Path("/opt/webshop")}
    assert requirements is not None
    assert requirements["WebShop"].expected_revision == "abc123"
    assert requirements["WebShop"].require_clean_working_tree is True


def test_build_source_checkout_contract_allows_explicit_dirty_checkout() -> None:
    arguments = _source_arguments(
        source_checkout=["webshop=/opt/webshop"],
        require_source_revision=["webshop=abc123"],
        allow_dirty_source_checkout=["WEBSHOP"],
    )

    _, requirements = paired_cli._build_source_checkout_contract(arguments)

    assert requirements is not None
    assert requirements["webshop"].require_clean_working_tree is False


def test_build_source_checkout_contract_rejects_unknown_dirty_name() -> None:
    arguments = _source_arguments(
        source_checkout=["webshop=/opt/webshop"],
        require_source_revision=["webshop=abc123"],
        allow_dirty_source_checkout=["alfworld"],
    )

    with pytest.raises(ValueError, match="must be declared sources"):
        paired_cli._build_source_checkout_contract(arguments)


def test_build_source_checkout_contract_rejects_duplicate_names_ignoring_case() -> None:
    arguments = _source_arguments(
        source_checkout=["webshop=/opt/webshop", "WebShop=/tmp/webshop"],
        require_source_revision=["webshop=abc123"],
    )

    with pytest.raises(ValueError, match="unique ignoring case"):
        paired_cli._build_source_checkout_contract(arguments)


def test_main_routes_declared_sources_through_controlled_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "paired.json"
    arguments = Namespace(
        benchmark="webshop",
        episodes=2,
        max_steps=4,
        seeds="11,17",
        environment_factory="tests.test_external_benchmark:make_environment",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        transfer_success_evaluator=None,
        baseline_policy_factory="tests.test_external_benchmark:make_policy",
        baseline_action_policy_factory=None,
        treatment_policy_factory="tests.test_external_benchmark:make_memory_policy",
        treatment_action_policy_factory=None,
        minimum_trust=0.0,
        baseline_label="baseline",
        treatment_label="treatment",
        output=output_path,
        manifest=None,
        overwrite=False,
        probe_action=None,
        require_code_revision=None,
        require_clean_working_tree=False,
        require_dependency_version=None,
        source_checkout=["webshop=/opt/webshop"],
        require_source_revision=["webshop=abc123"],
        allow_dirty_source_checkout=None,
    )
    admitted_source = {
        "webshop": SourceCheckoutProvenance(
            revision="abc123",
            working_tree_state="clean",
        )
    }
    captured: dict[str, object] = {}

    def fake_controlled(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["controlled_kwargs"] = kwargs
        return SimpleNamespace(
            paired_result=object(),
            runtime_provenance=SimpleNamespace(to_dict=lambda: {"code_revision": "remem123"}),
            source_checkout_provenance=admitted_source,
        )

    def fail_legacy(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy paired runner must not execute for source-controlled runs")

    def fake_save(result: object, path: Path, **kwargs: object) -> Path:
        captured["saved_result"] = result
        captured["saved_kwargs"] = kwargs
        return path

    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(paired_cli, "run_controlled_paired_external_benchmarks", fake_controlled)
    monkeypatch.setattr(paired_cli, "run_paired_external_benchmarks_with_preflight", fail_legacy)
    monkeypatch.setattr(paired_cli, "save_paired_execution_result", fake_save)

    assert paired_cli.main() == 0

    controlled_kwargs = captured["controlled_kwargs"]
    assert isinstance(controlled_kwargs, dict)
    runtime_requirements = controlled_kwargs["runtime_requirements"]
    assert isinstance(runtime_requirements, RuntimeRequirements)
    assert controlled_kwargs["source_checkout_paths"] == {"webshop": Path("/opt/webshop")}
    source_requirements = controlled_kwargs["source_checkout_requirements"]
    assert source_requirements["webshop"].expected_revision == "abc123"

    saved_kwargs = captured["saved_kwargs"]
    assert isinstance(saved_kwargs, dict)
    assert saved_kwargs["source_checkout_provenance"] is admitted_source
    assert saved_kwargs["source_checkout_requirements"] is source_requirements
    assert saved_kwargs["runtime_requirements"] is None
