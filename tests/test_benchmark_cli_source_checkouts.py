from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.benchmark_cli as benchmark_cli


def _arguments(tmp_path: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "benchmark": "webshop",
        "episodes": 2,
        "max_steps": 4,
        "seed": 11,
        "seeds": None,
        "environment_factory": "tests.test_external_benchmark:make_environment",
        "policy_factory": "tests.test_external_benchmark:make_policy",
        "action_policy_factory": None,
        "minimum_trust": 0.0,
        "success_evaluator": "tests.test_external_benchmark:evaluate_success",
        "transfer_success_evaluator": None,
        "output": tmp_path / "benchmark.json",
        "manifest": None,
        "observability_output": None,
        "overwrite": False,
        "preflight": False,
        "runtime_preflight": False,
        "repeated_runtime_preflight": False,
        "preflight_before_run": False,
        "probe_action": None,
        "require_code_revision": None,
        "require_clean_working_tree": False,
        "require_dependency_version": None,
        "source_checkout": ["WebShop=/opt/webshop"],
        "require_source_revision": ["webshop=abc123"],
        "allow_dirty_source_checkout": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_source_checkout_contract_is_case_insensitive_and_clean_by_default(
    tmp_path: Path,
) -> None:
    paths, requirements = benchmark_cli._build_source_checkout_contract(_arguments(tmp_path))

    assert paths == {"WebShop": Path("/opt/webshop")}
    assert requirements is not None
    assert requirements["WebShop"].expected_revision == "abc123"
    assert requirements["WebShop"].require_clean_working_tree is True


def test_build_source_checkout_contract_allows_explicit_dirty_checkout(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, allow_dirty_source_checkout=["WEBSHOP"])

    _, requirements = benchmark_cli._build_source_checkout_contract(arguments)

    assert requirements is not None
    assert requirements["WebShop"].require_clean_working_tree is False


def test_build_source_checkout_contract_requires_matching_names(tmp_path: Path) -> None:
    arguments = _arguments(
        tmp_path,
        source_checkout=["webshop=/opt/webshop"],
        require_source_revision=["alfworld=def456"],
    )

    with pytest.raises(ValueError, match="must declare the same names"):
        benchmark_cli._build_source_checkout_contract(arguments)


def test_callable_only_preflight_rejects_source_checkout_requirements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, preflight=True)
    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)

    with pytest.raises(ValueError, match="source-checkout requirements require"):
        benchmark_cli.main()


def test_runtime_preflight_forwards_source_checkout_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, runtime_preflight=True)
    captured: dict[str, object] = {}

    def fake_controlled(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(step_result=None)

    def fail_legacy(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy runtime preflight must not run for source-controlled input")

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "validate_controlled_external_benchmark_runtime",
        fake_controlled,
    )
    monkeypatch.setattr(benchmark_cli, "validate_external_benchmark_runtime", fail_legacy)

    assert benchmark_cli.main() == 0
    assert captured["runtime_requirements"] is None
    assert captured["source_checkout_paths"] == {"WebShop": Path("/opt/webshop")}
    requirements = captured["source_checkout_requirements"]
    assert isinstance(requirements, dict)
    assert requirements["WebShop"].expected_revision == "abc123"


def test_repeated_runtime_preflight_forwards_source_checkout_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(
        tmp_path,
        seed=None,
        seeds="11,17",
        repeated_runtime_preflight=True,
    )
    captured: dict[str, object] = {}

    def fake_repeated(*args: object, **kwargs: object) -> tuple[object, ...]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ()

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        benchmark_cli,
        "validate_repeated_external_benchmark_runtime",
        fake_repeated,
    )

    assert benchmark_cli.main() == 0
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["runtime_requirements"] is None
    assert kwargs["source_checkout_paths"] == {"WebShop": Path("/opt/webshop")}
    requirements = kwargs["source_checkout_requirements"]
    assert requirements["WebShop"].expected_revision == "abc123"


def test_measured_single_run_routes_source_contract_through_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path, probe_action="search")
    captured: dict[str, object] = {}
    report = object()

    def fake_run(*args: object, **kwargs: object) -> object:
        captured["run_kwargs"] = kwargs
        return report

    def fail_direct(*args: object, **kwargs: object) -> object:
        raise AssertionError("direct measured execution must not bypass source admission")

    def fake_persist(path: Path, **kwargs: object) -> Path:
        captured["persist_kwargs"] = kwargs
        return path

    monkeypatch.setattr(benchmark_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(benchmark_cli, "run_external_benchmark_with_preflight", fake_run)
    monkeypatch.setattr(benchmark_cli, "run_external_benchmark", fail_direct)
    monkeypatch.setattr(
        benchmark_cli,
        "collect_runtime_provenance",
        lambda **kwargs: SimpleNamespace(to_dict=lambda: {"code_revision": "remem123"}),
    )
    monkeypatch.setattr(benchmark_cli, "_persist_benchmark_bundle", fake_persist)

    assert benchmark_cli.main() == 0
    run_kwargs = captured["run_kwargs"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["probe_action"] == "search"
    assert run_kwargs["runtime_requirements"] is None
    assert run_kwargs["source_checkout_paths"] == {"WebShop": Path("/opt/webshop")}
    requirements = run_kwargs["source_checkout_requirements"]
    assert requirements["WebShop"].expected_revision == "abc123"
