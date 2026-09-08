from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

from experiments import environment_cli
from remem.environments.dependencies import DependencyStatus


def test_main_reports_all_benchmarks_and_succeeds_when_required_dependency_is_available(
    capsys,
) -> None:
    statuses = (
        DependencyStatus("alfworld", True, "alfworld"),
        DependencyStatus("webshop", False, "webshop"),
    )

    with (
        patch.object(environment_cli, "parse_args", return_value=Namespace(require=["alfworld"])),
        patch.object(environment_cli, "check_benchmark_dependencies", return_value=statuses),
    ):
        result = environment_cli.main()

    assert result == 0
    assert capsys.readouterr().out == (
        "alfworld: available (import=alfworld)\nwebshop: missing (import=webshop)\n"
    )


def test_main_fails_when_required_dependency_is_missing(capsys) -> None:
    statuses = (
        DependencyStatus("alfworld", False, "alfworld"),
        DependencyStatus("webshop", True, "webshop"),
    )

    with (
        patch.object(environment_cli, "parse_args", return_value=Namespace(require=["alfworld"])),
        patch.object(environment_cli, "check_benchmark_dependencies", return_value=statuses),
    ):
        result = environment_cli.main()

    assert result == 1
    assert capsys.readouterr().out == (
        "alfworld: missing (import=alfworld)\n"
        "webshop: available (import=webshop)\n"
        "missing required benchmark dependencies: alfworld\n"
    )


def test_main_requires_all_benchmarks_by_default(capsys) -> None:
    statuses = (
        DependencyStatus("alfworld", True, "alfworld"),
        DependencyStatus("webshop", False, "webshop"),
    )

    with (
        patch.object(environment_cli, "parse_args", return_value=Namespace(require=None)),
        patch.object(environment_cli, "check_benchmark_dependencies", return_value=statuses),
    ):
        result = environment_cli.main()

    assert result == 1
    assert "missing required benchmark dependencies: webshop" in capsys.readouterr().out
