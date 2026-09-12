from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCRIPTS = {
    "remem-ablation": "experiments.cli:main",
    "remem-benchmark": "experiments.benchmark_cli:main",
    "remem-paired-benchmark": "experiments.paired_benchmark_cli:main",
    "remem-verify-benchmark": "experiments.verify_benchmark_artifact:main",
    "remem-verify-preflight": "experiments.verify_preflight_evidence:main",
    "remem-check-environments": "experiments.environment_cli:main",
}


def _load_project_scripts() -> dict[str, str]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    return project["project"]["scripts"]


def _resolve_entry_point(entry_point: str) -> object:
    module_name, separator, attribute_name = entry_point.partition(":")
    assert separator == ":"
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def test_console_scripts_match_supported_cli_contract() -> None:
    assert _load_project_scripts() == EXPECTED_SCRIPTS


def test_console_script_contract_is_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert _load_project_scripts() == EXPECTED_SCRIPTS


def test_console_script_entry_points_resolve() -> None:
    for entry_point in EXPECTED_SCRIPTS.values():
        target = _resolve_entry_point(entry_point)
        assert callable(target)
