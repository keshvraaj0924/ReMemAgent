from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from experiments.benchmark_cli import _parse_seeds
from experiments.paired_benchmark_cli import _build_spec, _parse_seeds as parse_paired_seeds
from experiments.paired_benchmark_cli import _prepare_output_path, parse_args


def _arguments(**overrides) -> Namespace:
    values = {
        "benchmark": "synthetic-eval",
        "episodes": 2,
        "max_steps": 4,
        "environment_factory": "tests.test_external_benchmark:make_environment",
        "success_evaluator": "tests.test_external_benchmark:evaluate_success",
        "transfer_success_evaluator": None,
        "minimum_trust": 0.25,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_spec_supports_baseline_action_policy_factory() -> None:
    spec = _build_spec(
        _arguments(),
        policy_factory=None,
        action_policy_factory="tests.test_external_benchmark:make_policy",
    )

    assert spec.policy_factory is None
    assert spec.action_policy_factory == "tests.test_external_benchmark:make_policy"
    assert spec.minimum_trust == 0.25
    assert spec.seed is None


def test_build_spec_supports_treatment_policy_factory() -> None:
    spec = _build_spec(
        _arguments(),
        policy_factory="tests.test_external_benchmark:make_memory_policy",
        action_policy_factory=None,
    )

    assert spec.policy_factory == "tests.test_external_benchmark:make_memory_policy"
    assert spec.action_policy_factory is None


def test_paired_seed_parser_rejects_empty_parts() -> None:
    try:
        parse_paired_seeds("11,,17")
    except ValueError as exc:
        assert "comma-separated integers" in str(exc)
    else:
        raise AssertionError("expected malformed seed list to be rejected")


def test_paired_seed_parser_preserves_requested_order() -> None:
    assert parse_paired_seeds("17,11,29") == (17, 11, 29)


def test_existing_seed_parser_remains_unchanged() -> None:
    assert _parse_seeds("17,11") == (17, 11)


def test_parse_args_requires_one_baseline_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-paired-benchmark",
            "--benchmark",
            "synthetic-eval",
            "--episodes",
            "2",
            "--max-steps",
            "4",
            "--seeds",
            "11,17",
            "--environment-factory",
            "tests.test_external_benchmark:make_environment",
            "--success-evaluator",
            "tests.test_external_benchmark:evaluate_success",
            "--treatment-policy-factory",
            "tests.test_external_benchmark:make_memory_policy",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_both_policy_factory_forms_for_one_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-paired-benchmark",
            "--benchmark",
            "synthetic-eval",
            "--episodes",
            "2",
            "--max-steps",
            "4",
            "--seeds",
            "11,17",
            "--environment-factory",
            "tests.test_external_benchmark:make_environment",
            "--success-evaluator",
            "tests.test_external_benchmark:evaluate_success",
            "--baseline-policy-factory",
            "tests.test_external_benchmark:make_policy",
            "--baseline-action-policy-factory",
            "tests.test_external_benchmark:make_policy",
            "--treatment-policy-factory",
            "tests.test_external_benchmark:make_memory_policy",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_prepare_output_path_rejects_existing_file_without_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "paired.json"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        _prepare_output_path(output_path, overwrite=False)


def test_prepare_output_path_allows_existing_file_with_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "paired.json"
    output_path.write_text("existing", encoding="utf-8")

    assert _prepare_output_path(output_path, overwrite=True) == output_path


def test_parse_args_exposes_overwrite_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "remem-paired-benchmark",
            "--benchmark",
            "synthetic-eval",
            "--episodes",
            "2",
            "--max-steps",
            "4",
            "--seeds",
            "11,17",
            "--environment-factory",
            "tests.test_external_benchmark:make_environment",
            "--success-evaluator",
            "tests.test_external_benchmark:evaluate_success",
            "--baseline-policy-factory",
            "tests.test_external_benchmark:make_policy",
            "--treatment-policy-factory",
            "tests.test_external_benchmark:make_memory_policy",
            "--overwrite",
        ],
    )

    arguments = parse_args()

    assert arguments.overwrite is True
