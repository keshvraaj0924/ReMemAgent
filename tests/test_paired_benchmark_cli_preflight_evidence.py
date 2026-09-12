"""Regression coverage for paired CLI machine-readable readiness evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import experiments.paired_benchmark_cli as paired_cli
from experiments.paired_source_preflight import ControlledPairedPreflightResult
from experiments.preflight_evidence import verify_controlled_paired_preflight_evidence
from experiments.runtime_provenance import RuntimeProvenance
from experiments.source_checkouts import SourceCheckoutProvenance


def _arguments(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "benchmark": "webshop",
        "episodes": 2,
        "max_steps": 4,
        "seeds": "11,17",
        "environment_factory": "tests.test_external_benchmark:make_environment",
        "success_evaluator": "tests.test_external_benchmark:evaluate_success",
        "transfer_success_evaluator": None,
        "baseline_policy_factory": "tests.test_external_benchmark:make_policy",
        "baseline_action_policy_factory": None,
        "treatment_policy_factory": "tests.test_external_benchmark:make_memory_policy",
        "treatment_action_policy_factory": None,
        "minimum_trust": 0.0,
        "baseline_label": "baseline",
        "treatment_label": "treatment",
        "output": tmp_path / "paired.json",
        "manifest": None,
        "overwrite": False,
        "probe_action": "search",
        "preflight_only": True,
        "preflight_evidence": tmp_path / "readiness.json",
        "strict_reproducibility": False,
        "require_code_revision": None,
        "require_clean_working_tree": False,
        "require_dependency_version": None,
        "source_checkout": [f"webshop={tmp_path / 'webshop'}"],
        "require_source_revision": [f"webshop={'c' * 40}"],
        "allow_dirty_source_checkout": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _controlled_result() -> ControlledPairedPreflightResult:
    return ControlledPairedPreflightResult(
        runtime_provenance=RuntimeProvenance(
            schema_version=1,
            code_revision="a" * 40,
            working_tree_state="clean",
            python_version="3.12.0",
            platform="test-platform",
            package_version="0.1.0",
            dependency_fingerprint="b" * 64,
            dependency_versions={"pytest": "8.0.0"},
        ),
        source_checkout_provenance={
            "webshop": SourceCheckoutProvenance(
                revision="c" * 40,
                working_tree_state="clean",
            )
        },
    )


def _forbid_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("readiness evidence must never trigger measured execution")

    monkeypatch.setattr(paired_cli, "run_controlled_paired_external_benchmarks", fail)
    monkeypatch.setattr(paired_cli, "run_paired_external_benchmarks_with_preflight", fail)
    monkeypatch.setattr(paired_cli, "save_paired_execution_result", fail)
    monkeypatch.setattr(paired_cli, "save_benchmark_artifact_manifest", fail)


def test_preflight_only_persists_exact_controlled_readiness_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    controlled_result = _controlled_result()
    observed: dict[str, object] = {}

    def controlled_preflight(baseline_spec, treatment_spec, seeds, **kwargs):
        observed["seeds"] = seeds
        observed.update(kwargs)
        return controlled_result

    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        paired_cli,
        "preflight_controlled_paired_external_benchmarks",
        controlled_preflight,
    )
    _forbid_measurement(monkeypatch)

    assert paired_cli.main() == 0

    payload = json.loads(arguments.preflight_evidence.read_text(encoding="utf-8"))
    verify_controlled_paired_preflight_evidence(payload)
    assert payload["runtime_provenance"] == controlled_result.runtime_provenance.to_dict()
    assert payload["source_checkout_provenance"]["webshop"]["revision"] == "c" * 40
    assert observed["seeds"] == (11, 17)
    assert not arguments.output.exists()


def test_preflight_evidence_requires_preflight_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path, preflight_only=False)
    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)
    _forbid_measurement(monkeypatch)

    with pytest.raises(SystemExit, match="--preflight-evidence requires --preflight-only"):
        paired_cli.main()

    assert not arguments.preflight_evidence.exists()


def test_preflight_evidence_requires_declared_source_checkouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(
        tmp_path,
        source_checkout=None,
        require_source_revision=None,
    )
    monkeypatch.setattr(paired_cli, "parse_args", lambda: arguments)
    _forbid_measurement(monkeypatch)

    with pytest.raises(SystemExit, match="requires declared source checkouts"):
        paired_cli.main()

    assert not arguments.preflight_evidence.exists()
