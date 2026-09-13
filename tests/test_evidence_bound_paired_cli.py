"""Regression coverage for evidence-bound paired CLI delegation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import experiments.evidence_bound_paired_cli as evidence_cli
from experiments.research_experiment_plan import (
    build_research_experiment_plan,
    write_research_experiment_plan,
)

REMEM_REVISION = "1" * 40
SOURCE_REVISION = "2" * 40


def _valid_readiness_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "runtime_provenance": {},
        "source_checkout_requirements": {},
        "source_checkout_requirements_sha256": "a" * 64,
        "source_checkout_provenance": {},
        "source_checkout_provenance_sha256": "b" * 64,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    payload["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _measurement_argv(tmp_path: Path) -> list[str]:
    return [
        "remem-paired-benchmark",
        "--benchmark",
        "webshop",
        "--episodes",
        "10",
        "--max-steps",
        "30",
        "--seeds",
        "11,17",
        "--environment-factory",
        "study.environments:build_webshop",
        "--success-evaluator",
        "study.metrics:is_success",
        "--baseline-policy-factory",
        "study.policies:build_baseline",
        "--treatment-action-policy-factory",
        "study.policies:build_remem_action",
        "--minimum-trust",
        "0.6",
        "--baseline-label",
        "baseline",
        "--treatment-label",
        "remem",
        "--strict-reproducibility",
        "--require-code-revision",
        REMEM_REVISION,
        "--require-clean-working-tree",
        "--require-dependency-version",
        "torch==2.6.0",
        "--source-checkout",
        f"webshop={tmp_path / 'webshop'}",
        "--require-source-revision",
        f"webshop={SOURCE_REVISION}",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--output",
        str(tmp_path / "paired.json"),
    ]


def _write_matching_plan(tmp_path: Path) -> Path:
    plan = build_research_experiment_plan(
        experiment_name="webshop-memory-study",
        remem_revision=REMEM_REVISION,
        benchmark_name="webshop",
        episode_count=10,
        max_steps=30,
        seeds=(11, 17),
        environment_factory="study.environments:build_webshop",
        success_evaluator="study.metrics:is_success",
        source_revisions={"webshop": SOURCE_REVISION},
        dependency_versions={"torch": "2.6.0"},
        baseline_policy_factory="study.policies:build_baseline",
        treatment_action_policy_factory="study.policies:build_remem_action",
        minimum_trust=0.6,
        baseline_label="baseline",
        treatment_label="remem",
    )
    plan_path = tmp_path / "research-plan.json"
    write_research_experiment_plan(plan_path, plan)
    return plan_path


def test_main_delegates_unchanged_without_evidence_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_argv: list[str] = []

    def delegated_main() -> int:
        observed_argv.extend(sys.argv)
        return 17

    monkeypatch.setattr(evidence_cli.paired_cli, "main", delegated_main)
    monkeypatch.setattr(sys, "argv", ["remem-paired-benchmark", "--benchmark", "webshop"])

    assert evidence_cli.main() == 17
    assert observed_argv == ["remem-paired-benchmark", "--benchmark", "webshop"]


def test_main_binds_controlled_measurement_and_persistence_to_readiness_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_payload = _valid_readiness_payload()
    readiness_path.write_text(json.dumps(readiness_payload), encoding="utf-8")
    original_runner = evidence_cli.paired_cli.run_controlled_paired_external_benchmarks
    observed: dict[str, object] = {}
    source_path = tmp_path / "webshop"
    source_revision = "c" * 40

    def evidence_bound_runner(readiness_evidence, *args, **kwargs):
        observed["evidence"] = readiness_evidence
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "controlled-result"

    def fake_saver(*args, **kwargs):
        observed["save_args"] = args
        observed["save_kwargs"] = kwargs
        return tmp_path / "paired.json"

    def delegated_main() -> int:
        observed["delegated_argv"] = list(sys.argv)
        observed["result"] = evidence_cli.paired_cli.run_controlled_paired_external_benchmarks(
            "baseline",
            "treatment",
            (11, 17),
            runtime_requirements="runtime",
            source_checkout_paths={"webshop": tmp_path},
            source_checkout_requirements={"webshop": "source"},
        )
        evidence_cli.paired_cli.save_paired_execution_result(
            "paired-result",
            tmp_path / "paired.json",
            runtime_provenance={"code_revision": "abc123"},
        )
        return 0

    monkeypatch.setattr(
        evidence_cli,
        "run_evidence_bound_paired_external_benchmarks",
        evidence_bound_runner,
    )
    monkeypatch.setattr(evidence_cli.paired_cli, "save_paired_execution_result", fake_saver)
    monkeypatch.setattr(evidence_cli.paired_cli, "main", delegated_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--source-checkout",
            f"webshop={source_path}",
            "--require-source-revision",
            f"webshop={source_revision}",
            "--benchmark",
            "webshop",
        ],
    )

    assert evidence_cli.main() == 0
    assert observed["evidence"] == readiness_payload
    assert observed["result"] == "controlled-result"
    assert observed["delegated_argv"] == [
        "remem-paired-benchmark",
        "--source-checkout",
        f"webshop={source_path}",
        "--require-source-revision",
        f"webshop={source_revision}",
        "--benchmark",
        "webshop",
    ]
    save_kwargs = observed["save_kwargs"]
    assert isinstance(save_kwargs, dict)
    runtime_provenance = save_kwargs["runtime_provenance"]
    assert isinstance(runtime_provenance, dict)
    assert runtime_provenance["code_revision"] == "abc123"
    assert (
        runtime_provenance[evidence_cli.PREFLIGHT_EVIDENCE_PROVENANCE_KEY]
        == readiness_payload["evidence_sha256"]
    )
    assert evidence_cli.paired_cli.run_controlled_paired_external_benchmarks is original_runner
    assert evidence_cli.paired_cli.save_paired_execution_result is fake_saver


def test_main_binds_matching_frozen_plan_digest_into_persisted_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_payload = _valid_readiness_payload()
    readiness_path.write_text(json.dumps(readiness_payload), encoding="utf-8")
    plan_path = _write_matching_plan(tmp_path)
    expected_plan = evidence_cli._load_experiment_plan(plan_path)
    observed: dict[str, object] = {}

    def evidence_bound_runner(readiness_evidence, *args, **kwargs):
        observed["evidence"] = readiness_evidence
        return "controlled-result"

    def fake_saver(*args, **kwargs):
        observed["save_kwargs"] = kwargs
        return tmp_path / "paired.json"

    def delegated_main() -> int:
        observed["delegated_argv"] = list(sys.argv)
        evidence_cli.paired_cli.run_controlled_paired_external_benchmarks(
            "baseline",
            "treatment",
            (11, 17),
            runtime_requirements="runtime",
            source_checkout_paths={"webshop": tmp_path},
            source_checkout_requirements={"webshop": "source"},
        )
        evidence_cli.paired_cli.save_paired_execution_result(
            "paired-result",
            tmp_path / "paired.json",
            runtime_provenance={"code_revision": REMEM_REVISION},
        )
        return 0

    monkeypatch.setattr(
        evidence_cli,
        "run_evidence_bound_paired_external_benchmarks",
        evidence_bound_runner,
    )
    monkeypatch.setattr(evidence_cli.paired_cli, "save_paired_execution_result", fake_saver)
    monkeypatch.setattr(evidence_cli.paired_cli, "main", delegated_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *_measurement_argv(tmp_path),
            "--require-preflight-evidence",
            str(readiness_path),
            "--require-experiment-plan",
            str(plan_path),
        ],
    )

    assert evidence_cli.main() == 0
    delegated_argv = observed["delegated_argv"]
    assert isinstance(delegated_argv, list)
    assert "--require-preflight-evidence" not in delegated_argv
    assert "--require-experiment-plan" not in delegated_argv
    save_kwargs = observed["save_kwargs"]
    assert isinstance(save_kwargs, dict)
    runtime_provenance = save_kwargs["runtime_provenance"]
    assert isinstance(runtime_provenance, dict)
    assert (
        runtime_provenance[evidence_cli.PREFLIGHT_EVIDENCE_PROVENANCE_KEY]
        == readiness_payload["evidence_sha256"]
    )
    assert runtime_provenance[evidence_cli.EXPERIMENT_PLAN_PROVENANCE_KEY] == expected_plan.sha256


def test_frozen_plan_contract_rejects_cli_drift(tmp_path: Path) -> None:
    plan = evidence_cli._load_experiment_plan(_write_matching_plan(tmp_path))
    argv = _measurement_argv(tmp_path)
    argv[argv.index("10")] = "12"
    arguments = evidence_cli._parse_delegated_arguments(argv)

    with pytest.raises(ValueError, match="episode_count"):
        evidence_cli._validate_experiment_plan_contract(plan, arguments)


def test_frozen_plan_contract_rejects_unpinned_or_dirty_execution(tmp_path: Path) -> None:
    plan = evidence_cli._load_experiment_plan(_write_matching_plan(tmp_path))
    argv = _measurement_argv(tmp_path)
    argv.remove("--require-clean-working-tree")
    arguments = evidence_cli._parse_delegated_arguments(argv)

    with pytest.raises(ValueError, match="require-clean-working-tree"):
        evidence_cli._validate_experiment_plan_contract(plan, arguments)


def test_main_rejects_experiment_plan_without_readiness_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_matching_plan(tmp_path)
    monkeypatch.setattr(
        evidence_cli.paired_cli,
        "main",
        lambda: pytest.fail("plan-only measurement must not delegate"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [*_measurement_argv(tmp_path), "--require-experiment-plan", str(plan_path)],
    )

    with pytest.raises(SystemExit, match="requires --require-preflight-evidence"):
        evidence_cli.main()


def test_main_rejects_tampered_evidence_before_delegation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_payload = _valid_readiness_payload()
    readiness_payload["runtime_provenance"] = {"code_revision": "drifted"}
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(readiness_payload), encoding="utf-8")
    monkeypatch.setattr(
        evidence_cli.paired_cli,
        "main",
        lambda: pytest.fail("tampered readiness evidence must block CLI delegation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--source-checkout=webshop=/tmp/webshop",
            f"--require-source-revision=webshop={'c' * 40}",
        ],
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evidence_cli.main()


def test_main_rejects_evidence_binding_for_preflight_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            f"--require-preflight-evidence={readiness_path}",
            "--preflight-only",
        ],
    )

    with pytest.raises(SystemExit, match="for measured execution"):
        evidence_cli.main()


def test_main_rejects_evidence_binding_without_controlled_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--benchmark",
            "webshop",
        ],
    )

    with pytest.raises(SystemExit, match="requires declared source checkouts"):
        evidence_cli.main()


def test_main_rejects_non_object_readiness_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remem-paired-benchmark",
            "--require-preflight-evidence",
            str(readiness_path),
            "--source-checkout=webshop=/tmp/webshop",
            f"--require-source-revision=webshop={'c' * 40}",
        ],
    )

    with pytest.raises(SystemExit, match="root must be a JSON object"):
        evidence_cli.main()


def test_extract_readiness_evidence_rejects_duplicates() -> None:
    with pytest.raises(SystemExit, match="may be specified only once"):
        evidence_cli._extract_readiness_evidence_path(
            [
                "remem-paired-benchmark",
                "--require-preflight-evidence=first.json",
                "--require-preflight-evidence",
                "second.json",
            ]
        )


def test_extract_experiment_plan_rejects_duplicates() -> None:
    with pytest.raises(SystemExit, match="may be specified only once"):
        evidence_cli._extract_experiment_plan_path(
            [
                "remem-paired-benchmark",
                "--require-experiment-plan=first.json",
                "--require-experiment-plan",
                "second.json",
            ]
        )
