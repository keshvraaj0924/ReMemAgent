from __future__ import annotations

import json

from experiments.benchmark_report import save_benchmark_report, save_repeated_benchmark_reports
from experiments.experiment_identity import build_experiment_identity

from test_benchmark_report import _build_report


RUNTIME_PROVENANCE = {
    "code_revision": "abc123",
    "python_version": "3.12.0",
}


def test_single_report_persists_experiment_identity(tmp_path) -> None:
    report = _build_report(seed=17)

    output_path = save_benchmark_report(
        report,
        tmp_path / "report.json",
        runtime_provenance=RUNTIME_PROVENANCE,
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["experiment_identity"] == build_experiment_identity(
        report.configuration,
        (17,),
        RUNTIME_PROVENANCE,
    )


def test_repeated_report_identity_is_independent_of_input_order(tmp_path) -> None:
    reports = (_build_report(seed=17), _build_report(seed=3))

    first_path = save_repeated_benchmark_reports(
        reports,
        tmp_path / "first.json",
        runtime_provenance=RUNTIME_PROVENANCE,
    )
    second_path = save_repeated_benchmark_reports(
        tuple(reversed(reports)),
        tmp_path / "second.json",
        runtime_provenance=RUNTIME_PROVENANCE,
    )

    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first["experiment_identity"] == second["experiment_identity"]


def test_repeated_report_identity_changes_when_seed_set_changes(tmp_path) -> None:
    first_path = save_repeated_benchmark_reports(
        (_build_report(seed=3), _build_report(seed=17)),
        tmp_path / "first.json",
        runtime_provenance=RUNTIME_PROVENANCE,
    )
    second_path = save_repeated_benchmark_reports(
        (_build_report(seed=3), _build_report(seed=19)),
        tmp_path / "second.json",
        runtime_provenance=RUNTIME_PROVENANCE,
    )

    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first["experiment_identity"] != second["experiment_identity"]
