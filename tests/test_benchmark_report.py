from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from experiments.benchmark_report import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    benchmark_configuration_fingerprint,
    benchmark_report_to_dict,
    save_benchmark_report,
    save_repeated_benchmark_reports,
)
from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunConfiguration, BenchmarkRunReport
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep
from remem.memory.store import MemoryStore


def _build_report(
    *,
    seed: int | None = None,
    max_steps: int = 1,
    minimum_trust: float = 0.0,
) -> BenchmarkRunReport:
    episode = EpisodeResult(
        initial_observation="start",
        steps=(
            EpisodeStep(
                step_index=0,
                observation="start",
                action="look",
                result=StepResult(
                    observation="next",
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    info={"opaque": object()},
                ),
            ),
        ),
        total_reward=1.0,
        terminated=True,
        truncated=False,
    )
    return BenchmarkRunReport(
        benchmark_name="alfworld-test",
        episodes=(
            BenchmarkEpisodeReport(
                episode_id="alfworld-test:0",
                episode=episode,
                episode_success=True,
                retained_memory_count=1,
            ),
        ),
        final_memory_count=1,
        seed=seed,
        configuration=BenchmarkRunConfiguration(
            benchmark_name="alfworld-test",
            episode_count=1,
            max_steps=max_steps,
            seed=seed,
            environment_factory="tests.test_benchmark_report:make_environment",
            policy_factory="tests.test_benchmark_report:make_policy",
            success_evaluator="tests.test_benchmark_report:evaluate_success",
            minimum_trust=minimum_trust,
        ),
    )


def make_environment(seed: int) -> object:
    """Provide an importable placeholder factory for provenance metadata tests."""

    del seed
    return object()


def make_policy(seed: int, store: MemoryStore) -> Callable[[str], str]:
    """Provide an importable policy factory for provenance metadata tests."""

    del seed, store
    return lambda state: "look"


def evaluate_success(episode: EpisodeResult) -> bool:
    """Provide an importable evaluator for provenance metadata tests."""

    return episode.total_reward > 0


def test_benchmark_report_to_dict_preserves_measured_core_fields() -> None:
    payload = benchmark_report_to_dict(_build_report())

    assert payload["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert payload["benchmark_name"] == "alfworld-test"
    assert payload["episodes"][0]["episode"]["total_reward"] == 1.0
    assert payload["episodes"][0]["episode"]["steps"][0]["action"] == "look"
    assert "info" not in payload["episodes"][0]["episode"]["steps"][0]["result"]


def test_benchmark_report_to_dict_adds_configuration_fingerprint() -> None:
    payload = benchmark_report_to_dict(_build_report(seed=17))

    assert payload["configuration_fingerprint"] == benchmark_configuration_fingerprint(
        _build_report(seed=17).configuration
    )


def test_benchmark_report_to_dict_rejects_invalid_report() -> None:
    report = _build_report()
    invalid_episode = report.episodes[0]
    invalid_report = BenchmarkRunReport(
        benchmark_name=report.benchmark_name,
        episodes=(
            BenchmarkEpisodeReport(
                episode_id=invalid_episode.episode_id,
                episode=invalid_episode.episode,
                episode_success=invalid_episode.episode_success,
                retained_memory_count=-1,
            ),
        ),
        final_memory_count=report.final_memory_count,
        seed=report.seed,
        configuration=report.configuration,
    )

    with pytest.raises(ValueError, match="retained_memory_count"):
        benchmark_report_to_dict(invalid_report)


def test_configuration_fingerprint_is_independent_of_seed() -> None:
    first = _build_report(seed=1).configuration
    second = _build_report(seed=2).configuration

    assert first is not None
    assert second is not None
    assert benchmark_configuration_fingerprint(first) == benchmark_configuration_fingerprint(second)


def test_configuration_fingerprint_changes_when_configuration_changes() -> None:
    first = _build_report(seed=1, max_steps=1).configuration
    second = _build_report(seed=1, max_steps=2).configuration

    assert first is not None
    assert second is not None
    assert benchmark_configuration_fingerprint(first) != benchmark_configuration_fingerprint(second)


def test_save_benchmark_report_writes_json(tmp_path) -> None:
    output_path = save_benchmark_report(_build_report(), tmp_path / "nested" / "report.json")

    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert persisted["final_memory_count"] == 1
    assert persisted["episodes"][0]["transfer_outcomes"] == []


def test_save_benchmark_report_includes_runtime_provenance(tmp_path) -> None:
    output_path = save_benchmark_report(
        _build_report(),
        tmp_path / "report.json",
        runtime_provenance={"code_revision": "abc123", "python_version": "3.11.0"},
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["runtime_provenance"] == {
        "code_revision": "abc123",
        "python_version": "3.11.0",
    }


def test_save_repeated_reports_rejects_different_experimental_configuration(tmp_path) -> None:
    first = _build_report(seed=1, max_steps=1)
    second = _build_report(seed=2, max_steps=2)

    with pytest.raises(ValueError, match="share configuration"):
        save_repeated_benchmark_reports((first, second), tmp_path / "reports.json")


def test_save_repeated_reports_allows_seed_only_configuration_difference(tmp_path) -> None:
    first = _build_report(seed=1)
    second = _build_report(seed=2)

    output_path = save_repeated_benchmark_reports((first, second), tmp_path / "reports.json")

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert persisted["seeds"] == [1, 2]
    assert persisted["configuration_fingerprint"] == benchmark_configuration_fingerprint(
        first.configuration
    )
