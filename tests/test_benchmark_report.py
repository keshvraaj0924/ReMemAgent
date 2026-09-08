from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from experiments.benchmark_report import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    benchmark_configuration_fingerprint,
    benchmark_report_to_dict,
    save_benchmark_report,
    save_paired_benchmark_result,
    save_repeated_benchmark_reports,
)
from experiments.benchmark_statistics import BenchmarkConditionComparison, MetricSummary
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


def _build_comparison(seeds: tuple[int, ...] = (3, 17)) -> BenchmarkConditionComparison:
    summary = MetricSummary(
        mean=0.0,
        sample_stddev=0.0,
        standard_error=0.0,
        confidence_interval_95=(0.0, 0.0),
    )
