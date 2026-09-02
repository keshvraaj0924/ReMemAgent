"""Structural validation for benchmark reports before publication or analysis."""

from __future__ import annotations

from math import isfinite

from remem.benchmark import BenchmarkRunReport


def validate_benchmark_run_report(report: BenchmarkRunReport) -> None:
    """Validate the internal consistency of one benchmark run report.

    This is a structural integrity gate, not a scientific quality check. It
    verifies that persisted or externally assembled reports still describe the
    configuration under which they were produced and that trajectory-level
    fields are internally coherent.
    """

    if not report.benchmark_name.strip():
        raise ValueError("benchmark_name must not be empty")
    if report.final_memory_count < 0:
        raise ValueError("final_memory_count must be non-negative")

    episode_ids: set[str] = set()
    for episode in report.episodes:
        if not episode.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if episode.episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id!r}")
        episode_ids.add(episode.episode_id)

        trajectory = episode.episode
        if not trajectory.initial_observation.strip():
            raise ValueError("initial_observation must not be empty")
        if not isfinite(trajectory.total_reward):
            raise ValueError("total_reward must be finite")
        if episode.retained_memory_count < 0:
            raise ValueError("retained_memory_count must be non-negative")

        expected_step_indices = range(len(trajectory.steps))
        actual_step_indices = [step.step_index for step in trajectory.steps]
        if actual_step_indices != list(expected_step_indices):
            raise ValueError("episode step indices must be contiguous from zero")

        for step in trajectory.steps:
            if not step.observation.strip():
                raise ValueError("step observation must not be empty")
            if not step.action.strip():
                raise ValueError("step action must not be empty")
            if not isfinite(step.result.reward):
                raise ValueError("step reward must be finite")
            if not step.result.observation.strip():
                raise ValueError("step result observation must not be empty")

        if episode.transfer_count != len(episode.transfer_outcomes):
            raise ValueError("transfer_count is inconsistent with transfer outcomes")
        if episode.transfer_success_count > episode.transfer_count:
            raise ValueError("transfer success count cannot exceed transfer count")

    configuration = report.configuration
    if configuration is None:
        return
    if configuration.benchmark_name != report.benchmark_name:
        raise ValueError("configuration benchmark name does not match report")
    if configuration.episode_count != len(report.episodes):
        raise ValueError("configuration episode count does not match report")
    if configuration.seed != report.seed:
        raise ValueError("configuration seed does not match report")
    if configuration.episode_count < 0:
        raise ValueError("configuration episode_count must be non-negative")
    if configuration.max_steps <= 0:
        raise ValueError("configuration max_steps must be positive")
    if not 0.0 <= configuration.minimum_trust <= 1.0:
        raise ValueError("configuration minimum_trust must be between 0 and 1")
    for episode in report.episodes:
        if len(episode.episode.steps) > configuration.max_steps:
            raise ValueError("episode exceeds configured max_steps")
