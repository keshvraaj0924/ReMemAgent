import pytest

from remem.benchmark import BenchmarkEpisodeReport, BenchmarkRunConfiguration, BenchmarkRunReport
from remem.benchmark_validation import validate_benchmark_run_report
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _report(*, configuration: BenchmarkRunConfiguration | None = None) -> BenchmarkRunReport:
    episode = EpisodeResult(
        initial_observation="task",
        steps=(EpisodeStep(0, "task", "look", StepResult("done", 1.0, True, False)),),
        total_reward=1.0,
        terminated=True,
        truncated=False,
    )
    return BenchmarkRunReport(
        benchmark_name="synthetic",
        episodes=(BenchmarkEpisodeReport("synthetic:0", episode, True, 1),),
        final_memory_count=1,
        seed=7,
        configuration=configuration,
    )


def test_valid_report_passes_validation() -> None:
    validate_benchmark_run_report(_report())


def test_validation_rejects_duplicate_episode_ids() -> None:
    report = _report()
    duplicate = report.episodes[0]
    invalid = BenchmarkRunReport("synthetic", (duplicate, duplicate), 1, 7)
    with pytest.raises(ValueError, match="duplicate episode_id"):
        validate_benchmark_run_report(invalid)


def test_validation_rejects_non_contiguous_steps() -> None:
    report = _report()
    step = report.episodes[0].episode.steps[0]
    episode = EpisodeResult("task", (EpisodeStep(2, "task", "look", step.result),), 1.0, True, False)
    invalid = BenchmarkRunReport(
        "synthetic",
        (BenchmarkEpisodeReport("synthetic:0", episode, True, 0),),
        0,
        7,
    )
    with pytest.raises(ValueError, match="step indices"):
        validate_benchmark_run_report(invalid)


def test_validation_rejects_configuration_mismatch() -> None:
    configuration = BenchmarkRunConfiguration("synthetic", 2, 2, 7)
    with pytest.raises(ValueError, match="episode count"):
        validate_benchmark_run_report(_report(configuration=configuration))


def test_validation_rejects_episode_exceeding_max_steps() -> None:
    report = _report(configuration=BenchmarkRunConfiguration("synthetic", 1, 1, 7))
    step = report.episodes[0].episode.steps[0]
    episode = EpisodeResult(
        "task",
        (step, EpisodeStep(1, "done", "finish", StepResult("done", 0.0, True, False))),
        1.0,
        True,
        False,
    )
    invalid = BenchmarkRunReport(
        "synthetic",
        (BenchmarkEpisodeReport("synthetic:0", episode, True, 0),),
        0,
        7,
        report.configuration,
    )
    with pytest.raises(ValueError, match="exceeds configured max_steps"):
        validate_benchmark_run_report(invalid)
