from remem.benchmark import BenchmarkSuiteRunner
from remem.observability import ObservationCollector
from tests.test_benchmark import FakeEnvironment


def test_successful_suite_records_run_completion() -> None:
    collector = ObservationCollector()

    BenchmarkSuiteRunner(observation_collector=collector).run(
        benchmark_name="observability-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory=lambda index: FakeEnvironment(index),
        policy_factory=lambda index, store: lambda state: "act",
        success_evaluator=lambda episode: True,
    )

    snapshot = collector.snapshot()
    assert snapshot.counters["benchmark.runs"] == 1.0
    assert snapshot.counters["benchmark.runs.completed"] == 1.0


def test_failed_suite_does_not_record_run_completion() -> None:
    collector = ObservationCollector()

    def environment_factory(index: int) -> FakeEnvironment:
        raise RuntimeError("environment failed")

    try:
        BenchmarkSuiteRunner(observation_collector=collector).run(
            benchmark_name="observability-failure-smoke",
            episode_count=1,
            max_steps=1,
            environment_factory=environment_factory,
            policy_factory=lambda index, store: lambda state: "act",
            success_evaluator=lambda episode: True,
        )
    except RuntimeError as exc:
        assert str(exc) == "environment failed"
    else:
        raise AssertionError("failed suite should propagate its environment error")

    snapshot = collector.snapshot()
    assert snapshot.counters["benchmark.runs"] == 1.0
    assert snapshot.counters.get("benchmark.runs.completed", 0.0) == 0.0
