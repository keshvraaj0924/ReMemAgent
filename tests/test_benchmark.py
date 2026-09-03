from remem.benchmark import BenchmarkRunConfiguration, BenchmarkSuiteRunner
from remem.environments.base import StepResult
from remem.execution import Policy
from remem.memory.store import MemoryStore
from remem.observability import ObservationCollector


class FakeEnvironment:
    def __init__(self, episode_index: int) -> None:
        self.episode_index = episode_index
        self.closed = False
        self.actions: list[str] = []

    def reset(self, **kwargs: object) -> str:
        return f"state-{self.episode_index}"

    def step(self, action: str) -> StepResult:
        self.actions.append(action)
        return StepResult(
            observation="done",
            reward=float(self.episode_index + 1),
            terminated=True,
            truncated=False,
        )

    def close(self) -> None:
        self.closed = True


class FailingEnvironment(FakeEnvironment):
    def step(self, action: str) -> StepResult:
        raise RuntimeError("step failed")


class CloseFailingEnvironment(FakeEnvironment):
    def close(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")


class StepAndCloseFailingEnvironment(FailingEnvironment):
    def close(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")


def test_benchmark_runner_shares_memory_store_and_closes_environments() -> None:
    environments: list[FakeEnvironment] = []
    seen_memory_counts: list[int] = []

    def environment_factory(index: int) -> FakeEnvironment:
        environment = FakeEnvironment(index)
        environments.append(environment)
        return environment

    def policy_factory(index: int, store: MemoryStore) -> Policy:
        seen_memory_counts.append(len(store.all()))
        return lambda state: f"act-{index}"

    report = BenchmarkSuiteRunner().run(
        benchmark_name="alfworld-smoke",
        episode_count=2,
        max_steps=2,
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        success_evaluator=lambda episode: episode.total_reward > 0,
    )

    assert report.benchmark_name == "alfworld-smoke"
    assert report.success_count == 2
    assert report.success_rate == 1.0
    assert report.mean_reward == 1.5
    assert report.final_memory_count == 2
    assert seen_memory_counts == [0, 1]
    assert all(environment.closed for environment in environments)


def test_benchmark_runner_records_observability_counters_and_duration() -> None:
    collector = ObservationCollector()

    report = BenchmarkSuiteRunner(observation_collector=collector).run(
        benchmark_name="observed-smoke",
        episode_count=2,
        max_steps=1,
        environment_factory=lambda index: FakeEnvironment(index),
        policy_factory=lambda index, store: lambda state: "act",
        success_evaluator=lambda episode: True,
    )

    snapshot = collector.snapshot()
    assert report.success_count == 2
    assert snapshot.counters["benchmark.runs"] == 1.0
    assert snapshot.counters["benchmark.episodes.started"] == 2.0
    assert snapshot.counters["benchmark.episodes.completed"] == 2.0
    assert snapshot.counters["benchmark.episodes.succeeded"] == 2.0
    assert snapshot.counters.get("benchmark.episodes.failed", 0.0) == 0.0
    assert snapshot.counters["benchmark.episodes.successful"] == 2.0
    assert snapshot.counters["benchmark.transfers.attributed"] == 0.0
    assert snapshot.durations_seconds["benchmark.episode.duration_seconds"] >= 0.0


def test_benchmark_runner_records_episode_failure_and_closes_environment() -> None:
    collector = ObservationCollector()
    environment = FailingEnvironment(0)

    def environment_factory(index: int) -> FailingEnvironment:
        return environment

    try:
        BenchmarkSuiteRunner(observation_collector=collector).run(
            benchmark_name="failure-smoke",
            episode_count=1,
            max_steps=1,
            environment_factory=environment_factory,
            policy_factory=lambda index, store: lambda state: "act",
            success_evaluator=lambda episode: True,
        )
    except RuntimeError as exc:
        assert str(exc) == "step failed"
    else:
        raise AssertionError("episode failure should be propagated")

    snapshot = collector.snapshot()
    assert snapshot.counters["benchmark.episodes.started"] == 1.0
    assert snapshot.counters["benchmark.episodes.succeeded"] == 0.0
    assert snapshot.counters["benchmark.episodes.failed"] == 1.0
    assert snapshot.counters.get("benchmark.episodes.completed", 0.0) == 0.0
    assert environment.closed


def test_benchmark_runner_records_factory_failure() -> None:
    collector = ObservationCollector()

    def environment_factory(index: int) -> FakeEnvironment:
        raise RuntimeError("factory failed")

    try:
        BenchmarkSuiteRunner(observation_collector=collector).run(
            benchmark_name="factory-failure-smoke",
            episode_count=1,
            max_steps=1,
            environment_factory=environment_factory,
            policy_factory=lambda index, store: lambda state: "act",
            success_evaluator=lambda episode: True,
        )
    except RuntimeError as exc:
        assert str(exc) == "factory failed"
    else:
        raise AssertionError("factory failure should be propagated")

    snapshot = collector.snapshot()
    assert snapshot.counters["benchmark.episodes.started"] == 1.0
    assert snapshot.counters["benchmark.episodes.failed"] == 1.0


def test_benchmark_runner_allows_zero_episodes() -> None:
    report = BenchmarkSuiteRunner().run(
        benchmark_name="webshop-smoke",
        episode_count=0,
        max_steps=1,
        environment_factory=lambda index: FakeEnvironment(index),
        policy_factory=lambda index, store: lambda state: "act",
        success_evaluator=lambda episode: True,
    )

    assert report.episodes == ()
    assert report.success_rate == 0.0
    assert report.mean_reward == 0.0
    assert report.final_memory_count == 0


def test_benchmark_runner_rejects_invalid_configuration() -> None:
    runner = BenchmarkSuiteRunner()
    common = {
        "benchmark_name": "test",
        "episode_count": 1,
        "max_steps": 1,
        "environment_factory": lambda index: FakeEnvironment(index),
        "policy_factory": lambda index, store: lambda state: "act",
        "success_evaluator": lambda episode: True,
    }

    for kwargs in (
        {**common, "benchmark_name": "   "},
        {**common, "episode_count": -1},
        {**common, "max_steps": 0},
    ):
        try:
            runner.run(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid benchmark configuration should fail")


def test_benchmark_runner_rejects_boolean_integer_inputs_before_factory_calls() -> None:
    factory_calls: list[str] = []

    def environment_factory(index: int) -> FakeEnvironment:
        factory_calls.append("environment")
        return FakeEnvironment(index)

    common = {
        "benchmark_name": "strict-input-smoke",
        "episode_count": 1,
        "max_steps": 1,
        "environment_factory": environment_factory,
        "policy_factory": lambda index, store: lambda state: "act",
        "success_evaluator": lambda episode: True,
    }

    for kwargs in (
        {**common, "episode_count": True},
        {**common, "max_steps": False},
        {**common, "seed": True},
    ):
        try:
            BenchmarkSuiteRunner().run(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("boolean integer inputs should be rejected")

    assert factory_calls == []


def test_benchmark_runner_rejects_non_callable_dependencies_before_execution() -> None:
    common = {
        "benchmark_name": "callable-contract-smoke",
        "episode_count": 1,
        "max_steps": 1,
    }

    invalid_inputs = (
        {**common, "environment_factory": object(), "policy_factory": lambda index, store: lambda state: "act", "success_evaluator": lambda episode: True},
        {**common, "environment_factory": lambda index: FakeEnvironment(index), "policy_factory": object(), "success_evaluator": lambda episode: True},
        {**common, "environment_factory": lambda index: FakeEnvironment(index), "policy_factory": lambda index, store: lambda state: "act", "success_evaluator": object()},
    )

    for kwargs in invalid_inputs:
        try:
            BenchmarkSuiteRunner().run(**kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError("non-callable benchmark dependencies should fail")


def test_benchmark_runner_rejects_stale_provenance_configuration() -> None:
    configuration = BenchmarkRunConfiguration(
        benchmark_name="webshop-smoke",
        episode_count=1,
        max_steps=1,
        seed=7,
    )

    try:
        BenchmarkSuiteRunner().run(
            benchmark_name="webshop-smoke",
            episode_count=2,
            max_steps=1,
            environment_factory=lambda index: FakeEnvironment(index),
            policy_factory=lambda index, store: lambda state: "act",
            success_evaluator=lambda episode: True,
            seed=7,
            configuration=configuration,
        )
    except ValueError as exc:
        assert "configuration.episode_count" in str(exc)
    else:
        raise AssertionError("stale provenance should fail rather than mislabel a run")


def test_benchmark_runner_accepts_matching_provenance_configuration() -> None:
    configuration = BenchmarkRunConfiguration(
        benchmark_name="webshop-smoke",
        episode_count=1,
        max_steps=1,
        seed=7,
        environment_factory="module:environment_factory",
    )

    report = BenchmarkSuiteRunner().run(
        benchmark_name="webshop-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory=lambda index: FakeEnvironment(index),
        policy_factory=lambda index, store: lambda state: "act",
        success_evaluator=lambda episode: True,
        seed=7,
        configuration=configuration,
    )

    assert report.configuration == configuration


def test_benchmark_runner_surfaces_cleanup_failure_when_episode_succeeds() -> None:
    collector = ObservationCollector()

    try:
        BenchmarkSuiteRunner(observation_collector=collector).run(
            benchmark_name="cleanup-failure-smoke",
            episode_count=1,
            max_steps=1,
            environment_factory=lambda index: CloseFailingEnvironment(index),
            policy_factory=lambda index, store: lambda state: "act",
            success_evaluator=lambda episode: True,
        )
    except RuntimeError as exc:
        assert str(exc) == "close failed"
    else:
        raise AssertionError("cleanup failure should fail a successful run")

    snapshot = collector.snapshot()
    assert snapshot.counters["benchmark.environment.close_failures"] == 1.0


def test_benchmark_runner_does_not_mask_episode_failure_with_cleanup_failure() -> None:
    collector = ObservationCollector()

    try:
        BenchmarkSuiteRunner(observation_collector=collector).run(
            benchmark_name="dual-failure-smoke",
            episode_count=1,
            max_steps=1,
            environment_factory=lambda index: StepAndCloseFailingEnvironment(index),
            policy_factory=lambda index, store: lambda state: "act",
            success_evaluator=lambda episode: True,
        )
    except RuntimeError as exc:
        assert str(exc) == "step failed"
    else:
        raise AssertionError("episode failure should remain the primary exception")

    snapshot = collector.snapshot()
    assert snapshot.counters["benchmark.episodes.failed"] == 1.0
    assert snapshot.counters["benchmark.environment.close_failures"] == 1.0


def test_benchmark_run_configuration_rejects_boolean_and_non_finite_trust() -> None:
    for minimum_trust in (True, float("nan"), float("inf")):
        try:
            BenchmarkRunConfiguration(
                benchmark_name="strict-config",
                episode_count=1,
                max_steps=1,
                seed=1,
                minimum_trust=minimum_trust,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid minimum_trust should be rejected")


def test_benchmark_run_configuration_rejects_boolean_integer_metadata() -> None:
    common = {
        "benchmark_name": "strict-config",
        "episode_count": 1,
        "max_steps": 1,
        "seed": 1,
    }

    for field_name, invalid_value in (
        ("episode_count", True),
        ("max_steps", False),
        ("seed", True),
    ):
        values = {**common, field_name: invalid_value}
        try:
            BenchmarkRunConfiguration(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"boolean {field_name} should be rejected")
