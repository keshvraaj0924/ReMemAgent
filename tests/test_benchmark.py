from remem.benchmark import BenchmarkSuiteRunner
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
    assert snapshot.counters["benchmark.episodes.successful"] == 2.0
    assert snapshot.counters["benchmark.transfers.attributed"] == 0.0
    assert snapshot.durations_seconds["benchmark.episode.duration_seconds"] >= 0.0


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
