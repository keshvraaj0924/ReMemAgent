from remem.benchmark import BenchmarkSuiteRunner
from remem.environments.base import StepResult
from remem.execution import Policy
from remem.memory.policy import MemoryGuidedPolicy
from remem.memory.store import MemoryStore


class FakeEnvironment:
    def __init__(self, episode_index: int) -> None:
        self.episode_index = episode_index
        self.closed = False

    def reset(self, **kwargs: object) -> str:
        return f"state-{self.episode_index}"

    def step(self, action: str) -> StepResult:
        return StepResult(
            observation="done",
            reward=1.0,
            terminated=True,
            truncated=False,
        )

    def close(self) -> None:
        self.closed = True


def test_benchmark_runner_traces_memory_guided_transfer() -> None:
    store = MemoryStore()

    def policy_factory(index: int, memory_store: MemoryStore) -> Policy:
        if index == 0:
            return lambda state: "bootstrap"
        return MemoryGuidedPolicy(
            memory_store,
            action_policy=lambda state, guidance: "guided",
            query_builder=lambda state: "state-0",
        )

    report = BenchmarkSuiteRunner().run(
        benchmark_name="guided-smoke",
        episode_count=2,
        max_steps=2,
        environment_factory=lambda index: FakeEnvironment(index),
        policy_factory=policy_factory,
        success_evaluator=lambda episode: episode.total_reward > 0,
        store=store,
    )

    second_episode = report.episodes[1]
    assert second_episode.transfer_count == 1
    assert second_episode.transfer_success_count == 1
    assert report.transfer_count == 1
    assert report.transfer_success_rate == 1.0

    transferred_memory = store.get(second_episode.transfer_outcomes[0].memory_id)
    assert transferred_memory is not None
    assert transferred_memory.transfer_attempts == 1
    assert transferred_memory.transfer_successes == 1


def test_benchmark_runner_does_not_attribute_non_guided_policies() -> None:
    report = BenchmarkSuiteRunner().run(
        benchmark_name="plain-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory=lambda index: FakeEnvironment(index),
        policy_factory=lambda index, store: lambda state: "plain",
        success_evaluator=lambda episode: True,
    )

    assert report.transfer_count == 0
    assert report.transfer_success_rate == 0.0
    assert report.episodes[0].transfer_outcomes == ()


def test_transfer_recorder_rejects_misaligned_decision_history() -> None:
    from remem.execution import EpisodeResult
    from remem.memory.attribution import MemoryTransferRecorder
    from remem.memory.policy import MemoryGuidanceDecision

    episode = EpisodeResult(
        initial_observation="state",
        steps=(),
        total_reward=0.0,
        terminated=False,
        truncated=False,
    )
    decision = MemoryGuidanceDecision(None, "", 0.0, 0.0)

    try:
        MemoryTransferRecorder().record_episode(MemoryStore(), (decision,), episode)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned decision history should fail")
