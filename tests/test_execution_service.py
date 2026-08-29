from remem.execution_service import EpisodeExecutionService
from remem.environments.base import StepResult
from remem.memory.store import MemoryStore


class FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def reset(self, **kwargs: object) -> str:
        return "state-0"

    def step(self, action: str) -> StepResult:
        self.actions.append(action)
        return StepResult(
            observation="state-1",
            reward=1.0,
            terminated=True,
            truncated=False,
        )


def test_execute_and_record_runs_episode_and_ingests_memory() -> None:
    environment = FakeEnvironment()
    store = MemoryStore()
    service = EpisodeExecutionService(
        success_evaluator=lambda episode: episode.total_reward > 0.0,
    )

    result = service.execute_and_record(
        environment,
        lambda state: "act",
        store,
        episode_id=" episode-1 ",
        max_steps=3,
    )

    assert result.episode.total_reward == 1.0
    assert environment.actions == ["act"]
    assert len(result.ingestion.retained_memories) == 1
    assert store.all()[0].memory_id == "episode-1:step:0"
    assert store.all()[0].successes == 1


def test_execute_and_record_preserves_explicit_failure_attribution() -> None:
    store = MemoryStore()
    service = EpisodeExecutionService(success_evaluator=lambda episode: False)

    result = service.execute_and_record(
        FakeEnvironment(),
        lambda state: "act",
        store,
        episode_id="episode-2",
        max_steps=1,
    )

    assert result.ingestion.retained_memories[0].successes == 0
    assert result.ingestion.retained_memories[0].failures == 1


def test_execute_and_record_rejects_blank_episode_id() -> None:
    service = EpisodeExecutionService(success_evaluator=lambda episode: True)

    try:
        service.execute_and_record(
            FakeEnvironment(),
            lambda state: "act",
            MemoryStore(),
            episode_id="   ",
            max_steps=1,
        )
    except ValueError as exc:
        assert str(exc) == "episode_id must not be empty"
    else:
        raise AssertionError("blank episode_id should be rejected")
