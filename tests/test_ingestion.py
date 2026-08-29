"""Tests for episode-to-memory ingestion."""

from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeRunner
from remem.memory import EpisodeMemoryAttribution, EpisodeMemoryIngestor
from remem.memory.store import MemoryStore


class FakeEnvironment:
    """Deterministic environment for ingestion integration tests."""

    def __init__(self, results: list[StepResult]) -> None:
        self._results = iter(results)

    def reset(self, **_: object) -> str:
        return "start"

    def step(self, action: str) -> StepResult:
        return next(self._results)

    def close(self) -> None:
        return None


def _episode() -> EpisodeResult:
    return EpisodeRunner().run(
        FakeEnvironment([StepResult("goal", 1.0, True, False, {})]),
        lambda _: "finish",
        max_steps=3,
    )


def test_ingestor_records_episode_memories_in_store() -> None:
    store = MemoryStore()
    result = EpisodeMemoryIngestor().ingest(
        store,
        episode_id="episode-1",
        episode=_episode(),
        attribution=EpisodeMemoryAttribution(episode_success=True),
    )

    assert [memory.memory_id for memory in result.retained_memories] == ["episode-1:step:0"]
    assert store.get("episode-1:step:0") is not None


def test_ingestor_skips_semantic_duplicates() -> None:
    store = MemoryStore()
    ingestor = EpisodeMemoryIngestor()
    episode = _episode()
    attribution = EpisodeMemoryAttribution(episode_success=True)

    first = ingestor.ingest(store, episode_id="episode-1", episode=episode, attribution=attribution)
    second = ingestor.ingest(store, episode_id="episode-2", episode=episode, attribution=attribution)

    assert len(first.retained_memories) == 1
    assert second.retained_memories == ()
    assert len(store) == 1
