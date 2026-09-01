"""Tests for explicit episode-to-memory attribution."""

from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep
from remem.memory.episode import EpisodeMemoryAttribution, EpisodeMemoryRecorder


def _episode() -> EpisodeResult:
    return EpisodeResult(
        initial_observation="start",
        steps=(
            EpisodeStep(0, "start", "open door", StepResult("room", 0.0, False, False, {})),
            EpisodeStep(1, "room", "take key", StepResult("key", 1.0, True, False, {})),
        ),
        total_reward=1.0,
        terminated=True,
        truncated=False,
    )


def test_recorder_creates_deterministic_episodic_records() -> None:
    memories = EpisodeMemoryRecorder().record(
        "episode-7",
        _episode(),
        EpisodeMemoryAttribution(episode_success=True),
    )

    assert [memory.memory_id for memory in memories] == [
        "episode-7:step:0",
        "episode-7:step:1",
    ]
    assert memories[0].state == "start"
    assert memories[0].action == "open door"
    assert memories[0].outcome == "room"
    assert memories[0].reward == 0.0
    assert memories[0].successes == 1
    assert memories[0].failures == 0
    assert memories[1].metadata["episode_id"] == "episode-7"


def test_recorder_attributes_failure_without_inventing_reward_semantics() -> None:
    memories = EpisodeMemoryRecorder().record(
        "episode-8",
        _episode(),
        EpisodeMemoryAttribution(episode_success=False),
    )

    assert all(memory.successes == 0 for memory in memories)
    assert all(memory.failures == 1 for memory in memories)
    assert memories[1].reward == 1.0


def test_recorder_returns_no_records_for_empty_episode() -> None:
    empty_episode = EpisodeResult("start", (), 0.0, False, False)

    assert (
        EpisodeMemoryRecorder().record(
            "episode-empty", empty_episode, EpisodeMemoryAttribution(False)
        )
        == []
    )


def test_recorder_rejects_blank_episode_id() -> None:
    try:
        EpisodeMemoryRecorder().record("  ", _episode(), EpisodeMemoryAttribution(True))
    except ValueError as error:
        assert "episode_id" in str(error)
    else:
        raise AssertionError("blank episode id should be rejected")
