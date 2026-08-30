from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep
from remem.memory.attribution import MemoryTransferRecorder
from remem.memory.policy import MemoryGuidanceDecision
from remem.memory.store import MemoryStore
from remem.memory.types import MemoryRecord


def _decision(memory_id: str | None) -> MemoryGuidanceDecision:
    return MemoryGuidanceDecision(
        memory_id=memory_id,
        guidance="test guidance" if memory_id else "",
        similarity=0.9 if memory_id else 0.0,
        trust_confidence=0.9 if memory_id else 0.0,
    )


def test_record_episode_counts_only_terminal_measured_transfer() -> None:
    store = MemoryStore([MemoryRecord(memory_id="memory_a", state="state")])
    episode = EpisodeResult(
        initial_observation="start",
        steps=(
            EpisodeStep(0, "start", "move", StepResult("next", 0.0, False, False)),
            EpisodeStep(1, "next", "finish", StepResult("done", 1.0, True, False)),
        ),
        total_reward=1.0,
        terminated=True,
        truncated=False,
    )

    outcomes = MemoryTransferRecorder().record_episode(
        store,
        [_decision("memory_a"), _decision("memory_a")],
        episode,
    )

    assert len(outcomes) == 1
    assert outcomes[0].memory_id == "memory_a"
    memory = store.get("memory_a")
    assert memory is not None
    assert memory.transfer_attempts == 1
    assert memory.transfer_successes == 1


def test_record_episode_does_not_count_intermediate_step_as_transfer_failure() -> None:
    store = MemoryStore([MemoryRecord(memory_id="memory_a", state="state")])
    episode = EpisodeResult(
        initial_observation="start",
        steps=(EpisodeStep(0, "start", "move", StepResult("next", 0.0, False, False)),),
        total_reward=0.0,
        terminated=False,
        truncated=False,
    )

    outcomes = MemoryTransferRecorder().record_episode(
        store,
        [_decision("memory_a")],
        episode,
    )

    assert outcomes == ()
    memory = store.get("memory_a")
    assert memory is not None
    assert memory.transfer_attempts == 0
    assert memory.failures == 0
