from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep
from remem.memory.attribution import MemoryTransferRecorder
from remem.memory.policy import MemoryGuidanceDecision
from remem.memory.store import MemoryStore
from remem.memory.types import MemoryRecord


def _decision(memory_id: str | None) -> MemoryGuidanceDecision:
    return MemoryGuidanceDecision(
        route="memory" if memory_id is not None else "self_reasoning",
        memory_id=memory_id,
        confidence=0.9,
        rationale="test",
    )


def test_record_episode_counts_only_terminal_measured_transfer() -> None:
    store = MemoryStore([MemoryRecord(memory_id="memory_a", state="state")])
    episode = EpisodeResult(
        initial_observation="start",
        steps=(
            EpisodeStep(
                step_index=0,
                observation="start",
                action="move",
                result=StepResult("next", 0.0, False, False),
            ),
            EpisodeStep(
                step_index=1,
                observation="next",
                action="finish",
                result=StepResult("done", 1.0, True, False),
            ),
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


def test_record_episode_does_not_count_intermediate_failure_as_transfer_failure() -> None:
    store = MemoryStore([MemoryRecord(memory_id="memory_a", state="state")])
    episode = EpisodeResult(
        initial_observation="start",
        steps=(
            EpisodeStep(
                step_index=0,
                observation="start",
                action="move",
                result=StepResult("next", 0.0, False, False),
            ),
        ),
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
