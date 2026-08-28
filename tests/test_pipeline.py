"""Tests for the composable memory-guidance pipeline."""

from remem.memory.pipeline import MemoryGuidancePipeline
from remem.memory.store import MemoryStore
from remem.memory.types import MemoryRecord, MemoryStatus


def make_memory(memory_id: str, action: str, outcome: str) -> MemoryRecord:
    return MemoryRecord(memory_id=memory_id, state="The cabinet is closed.", action=action, outcome=outcome, reward=1.0)


def test_pipeline_retrieves_scores_and_reconstructs_active_memory() -> None:
    store = MemoryStore([
        make_memory("cabinet-1", "open the cabinet", "the object can then be placed"),
        make_memory("unrelated-1", "wash the plate", "the plate becomes clean"),
    ])
    candidates = MemoryGuidancePipeline().build_candidates(
        store, query="open cabinet object", current_state="The cabinet is closed in front of the agent."
    )
    assert [candidate.memory_id for candidate in candidates] == ["cabinet-1"]
    assert candidates[0].trust.confidence > 0.0
    assert "Current state:" in candidates[0].reconstruction.guidance


def test_pipeline_excludes_low_trust_candidates() -> None:
    store = MemoryStore([make_memory("cabinet-1", "open cabinet", "placement succeeds")])
    candidates = MemoryGuidancePipeline().build_candidates(
        store, query="open cabinet", current_state="cabinet is closed", minimum_trust=1.0
    )
    assert candidates == []


def test_pipeline_ignores_non_active_memories() -> None:
    memory = make_memory("retired-1", "open cabinet", "success")
    memory.status = MemoryStatus.RETIRED
    store = MemoryStore([memory])
    candidates = MemoryGuidancePipeline().build_candidates(
        store, query="open cabinet", current_state="cabinet is closed"
    )
    assert candidates == []
