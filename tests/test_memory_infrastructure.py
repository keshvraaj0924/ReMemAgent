from remem.memory.deduplication import MemoryDeduplicator
from remem.memory.retrieval import MemoryRetriever, RetrievalPolicy
from remem.memory.store import MemoryStore
from remem.memory.types import MemoryKind, MemoryRecord


def make_memory(memory_id: str, state: str, action: str, outcome: str, *, reward: float = 0.0) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        state=state,
        action=action,
        outcome=outcome,
        reward=reward,
    )


def test_store_rejects_duplicate_ids() -> None:
    memory = make_memory("m1", "open drawer", "open", "drawer opened")
    store = MemoryStore([memory])

    try:
        store.add(memory)
    except ValueError as error:
        assert "m1" in str(error)
    else:
        raise AssertionError("duplicate memory IDs must be rejected")


def test_retriever_ranks_by_token_overlap_deterministically() -> None:
    memories = [
        make_memory("m2", "open cabinet", "open", "cabinet opened"),
        make_memory("m1", "open drawer", "open", "drawer opened"),
    ]
    retriever = MemoryRetriever(RetrievalPolicy(top_k=2))

    results = retriever.retrieve("open drawer", memories)

    assert [result.memory.memory_id for result in results] == ["m1", "m2"]
    assert results[0].similarity > results[1].similarity


def test_deduplicator_keeps_stronger_evidence() -> None:
    weaker = make_memory("m1", "open drawer", "open drawer", "drawer opened", reward=0.4)
    stronger = make_memory("m2", "open drawer", "open drawer", "drawer opened", reward=0.9)

    result = MemoryDeduplicator().deduplicate([weaker, stronger])

    assert [memory.memory_id for memory in result] == ["m2"]


def test_failure_memory_can_be_retrieved_when_enabled() -> None:
    failure = MemoryRecord(
        memory_id="failure-1",
        state="closed cabinet",
        action="place object without opening cabinet",
        outcome="placement failed",
        kind=MemoryKind.FAILURE,
    )

    result = MemoryRetriever().retrieve("closed cabinet placement", [failure])

    assert result[0].memory.kind is MemoryKind.FAILURE
