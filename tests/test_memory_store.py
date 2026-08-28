from remem.memory.store import InMemoryStore, RetrievalPolicy
from remem.memory.types import MemoryKind, MemoryRecord


def make_memory(memory_id: str, kind: MemoryKind = MemoryKind.EPISODIC) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        state="the agent sees a container",
        action="inspect the container",
        outcome="successful",
        kind=kind,
    )


def test_retrieve_returns_highest_similarity_first() -> None:
    store = InMemoryStore()
    store.extend([make_memory("memory_a"), make_memory("memory_b")])

    results = store.retrieve({"memory_a": 0.61, "memory_b": 0.94})

    assert [result.memory.memory_id for result in results] == ["memory_b", "memory_a"]


def test_retrieve_can_exclude_failure_memories() -> None:
    store = InMemoryStore()
    store.extend([make_memory("success"), make_memory("failure", MemoryKind.FAILURE)])

    results = store.retrieve(
        {"success": 0.9, "failure": 0.99},
        RetrievalPolicy(include_failures=False),
    )

    assert [result.memory.memory_id for result in results] == ["success"]
