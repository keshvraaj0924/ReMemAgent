from remem.memory.deduplication import DeduplicationPolicy, MemoryDeduplicator, lexical_similarity
from remem.memory.types import MemoryRecord


def make_memory(memory_id: str, content: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        state="test state",
        action="test action",
        outcome=content,
    )


def test_lexical_similarity_is_symmetric() -> None:
    left = "open the drawer before placing the object"
    right = "place the object after opening the drawer"

    assert lexical_similarity(left, right) == lexical_similarity(right, left)


def test_threshold_constructor_matches_explicit_policy() -> None:
    threshold_constructor = MemoryDeduplicator(0.9)
    policy_constructor = MemoryDeduplicator(DeduplicationPolicy(0.9))

    assert threshold_constructor.policy == policy_constructor.policy


def test_deduplicator_rejects_near_identical_memory() -> None:
    candidate = make_memory("candidate", "open drawer before placing object")
    existing = [make_memory("existing", "open drawer before placing object")]

    assert MemoryDeduplicator(0.9).is_duplicate(candidate, existing)


def test_deduplicator_accepts_distinct_memory() -> None:
    candidate = make_memory("candidate", "verify the final state after moving an object")
    existing = [make_memory("existing", "open the drawer before placing an object")]

    assert not MemoryDeduplicator(0.9).is_duplicate(candidate, existing)
