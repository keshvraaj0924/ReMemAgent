from remem.memory.lifecycle import MemoryLifecycle
from remem.memory.reconstruction import MemoryReconstructor
from remem.memory.types import MemoryKind, MemoryRecord, MemoryStatus
from remem.routing.adaptive_router import CounterfactualRouter, Route


def memory(successes=8, failures=2, confidence=0.8):
    return MemoryRecord(
        "m1",
        "Open the container before placing the object.",
        uses=successes + failures,
        successes=successes,
        failures=failures,
        confidence=confidence,
    )


def test_router_rejects_negative_transfer():
    result = CounterfactualRouter().decide(memory(), 0.1, 0.1, 0.8)
    assert result.route is Route.SELF
    assert result.expected_delta < 0


def test_reconstructor_rejects_mismatch():
    result = MemoryReconstructor().reconstruct(memory(), "unrelated state", 0.1)
    assert result.rejected is True


def test_consolidation_promotes_episodic_memories():
    records = [MemoryRecord(str(i), "rule", uses=3, successes=3, confidence=0.7) for i in range(3)]
    consolidated = MemoryLifecycle().consolidate(records, "semantic-1", "Validated rule")
    assert consolidated.kind is MemoryKind.SEMANTIC
    assert consolidated.outcome == "Validated rule"
    assert consolidated.state == ""
    assert all(r.status is MemoryStatus.RETIRED for r in records)
