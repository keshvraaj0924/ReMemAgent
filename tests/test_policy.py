"""Tests for memory-guided policy composition."""

from remem.memory import MemoryRecord, MemoryStore
from remem.memory.policy import MemoryGuidedPolicy
from remem.memory.types import MemoryKind


def test_memory_guided_policy_passes_best_guidance_to_action_policy() -> None:
    store = MemoryStore(
        [
            MemoryRecord(
                memory_id="memory-1",
                task="open cabinet",
                state="The cabinet is closed.",
                action="open cabinet",
                outcome="The cabinet opened.",
                kind=MemoryKind.SUCCESS,
                empirical_success_rate=1.0,
            )
        ]
    )
    calls: list[tuple[str, str]] = []

    def action_policy(state: str, guidance: str) -> str:
        calls.append((state, guidance))
        return "open cabinet"

    policy = MemoryGuidedPolicy(store, action_policy)

    assert policy("The cabinet is closed.") == "open cabinet"
    assert calls[0][0] == "The cabinet is closed."
    assert "Relevant action: open cabinet" in calls[0][1]


def test_memory_guided_policy_allows_action_without_matching_memory() -> None:
    store = MemoryStore()
    received_guidance: list[str] = []

    def action_policy(_: str, guidance: str) -> str:
        received_guidance.append(guidance)
        return "look"

    policy = MemoryGuidedPolicy(store, action_policy)

    assert policy("An unfamiliar room.") == "look"
    assert received_guidance == [""]


def test_memory_guided_policy_rejects_empty_action() -> None:
    policy = MemoryGuidedPolicy(MemoryStore(), lambda *_: "")

    try:
        policy("A valid state")
    except ValueError as error:
        assert str(error) == "action_policy must return a non-empty action"
    else:
        raise AssertionError("Expected empty action to be rejected")


def test_memory_guided_policy_validates_trust_threshold() -> None:
    try:
        MemoryGuidedPolicy(MemoryStore(), lambda *_: "look", minimum_trust=1.1)
    except ValueError as error:
        assert str(error) == "minimum_trust must be between 0 and 1"
    else:
        raise AssertionError("Expected invalid trust threshold to be rejected")
