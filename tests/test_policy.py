from __future__ import annotations

import math

from remem.memory.policy import MemoryGuidedPolicy, MemoryGuidanceDecision
from remem.memory.store import MemoryStore


def test_memory_guided_policy_returns_empty_trace_without_matching_memory() -> None:
    policy = MemoryGuidedPolicy(MemoryStore(), lambda *_: "look")

    decision = policy.select_guidance("An unfamiliar room.")

    assert decision == MemoryGuidanceDecision(None, "", 0.0, 0.0)
    assert policy("An unfamiliar room.") == "look"


def test_memory_guided_policy_rejects_empty_action() -> None:
    policy = MemoryGuidedPolicy(MemoryStore(), lambda *_: "")

    try:
        policy("A valid state")
    except ValueError as error:
        assert str(error) == "action_policy must return a non-empty string"
    else:
        raise AssertionError("Expected empty action to be rejected")


def test_memory_guided_policy_validates_trust_threshold() -> None:
    try:
        MemoryGuidedPolicy(MemoryStore(), lambda *_: "look", minimum_trust=1.1)
    except ValueError as error:
        assert "minimum_trust" in str(error)
    else:
        raise AssertionError("Expected invalid trust threshold to be rejected")


def test_memory_guided_policy_rejects_non_finite_trust_threshold() -> None:
    for minimum_trust in (math.nan, math.inf, -math.inf):
        try:
            MemoryGuidedPolicy(MemoryStore(), lambda *_: "look", minimum_trust=minimum_trust)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("Expected non-finite trust threshold to be rejected")
