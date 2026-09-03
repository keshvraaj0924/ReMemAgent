"""Contract tests for memory-guided policy boundaries."""

import math

import pytest

from remem.memory.policy import MemoryGuidedPolicy
from remem.memory.store import MemoryStore


@pytest.mark.parametrize("minimum_trust", [True, False, "0.5"])
def test_memory_guided_policy_rejects_non_numeric_trust(minimum_trust: object) -> None:
    """Reject values that only appear numeric through Python coercion."""

    with pytest.raises(TypeError, match="minimum_trust"):
        MemoryGuidedPolicy(
            MemoryStore(),
            lambda state, guidance: "act",
            minimum_trust=minimum_trust,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("minimum_trust", [math.nan, math.inf, -math.inf])
def test_memory_guided_policy_rejects_non_finite_trust(minimum_trust: float) -> None:
    """Reject NaN and infinities before they enter retrieval thresholds."""

    with pytest.raises(ValueError, match="finite"):
        MemoryGuidedPolicy(
            MemoryStore(),
            lambda state, guidance: "act",
            minimum_trust=minimum_trust,
        )


def test_memory_guided_policy_rejects_non_callable_action_policy() -> None:
    """Require an explicit learned-component callable at construction."""

    with pytest.raises(TypeError, match="action_policy"):
        MemoryGuidedPolicy(MemoryStore(), object())  # type: ignore[arg-type]


def test_memory_guided_policy_rejects_non_callable_query_builder() -> None:
    """Require custom query builders to be callable before execution."""

    with pytest.raises(TypeError, match="query_builder"):
        MemoryGuidedPolicy(
            MemoryStore(),
            lambda state, guidance: "act",
            query_builder=object(),  # type: ignore[arg-type]
        )


def test_memory_guided_policy_rejects_non_string_state_before_retrieval() -> None:
    """Fail at the public state boundary rather than deep inside retrieval."""

    policy = MemoryGuidedPolicy(MemoryStore(), lambda state, guidance: "act")

    with pytest.raises(ValueError, match="current_state"):
        policy.select_guidance(123)  # type: ignore[arg-type]


def test_memory_guided_policy_rejects_non_string_action_result() -> None:
    """Prevent non-text actions from crossing the environment policy boundary."""

    policy = MemoryGuidedPolicy(
        MemoryStore(),
        lambda state, guidance: 1,  # type: ignore[return-value]
    )

    with pytest.raises(ValueError, match="action_policy"):
        policy("state")
