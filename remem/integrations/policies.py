"""Factories that compose caller-owned model policies with ReMemAgent memory."""

from __future__ import annotations

from collections.abc import Callable

from remem.execution import Policy
from remem.memory.pipeline import MemoryGuidancePipeline
from remem.memory.policy import GuidedActionPolicy, MemoryGuidedPolicy
from remem.memory.store import MemoryStore

ActionPolicyFactory = Callable[[int], GuidedActionPolicy]
MemoryGuidedPolicyFactory = Callable[[int, MemoryStore], Policy]


def build_memory_guided_policy_factory(
    action_policy_factory: ActionPolicyFactory,
    *,
    pipeline: MemoryGuidancePipeline | None = None,
    minimum_trust: float = 0.0,
) -> MemoryGuidedPolicyFactory:
    """Compose a caller-owned action-policy factory with memory guidance.

    The external model factory remains responsible for loading checkpoints,
    tokenization, inference, and action decoding. ReMemAgent only supplies the
    shared memory store and deterministic guidance layer around that policy.
    """

    if not callable(action_policy_factory):
        raise TypeError("action_policy_factory must be callable")
    if not 0.0 <= minimum_trust <= 1.0:
        raise ValueError("minimum_trust must be between 0 and 1")

    def create_policy(seed: int, store: MemoryStore) -> Policy:
        """Create one memory-guided policy for a benchmark episode."""

        action_policy = action_policy_factory(seed)
        if not callable(action_policy):
            raise TypeError("action_policy_factory must return a callable action policy")
        return MemoryGuidedPolicy(
            store,
            action_policy,
            pipeline=pipeline,
            minimum_trust=minimum_trust,
        )

    return create_policy


__all__ = [
    "ActionPolicyFactory",
    "MemoryGuidedPolicyFactory",
    "build_memory_guided_policy_factory",
]
