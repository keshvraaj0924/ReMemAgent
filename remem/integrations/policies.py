"""Factories that compose caller-owned model policies with ReMemAgent memory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from remem.execution import Policy
from remem.memory.pipeline import MemoryGuidancePipeline
from remem.memory.policy import GuidedActionPolicy, MemoryGuidedPolicy
from remem.memory.store import MemoryStore

ActionPolicyFactory = Callable[[int], GuidedActionPolicy]
MemoryGuidedPolicyFactory = Callable[[int, MemoryStore], Policy]


@dataclass(frozen=True, slots=True)
class PolicyContractReport:
    """Observed output from a policy contract probe."""

    action: str


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
    _validate_minimum_trust(minimum_trust)

    def create_policy(seed: int, store: MemoryStore) -> Policy:
        """Create one memory-guided policy for a benchmark episode."""

        _validate_seed(seed)
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


def validate_policy_contract(
    policy_factory: MemoryGuidedPolicyFactory,
    *,
    seed: int,
    observation: str,
    store: MemoryStore | None = None,
) -> PolicyContractReport:
    """Construct and probe a policy against one normalized observation.

    The probe intentionally uses a caller-supplied observation and an isolated
    memory store. This validates the same ``seed -> policy -> action`` contract
    used by benchmark execution without coupling the integration layer to a
    particular model SDK or checkpoint format.
    """

    _validate_seed(seed)
    if not isinstance(observation, str) or not observation.strip():
        raise ValueError("observation must be a non-empty string")
    if not callable(policy_factory):
        raise TypeError("policy_factory must be callable")

    selected_store = store if store is not None else MemoryStore()
    policy = policy_factory(seed, selected_store)
    if not callable(policy):
        raise TypeError("policy_factory must return a callable policy")

    action = policy(observation)
    if not isinstance(action, str) or not action.strip():
        raise ValueError("policy must return a non-empty string action")
    return PolicyContractReport(action=action)


def _validate_minimum_trust(value: object) -> None:
    """Require a finite numeric trust threshold in the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("minimum_trust must be a number between 0 and 1")
    if not isfinite(float(value)):
        raise ValueError("minimum_trust must be finite")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("minimum_trust must be between 0 and 1")


def _validate_seed(seed: object) -> None:
    """Reject values that are not exact integer episode seeds."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


__all__ = [
    "ActionPolicyFactory",
    "MemoryGuidedPolicyFactory",
    "PolicyContractReport",
    "build_memory_guided_policy_factory",
    "validate_policy_contract",
]
