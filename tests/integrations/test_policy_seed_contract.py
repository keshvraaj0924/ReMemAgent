from __future__ import annotations

import pytest

from remem.integrations.policies import (
    build_memory_guided_policy_factory,
    validate_policy_contract,
)
from remem.memory.store import MemoryStore


def _action_policy_factory(seed: int):
    return lambda observation: f"look {seed}"


def _policy_factory(seed: int, store: MemoryStore):
    return lambda observation: f"look {seed}"


@pytest.mark.parametrize("seed", [True, False, 1.5, "7"])
def test_build_memory_guided_policy_factory_rejects_invalid_seed(seed: object) -> None:
    factory = build_memory_guided_policy_factory(_action_policy_factory)

    with pytest.raises(TypeError, match="seed must be an integer"):
        factory(seed, MemoryStore())  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, False, 1.5, "7"])
def test_validate_policy_contract_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        validate_policy_contract(
            _policy_factory,
            seed=seed,  # type: ignore[arg-type]
            observation="You are in a room.",
        )


def test_policy_seed_is_forwarded_after_validation() -> None:
    observed_seeds: list[int] = []

    def action_policy_factory(seed: int):
        observed_seeds.append(seed)
        return lambda observation: "look"

    factory = build_memory_guided_policy_factory(action_policy_factory)
    factory(17, MemoryStore())

    assert observed_seeds == [17]


def test_policy_contract_preserves_valid_seed() -> None:
    report = validate_policy_contract(
        _policy_factory,
        seed=17,
        observation="You are in a room.",
    )

    assert report.action == "look 17"
