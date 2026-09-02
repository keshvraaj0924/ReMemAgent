"""Tests for caller-owned model policy composition."""

from __future__ import annotations

import pytest

from remem.integrations.policies import (
    build_memory_guided_policy_factory,
    validate_policy_contract,
)
from remem.memory.store import MemoryStore


def test_factory_passes_episode_seed_to_action_policy_factory() -> None:
    received_seeds: list[int] = []

    def action_policy_factory(seed: int):
        received_seeds.append(seed)
        return lambda state, guidance: f"{state}:{guidance}"

    policy_factory = build_memory_guided_policy_factory(action_policy_factory)
    policy = policy_factory(17, MemoryStore())

    assert policy("observe") == "observe:"
    assert received_seeds == [17]


def test_factory_composes_memory_guidance_without_owning_model_logic() -> None:
    def action_policy_factory(_: int):
        return lambda state, guidance: f"act({state}|{guidance})"

    policy_factory = build_memory_guided_policy_factory(action_policy_factory)
    policy = policy_factory(3, MemoryStore())

    assert policy("open the door") == "act(open the door|)"


def test_factory_rejects_invalid_action_policy_factory() -> None:
    with pytest.raises(TypeError, match="action_policy_factory must be callable"):
        build_memory_guided_policy_factory(None)  # type: ignore[arg-type]


def test_factory_rejects_invalid_action_policy_result() -> None:
    policy_factory = build_memory_guided_policy_factory(lambda _: "not callable")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must return a callable"):
        policy_factory(1, MemoryStore())


def test_factory_rejects_invalid_minimum_trust() -> None:
    with pytest.raises(ValueError, match="minimum_trust"):
        build_memory_guided_policy_factory(
            lambda _: lambda state, guidance: state,
            minimum_trust=1.1,
        )


def test_validate_policy_contract_probes_seed_and_observation() -> None:
    received: list[tuple[int, str]] = []

    def policy_factory(seed: int, store: MemoryStore):
        del store

        def policy(observation: str) -> str:
            received.append((seed, observation))
            return "look"

        return policy

    report = validate_policy_contract(
        policy_factory,
        seed=41,
        observation="room description",
    )

    assert report.action == "look"
    assert received == [(41, "room description")]


def test_validate_policy_contract_rejects_empty_observation() -> None:
    with pytest.raises(ValueError, match="observation"):
        validate_policy_contract(
            lambda seed, store: lambda observation: "look",
            seed=1,
            observation=" ",
        )


def test_validate_policy_contract_rejects_invalid_action() -> None:
    with pytest.raises(ValueError, match="non-empty string action"):
        validate_policy_contract(
            lambda seed, store: lambda observation: "",
            seed=1,
            observation="room",
        )
