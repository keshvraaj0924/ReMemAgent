"""Regression tests for the public memory package API."""

from remem.memory import FailureMemoryBuilder, FailureObservation


def test_failure_memory_types_are_public_exports() -> None:
    observation = FailureObservation("state", "action", "failed")
    memory = FailureMemoryBuilder().build("failure-1", observation)

    assert memory.memory_id == "failure-1"
