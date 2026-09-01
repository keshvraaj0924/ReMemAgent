"""Regression tests for the public lifecycle API."""

from remem.memory import LifecyclePolicy, MemoryLifecycle


def test_lifecycle_types_are_public_exports() -> None:
    lifecycle = MemoryLifecycle(LifecyclePolicy())

    assert lifecycle.policy.consolidation_threshold == 3
