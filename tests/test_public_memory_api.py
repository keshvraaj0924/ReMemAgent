"""Tests for the stable public memory package exports."""


def test_memory_package_exports_core_types_and_components() -> None:
    from remem.memory import (
        MemoryDeduplicator,
        MemoryKind,
        MemoryRecord,
        MemoryStore,
    )

    assert MemoryDeduplicator is not None
    assert MemoryKind is not None
    assert MemoryRecord is not None
    assert MemoryStore is not None
