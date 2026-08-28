"""Tests for state-aligned memory reconstruction."""

from remem.memory.reconstruction import MemoryReconstructor
from remem.memory.types import MemoryKind, MemoryRecord


def build_memory(kind: MemoryKind = MemoryKind.EPISODIC) -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory-1",
        state="A closed cabinet contains an apple.",
        action="Open the cabinet before placing the apple.",
        outcome="Opening the cabinet first allowed the placement to succeed.",
        kind=kind,
    )


def test_reconstructor_accepts_well_aligned_memory() -> None:
    result = MemoryReconstructor().reconstruct(
        build_memory(),
        current_state="A closed cabinet is in front of the agent.",
        context_alignment=0.9,
    )
    assert result.rejected is False
    assert result.context_alignment == 0.9
    assert "Relevant action:" in result.guidance
    assert "Current state:" in result.guidance


def test_reconstructor_rejects_low_alignment() -> None:
    result = MemoryReconstructor(rejection_threshold=0.4).reconstruct(
        build_memory(),
        current_state="A sink contains a dirty plate.",
        context_alignment=0.2,
    )
    assert result.rejected is True
    assert result.guidance == ""
    assert "threshold" in result.reason


def test_failure_memory_becomes_avoidance_guidance() -> None:
    result = MemoryReconstructor().reconstruct(
        build_memory(MemoryKind.FAILURE),
        current_state="The cabinet is closed.",
        context_alignment=0.8,
    )
    assert result.rejected is False
    assert result.guidance.startswith("Avoid:")


def test_reconstructor_clamps_alignment_score() -> None:
    result = MemoryReconstructor().reconstruct(
        build_memory(),
        current_state="The cabinet is closed.",
        context_alignment=1.7,
    )
    assert result.context_alignment == 1.0


def test_reconstructor_requires_current_state() -> None:
    try:
        MemoryReconstructor().reconstruct(build_memory(), "", 0.8)
    except ValueError as error:
        assert str(error) == "current_state must not be empty"
    else:
        raise AssertionError("Expected ValueError for an empty current state")
