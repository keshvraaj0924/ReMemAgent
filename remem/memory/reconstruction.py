"""Deterministic memory reconstruction primitives.

The reconstruction layer deliberately avoids replaying a stored trajectory. It
converts a memory into compact, state-aligned guidance and rejects memories
whose contextual alignment is too weak to justify transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import MemoryKind, MemoryRecord

DEFAULT_REJECTION_THRESHOLD = 0.25


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """Result of adapting one memory to the current agent state."""

    memory_id: str
    guidance: str
    context_alignment: float
    rejected: bool
    reason: str = ""


class MemoryReconstructor:
    """Transform stored experience into state-aligned guidance."""

    def __init__(self, rejection_threshold: float = DEFAULT_REJECTION_THRESHOLD) -> None:
        if not 0.0 <= rejection_threshold <= 1.0:
            raise ValueError("rejection_threshold must be between 0 and 1")
        self.rejection_threshold = rejection_threshold

    def reconstruct(
        self,
        memory: MemoryRecord,
        current_state: str,
        context_alignment: float,
    ) -> Reconstruction:
        """Reconstruct ``memory`` for ``current_state`` or reject it."""
        if not current_state.strip():
            raise ValueError("current_state must not be empty")

        alignment_score = _clamp_unit_interval(context_alignment)
        if alignment_score < self.rejection_threshold:
            return Reconstruction(
                memory_id=memory.memory_id,
                guidance="",
                context_alignment=alignment_score,
                rejected=True,
                reason="context alignment below reconstruction threshold",
            )

        guidance_prefix = "Avoid" if memory.kind is MemoryKind.FAILURE else "Guidance"
        guidance = (
            f"{guidance_prefix}: {memory.outcome.strip()}\n"
            f"Relevant action: {memory.action.strip()}\n"
            f"Current state: {current_state.strip()}"
        )
        return Reconstruction(
            memory_id=memory.memory_id,
            guidance=guidance,
            context_alignment=alignment_score,
            rejected=False,
        )


def _clamp_unit_interval(value: float) -> float:
    """Clamp a numeric score into the inclusive [0, 1] interval."""
    return max(0.0, min(1.0, value))
