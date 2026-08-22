"""Deterministic reconstruction primitives used before memory is trusted."""
from __future__ import annotations
from dataclasses import dataclass
from .types import MemoryRecord

@dataclass(frozen=True, slots=True)
class Reconstruction:
    memory_id: str
    guidance: str
    context_alignment: float
    rejected: bool
    reason: str = ""

class MemoryReconstructor:
    """Turn an experience into state-aligned guidance without replaying raw text."""
    def reconstruct(self, memory: MemoryRecord, current_state: str, context_match: float) -> Reconstruction:
        if not current_state.strip():
            raise ValueError("current_state must not be empty")
        score = max(0.0, min(1.0, context_match))
        if score < 0.25:
            return Reconstruction(memory.memory_id, "", score, True, "low context alignment")
        prefix = "Avoid: " if memory.kind.value == "failure" else "Guidance: "
        guidance = f"{prefix}{memory.content.strip()}\nCurrent state: {current_state.strip()}"
        return Reconstruction(memory.memory_id, guidance, score, False)
