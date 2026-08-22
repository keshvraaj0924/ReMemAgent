"""Typed domain objects used by the ReMemAgent memory engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryKind(str, Enum):
    """Stage or semantic role of an experience in the memory lifecycle."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """An experience stored for possible future transfer."""

    memory_id: str
    state: str
    action: str
    outcome: str
    kind: MemoryKind = MemoryKind.EPISODIC
    reward: float = 0.0
    uses: int = 0
    successes: int = 0
    failures: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def empirical_success_rate(self) -> float:
        attempts = self.successes + self.failures
        return self.successes / attempts if attempts else 0.0


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """A memory candidate returned for the current observation."""

    memory: MemoryRecord
    similarity: float
    reconstructed_guidance: str = ""


@dataclass(frozen=True, slots=True)
class CounterfactualScore:
    """Estimated value difference between using and ignoring memory."""

    with_memory: float
    without_memory: float

    @property
    def delta(self) -> float:
        return self.with_memory - self.without_memory


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """Final routing decision for a memory candidate or candidate set."""

    route: str
    confidence: float
    expected_delta: float
    reason: str
