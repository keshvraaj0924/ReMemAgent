"""Typed domain objects used by the ReMemAgent memory engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryKind(str, Enum):
    """Semantic role a memory plays in the agent's memory hierarchy."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    FAILURE = "failure"


class MemoryStatus(str, Enum):
    """Lifecycle state of a stored memory."""

    ACTIVE = "active"
    STALE = "stale"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """A memory candidate paired with its retrieval relevance score."""

    memory: "MemoryRecord"
    similarity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("similarity must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CounterfactualScore:
    """Utility estimates for memory-guided and self-reasoning paths."""

    with_memory: float
    without_memory: float

    @property
    def delta(self) -> float:
        """Return the estimated utility benefit of using memory."""

        return self.with_memory - self.without_memory


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """Routing decision produced by the counterfactual policy."""

    route: str
    confidence: float
    expected_delta: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(slots=True)
class MemoryRecord:
    """An experience stored for possible future transfer."""

    memory_id: str
    state: str
    action: str = ""
    outcome: str = ""
    kind: MemoryKind = MemoryKind.EPISODIC
    reward: float = 0.0
    uses: int = 0
    successes: int = 0
    failures: int = 0
    transfer_attempts: int = 0
    transfer_successes: int = 0
    confidence: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def empirical_success_rate(self) -> float:
        """Return the observed success rate, using a neutral prior when unused."""

        attempts = self.successes + self.failures
        return self.successes / attempts if attempts else 0.5

    @property
    def empirical_success(self) -> float:
        """Return the empirical success score used by legacy routing clients."""

        return self.empirical_success_rate

    @property
    def transferability(self) -> float:
        """Return observed success when this memory is transferred to new contexts."""

        return self.transfer_successes / self.transfer_attempts if self.transfer_attempts else 0.5

    def record_use(self, success: bool, transferred: bool = False) -> None:
        """Record one memory use and optionally attribute it to transfer."""

        self.uses += 1
        self.last_used_at = datetime.now(timezone.utc)
        if success:
            self.successes += 1
        else:
            self.failures += 1
        if transferred:
            self.transfer_attempts += 1
            if success:
                self.transfer_successes += 1
