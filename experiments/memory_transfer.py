"""Per-memory analysis of synthetic transfer outcomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from experiments.synthetic_negative_transfer import BenchmarkCaseResult, BenchmarkResult
    from remem.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class MemoryTransferSummary:
    """Aggregate benchmark outcomes for one memory identity."""

    memory_id: str
    transfer_attempts: int
    negative_transfer_cases: int
    routing_regret: float

    @property
    def negative_transfer_rate(self) -> float:
        """Return the fraction of this memory's selected cases that harmed utility."""

        return (
            self.negative_transfer_cases / self.transfer_attempts
            if self.transfer_attempts
            else 0.0
        )


def summarize_memory_transfers(
    case_results: Iterable["BenchmarkCaseResult"],
) -> tuple[MemoryTransferSummary, ...]:
    """Group selected benchmark outcomes by memory identity."""

    attempts: defaultdict[str, int] = defaultdict(int)
    negative_cases: defaultdict[str, int] = defaultdict(int)
    regret: defaultdict[str, float] = defaultdict(float)

    for result in case_results:
        if result.selected_route != "memory" or result.memory_id is None:
            continue
        attempts[result.memory_id] += 1
        if result.negative_transfer:
            negative_cases[result.memory_id] += 1
        regret[result.memory_id] += result.regret

    return tuple(
        MemoryTransferSummary(
            memory_id=memory_id,
            transfer_attempts=attempts[memory_id],
            negative_transfer_cases=negative_cases[memory_id],
            routing_regret=regret[memory_id],
        )
        for memory_id in sorted(attempts)
    )


def record_transfer_outcomes(
    benchmark_result: "BenchmarkResult",
    memory_store: "MemoryStore",
) -> int:
    """Record only explicit, selected transfer outcomes in the memory store."""

    recorded_count = 0
    for case_result in benchmark_result.case_results:
        if (
            case_result.selected_route != "memory"
            or case_result.memory_id is None
            or case_result.transfer_success is None
        ):
            continue
        memory_store.record_transfer_outcome(
            case_result.memory_id,
            success=case_result.transfer_success,
        )
        recorded_count += 1
    return recorded_count
