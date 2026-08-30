"""Outcome attribution for memory-guided transfer attempts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from remem.execution import EpisodeResult, EpisodeStep

from .policy import MemoryGuidanceDecision
from .store import MemoryStore

TransferSuccessEvaluator = Callable[[EpisodeStep, EpisodeResult], bool]


@dataclass(frozen=True, slots=True)
class MemoryTransferOutcome:
    """Observed outcome of one action decision informed by a memory."""

    memory_id: str
    success: bool


class MemoryTransferRecorder:
    """Attribute observed outcomes to memories that supplied guidance."""

    def record(
        self,
        store: MemoryStore,
        decision: MemoryGuidanceDecision,
        *,
        success: bool,
    ) -> MemoryTransferOutcome | None:
        """Record a transfer attempt when a concrete memory was selected."""

        if decision.memory_id is None:
            return None
        store.record_transfer_outcome(decision.memory_id, success=success)
        return MemoryTransferOutcome(memory_id=decision.memory_id, success=success)

    def record_episode(
        self,
        store: MemoryStore,
        decisions: Sequence[MemoryGuidanceDecision],
        episode: EpisodeResult,
        *,
        success_evaluator: TransferSuccessEvaluator | None = None,
    ) -> tuple[MemoryTransferOutcome, ...]:
        """Attribute step-level transfer outcomes for one executed episode.

        Decisions are aligned with executed steps by index. The default
        evaluator credits only a memory-selected terminal action when the
        episode ended successfully; benchmark-specific evaluators can provide
        finer-grained causal criteria.
        """

        if len(decisions) != len(episode.steps):
            raise ValueError("decision history must contain one entry per episode step")
        evaluator = success_evaluator or _default_transfer_success
        outcomes: list[MemoryTransferOutcome] = []
        for decision, step in zip(decisions, episode.steps):
            outcome = self.record(
                store,
                decision,
                success=evaluator(step, episode),
            )
            if outcome is not None:
                outcomes.append(outcome)
        return tuple(outcomes)


def _default_transfer_success(step: EpisodeStep, episode: EpisodeResult) -> bool:
    """Credit only a successful terminal action under the default policy."""

    return step.result.done and episode.terminated and episode.total_reward > 0.0
