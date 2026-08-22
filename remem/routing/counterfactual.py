"""Counterfactual routing between memory-guided and self-reasoning paths."""

from __future__ import annotations

from collections.abc import Callable

from remem.memory.types import CounterfactualScore, MemoryDecision


class CounterfactualRouter:
    """Select memory use only when its estimated benefit is positive.

    ``evaluate_with_memory`` and ``evaluate_without_memory`` are deliberately
    injected callables. This keeps the research layer independent of a
    particular LLM, environment, or reward implementation.
    """

    def __init__(self, minimum_delta: float = 0.0) -> None:
        self.minimum_delta = float(minimum_delta)

    def route(
        self,
        *,
        evaluate_with_memory: Callable[[], float],
        evaluate_without_memory: Callable[[], float],
    ) -> tuple[CounterfactualScore, MemoryDecision]:
        with_memory = float(evaluate_with_memory())
        without_memory = float(evaluate_without_memory())
        score = CounterfactualScore(with_memory, without_memory)

        if score.delta > self.minimum_delta:
            decision = MemoryDecision(
                route="memory",
                confidence=_confidence(score.delta),
                expected_delta=score.delta,
                reason="memory path has higher estimated utility",
            )
        else:
            decision = MemoryDecision(
                route="self_reasoning",
                confidence=_confidence(-score.delta),
                expected_delta=score.delta,
                reason="memory does not clear the expected-benefit threshold",
            )
        return score, decision


def _confidence(delta: float) -> float:
    """Convert a utility gap into a bounded routing confidence."""

    return max(0.0, min(1.0, abs(float(delta))))
