"""Paired comparison of policies on identical environment seeds."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from remem.environment_evaluation import EnvironmentEvaluation


@dataclass(frozen=True, slots=True)
class PairedSeedDelta:
    """Per-seed reward and step deltas between candidate and baseline policies."""

    seed: int
    reward_delta: float
    step_delta: int


@dataclass(frozen=True, slots=True)
class EnvironmentPolicyComparison:
    """Immutable paired comparison for two evaluations over identical seeds."""

    baseline: EnvironmentEvaluation
    candidate: EnvironmentEvaluation
    paired_deltas: tuple[PairedSeedDelta, ...]

    @property
    def mean_reward_delta(self) -> float:
        """Return candidate-minus-baseline mean reward delta."""

        return sum(delta.reward_delta for delta in self.paired_deltas) / len(
            self.paired_deltas
        )

    @property
    def mean_step_delta(self) -> float:
        """Return candidate-minus-baseline mean transition-count delta."""

        return sum(delta.step_delta for delta in self.paired_deltas) / len(
            self.paired_deltas
        )

    @property
    def improvement_rate(self) -> float:
        """Return fraction of seeds where candidate reward exceeds baseline reward."""

        improved = sum(delta.reward_delta > 0.0 for delta in self.paired_deltas)
        return improved / len(self.paired_deltas)

    @property
    def regression_rate(self) -> float:
        """Return fraction of seeds where candidate reward is below baseline reward."""

        regressed = sum(delta.reward_delta < 0.0 for delta in self.paired_deltas)
        return regressed / len(self.paired_deltas)


def compare_environment_evaluations(
    baseline: EnvironmentEvaluation,
    candidate: EnvironmentEvaluation,
) -> EnvironmentPolicyComparison:
    """Compare evaluations seed-by-seed without hiding task-order mismatches.

    Paired evaluation is only valid when both policies were evaluated on the exact
    same ordered seed suite. Rejecting reordered or incomplete suites prevents an
    aggregate comparison from accidentally pairing different benchmark tasks.
    """

    if not isinstance(baseline, EnvironmentEvaluation):
        raise TypeError("baseline must be an EnvironmentEvaluation")
    if not isinstance(candidate, EnvironmentEvaluation):
        raise TypeError("candidate must be an EnvironmentEvaluation")

    baseline_seeds = tuple(episode.seed for episode in baseline.episodes)
    candidate_seeds = tuple(episode.seed for episode in candidate.episodes)
    if baseline_seeds != candidate_seeds:
        raise ValueError("baseline and candidate must use identical ordered seeds")

    paired_deltas = tuple(
        PairedSeedDelta(
            seed=baseline_episode.seed,
            reward_delta=(
                candidate_episode.result.total_reward
                - baseline_episode.result.total_reward
            ),
            step_delta=(
                candidate_episode.result.step_count - baseline_episode.result.step_count
            ),
        )
        for baseline_episode, candidate_episode in zip(
            baseline.episodes, candidate.episodes, strict=True
        )
    )
    comparison = EnvironmentPolicyComparison(
        baseline=baseline,
        candidate=candidate,
        paired_deltas=paired_deltas,
    )
    if not all(
        isfinite(value)
        for value in (
            comparison.mean_reward_delta,
            comparison.mean_step_delta,
            comparison.improvement_rate,
            comparison.regression_rate,
        )
    ):
        raise ValueError("comparison aggregates must be finite")
    return comparison


__all__ = [
    "EnvironmentPolicyComparison",
    "PairedSeedDelta",
    "compare_environment_evaluations",
]
