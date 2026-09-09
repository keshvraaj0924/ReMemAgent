"""Tests for paired benchmark inferential analysis."""

from __future__ import annotations

import pytest

from experiments.paired_analysis import METRIC_NAMES, analyze_paired_benchmark_reports
from remem.benchmark import BenchmarkRunConfiguration, BenchmarkSuiteRunner
from remem.environments.base import StepResult
from remem.execution import Policy
from remem.memory.store import MemoryStore


class MetricEnvironment:
    """Minimal environment with deterministic success and reward behavior."""

    def __init__(self, episode_index: int, successful_episode_count: int, reward: float) -> None:
        self.episode_index = episode_index
        self.successful_episode_count = successful_episode_count
        self.reward = reward

    def reset(self, **kwargs: object) -> str:
        """Return a deterministic observation."""

        return f"state-{self.episode_index}"

    def step(self, action: str) -> StepResult:
        """Finish immediately with a deterministic reward."""

        success_reward = 1.0 if self.episode_index < self.successful_episode_count else 0.0
        return StepResult(
            observation="done",
            reward=success_reward + self.reward,
            terminated=True,
            truncated=False,
        )

    def close(self) -> None:
        """Release no-op test resources."""


def _run(
    seed: int,
    *,
    successful_episode_count: int,
    reward_offset: float,
    policy_name: str,
) :
    """Run one deterministic condition with explicit provenance metadata."""

    configuration = BenchmarkRunConfiguration(
        benchmark_name="synthetic",
        episode_count=10,
        max_steps=5,
        seed=seed,
        environment_factory="tests.synthetic.environment",
        policy_factory=policy_name,
        success_evaluator="tests.synthetic.success",
        transfer_success_evaluator="tests.synthetic.transfer",
    )

    def environment_factory(index: int) -> MetricEnvironment:
        return MetricEnvironment(index, successful_episode_count, reward_offset)

    def policy_factory(index: int, store: MemoryStore) -> Policy:
        return lambda state: "noop"

    return BenchmarkSuiteRunner().run(
        benchmark_name="synthetic",
        episode_count=10,
        max_steps=5,
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        success_evaluator=lambda episode: episode.total_reward >= 1.0,
        seed=seed,
        configuration=configuration,
    )


def test_analyze_paired_reports_computes_effects_and_adjusted_p_values() -> None:
    """All primary metrics receive inferential and effect-size statistics."""

    baseline = [
        _run(1, successful_episode_count=4, reward_offset=0.0, policy_name="baseline"),
        _run(2, successful_episode_count=5, reward_offset=0.0, policy_name="baseline"),
        _run(3, successful_episode_count=6, reward_offset=0.0, policy_name="baseline"),
    ]
    treatment = [
        _run(1, successful_episode_count=5, reward_offset=0.5, policy_name="treatment"),
        _run(2, successful_episode_count=6, reward_offset=0.5, policy_name="treatment"),
        _run(3, successful_episode_count=7, reward_offset=0.5, policy_name="treatment"),
    ]

    analysis = analyze_paired_benchmark_reports(baseline, treatment)

    assert analysis.seeds == (1, 2, 3)
    assert set(analysis.metrics) == set(METRIC_NAMES)
    assert analysis.metrics["success_rate"].observed_mean_delta == pytest.approx(0.1)
    assert analysis.metrics["success_rate"].effect_size_dz is None
    assert analysis.metrics["success_rate"].p_value == pytest.approx(0.25)
    assert analysis.metrics["success_rate"].adjusted_p_value == pytest.approx(0.75)
    assert analysis.metrics["mean_reward"].sample_size == 3


def test_analyze_paired_reports_preserves_zero_variance_effect_size_as_none() -> None:
    """A constant paired effect has no finite Cohen's d_z denominator."""

    baseline = [_run(seed, successful_episode_count=5, reward_offset=0.0, policy_name="baseline") for seed in (1, 2, 3)]
    treatment = [_run(seed, successful_episode_count=6, reward_offset=0.0, policy_name="treatment") for seed in (1, 2, 3)]

    analysis = analyze_paired_benchmark_reports(baseline, treatment)

    assert analysis.metrics["success_rate"].effect_size_dz is None
    assert analysis.metrics["mean_reward"].effect_size_dz is None
    assert analysis.metrics["transfer_success_rate"].effect_size_dz is None


def test_analyze_paired_reports_rejects_unpaired_seed_sets() -> None:
    """Inferential analysis must retain the strict paired-seed contract."""

    baseline = [_run(1, successful_episode_count=5, reward_offset=0.0, policy_name="baseline")]
    treatment = [_run(2, successful_episode_count=6, reward_offset=0.0, policy_name="treatment")]

    with pytest.raises(ValueError, match="same seed set"):
        analyze_paired_benchmark_reports(baseline, treatment)
