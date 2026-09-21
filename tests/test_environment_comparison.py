from __future__ import annotations

import pytest

from remem.environment_comparison import compare_environment_evaluations
from remem.environment_evaluation import EnvironmentEvaluation, SeededEpisodeResult
from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep


def _episode(seed: int, reward: float, step_count: int) -> EpisodeResult:
    steps = tuple(
        EpisodeStep(
            step_index=index,
            observation=f"seed:{seed}:step:{index}",
            action="act",
            result=StepResult(
                observation=f"seed:{seed}:step:{index + 1}",
                reward=reward if index == step_count - 1 else 0.0,
                terminated=index == step_count - 1,
            ),
        )
        for index in range(step_count)
    )
    return EpisodeResult(
        initial_observation=f"seed:{seed}",
        steps=steps,
        total_reward=reward,
        terminated=True,
    )


def _evaluation(*results: tuple[int, float, int]) -> EnvironmentEvaluation:
    return EnvironmentEvaluation(
        episodes=tuple(
            SeededEpisodeResult(seed=seed, result=_episode(seed, reward, step_count))
            for seed, reward, step_count in results
        )
    )


def test_compare_environment_evaluations_preserves_paired_deltas() -> None:
    baseline = _evaluation((3, 1.0, 4), (1, 2.0, 3), (2, 3.0, 2))
    candidate = _evaluation((3, 2.0, 3), (1, 1.0, 5), (2, 3.0, 2))

    comparison = compare_environment_evaluations(baseline, candidate)

    assert tuple(delta.seed for delta in comparison.paired_deltas) == (3, 1, 2)
    assert tuple(delta.reward_delta for delta in comparison.paired_deltas) == (
        1.0,
        -1.0,
        0.0,
    )
    assert tuple(delta.step_delta for delta in comparison.paired_deltas) == (-1, 2, 0)
    assert comparison.mean_reward_delta == pytest.approx(0.0)
    assert comparison.mean_step_delta == pytest.approx(1 / 3)
    assert comparison.improvement_rate == pytest.approx(1 / 3)
    assert comparison.regression_rate == pytest.approx(1 / 3)


def test_compare_environment_evaluations_rejects_reordered_seeds() -> None:
    baseline = _evaluation((1, 1.0, 1), (2, 2.0, 1))
    candidate = _evaluation((2, 2.0, 1), (1, 1.0, 1))

    with pytest.raises(ValueError, match="identical ordered seeds"):
        compare_environment_evaluations(baseline, candidate)


def test_compare_environment_evaluations_rejects_incomplete_seed_suite() -> None:
    baseline = _evaluation((1, 1.0, 1), (2, 2.0, 1))
    candidate = _evaluation((1, 1.0, 1))

    with pytest.raises(ValueError, match="identical ordered seeds"):
        compare_environment_evaluations(baseline, candidate)


def test_compare_environment_evaluations_rejects_wrong_input_type() -> None:
    evaluation = _evaluation((1, 1.0, 1))

    with pytest.raises(TypeError, match="baseline must"):
        compare_environment_evaluations(object(), evaluation)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="candidate must"):
        compare_environment_evaluations(evaluation, object())  # type: ignore[arg-type]
