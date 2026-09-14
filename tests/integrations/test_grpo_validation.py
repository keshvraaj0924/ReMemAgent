from fractions import Fraction
from math import inf, nan

import pytest

from remem.integrations.grpo import GrpoBatch, GrpoSample


def make_sample(**overrides: object) -> GrpoSample:
    values: dict[str, object] = {
        "prompt": "task",
        "completion": "act",
        "reward": 1.0,
        "group_id": "group-1",
        "memory_ids": (),
    }
    values.update(overrides)
    return GrpoSample(**values)  # type: ignore[arg-type]


def test_grpo_sample_rejects_non_finite_reward() -> None:
    with pytest.raises(ValueError, match="reward must be finite"):
        make_sample(reward=nan)
    with pytest.raises(ValueError, match="reward must be finite"):
        make_sample(reward=inf)


def test_grpo_sample_rejects_non_real_reward() -> None:
    with pytest.raises(TypeError, match="reward must be a real number"):
        make_sample(reward=True)
    with pytest.raises(TypeError, match="reward must be a real number"):
        make_sample(reward="1.0")


def test_grpo_sample_normalizes_real_reward_to_float() -> None:
    sample = make_sample(reward=Fraction(3, 2))

    assert sample.reward == 1.5
    assert type(sample.reward) is float


def test_grpo_sample_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        make_sample(prompt="   ")
    with pytest.raises(ValueError, match="completion must be a non-empty string"):
        make_sample(completion="")
    with pytest.raises(ValueError, match="group_id must be a non-empty string"):
        make_sample(group_id=" ")


def test_grpo_sample_rejects_empty_memory_identifier() -> None:
    with pytest.raises(ValueError, match="memory_ids must contain only non-empty strings"):
        make_sample(memory_ids=("memory-1", " "))


def test_grpo_batch_rejects_non_finite_advantage() -> None:
    samples = (make_sample(), make_sample())
    with pytest.raises(ValueError, match="advantage must be finite"):
        GrpoBatch(samples=samples, advantages=(0.5, nan))


def test_grpo_batch_rejects_non_real_advantage() -> None:
    samples = (make_sample(), make_sample())
    with pytest.raises(TypeError, match="advantage must be a real number"):
        GrpoBatch(samples=samples, advantages=(0.5, True))
    with pytest.raises(TypeError, match="advantage must be a real number"):
        GrpoBatch(samples=samples, advantages=(0.5, "1.0"))  # type: ignore[arg-type]


def test_grpo_batch_normalizes_real_advantages_to_float() -> None:
    samples = (make_sample(), make_sample())
    batch = GrpoBatch(
        samples=samples,
        advantages=(Fraction(-1, 2), Fraction(1, 2)),  # type: ignore[arg-type]
    )

    assert batch.advantages == (-0.5, 0.5)
    assert all(type(advantage) is float for advantage in batch.advantages)


def test_grpo_batch_accepts_finite_aligned_advantages() -> None:
    samples = (make_sample(), make_sample())
    batch = GrpoBatch(samples=samples, advantages=(-1.0, 1.0))

    assert batch.to_dicts()[0]["advantage"] == -1.0
    assert batch.to_dicts()[1]["advantage"] == 1.0
