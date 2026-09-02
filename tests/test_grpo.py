from __future__ import annotations

import pytest

from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep
from remem.integrations.grpo import GrpoSample, build_grpo_batch


def _build_sample(reward: float, group_id: str) -> GrpoSample:
    return GrpoSample(
        prompt="task",
        completion="finish",
        reward=reward,
        group_id=group_id,
        memory_ids=(),
    )


def test_build_grpo_batch_computes_comparative_advantages() -> None:
    batch = build_grpo_batch(
        (
            _build_sample(0.0, "task-1"),
            _build_sample(2.0, "task-1"),
            _build_sample(1.0, "task-2"),
            _build_sample(3.0, "task-2"),
        )
    )

    assert batch.advantages == pytest.approx((-1.0, 1.0, -1.0, 1.0))


def test_build_grpo_batch_rejects_singleton_groups() -> None:
    with pytest.raises(ValueError, match="singleton groups: task-1"):
        build_grpo_batch(
            (
                _build_sample(1.0, "task-1"),
                _build_sample(2.0, "task-2"),
            )
        )


def test_grpo_batch_validation_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="zero samples"):
        build_grpo_batch(())


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("prompt", None),
        ("completion", None),
        ("group_id", None),
    ),
)
def test_grpo_sample_rejects_non_string_text_fields(field_name: str, value: object) -> None:
    fields: dict[str, object] = {
        "prompt": "task",
        "completion": "finish",
        "reward": 1.0,
        "group_id": "group",
        "memory_ids": (),
    }
    fields[field_name] = value

    with pytest.raises(ValueError, match="non-empty string"):
        GrpoSample(**fields)  # type: ignore[arg-type]


def test_grpo_sample_rejects_non_tuple_memory_ids() -> None:
    with pytest.raises(TypeError, match="tuple of strings"):
        GrpoSample(
            prompt="task",
            completion="finish",
            reward=1.0,
            group_id="group",
            memory_ids=["memory-1"],  # type: ignore[arg-type]
        )


def test_episode_fixture_remains_compatible_with_sample_contract() -> None:
    episode = EpisodeResult(
        initial_observation="task",
        steps=(
            EpisodeStep(
                step_index=0,
                observation="task",
                action="finish",
                result=StepResult(
                    observation="done",
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                ),
            ),
        ),
        total_reward=1.0,
        terminated=True,
        truncated=False,
    )

    sample = GrpoSample(
        prompt=episode.initial_observation,
        completion=episode.steps[0].action,
        reward=episode.total_reward,
        group_id="task",
        memory_ids=(),
    )

    assert sample.to_dict()["reward"] == 1.0
