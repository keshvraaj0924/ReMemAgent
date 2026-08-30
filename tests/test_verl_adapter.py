"""Tests for the framework-neutral verl batch dispatch boundary."""

from remem.integrations.verl import VerlTrajectory, build_verl_training_batch
from remem.integrations.verl_adapter import dispatch_verl_training_batch


def _trajectory(reward: float) -> VerlTrajectory:
    return VerlTrajectory(
        prompt_ids=(1,),
        response_ids=(2, 3),
        response_mask=(1, 1),
        reward=reward,
        metadata={"memory_ids": []},
    )


def test_dispatch_preserves_row_order_and_advantages() -> None:
    batch = build_verl_training_batch(
        (_trajectory(1.0), _trajectory(3.0)),
        (-1.0, 1.0),
    )
    received: list[dict[str, object]] = []

    result = dispatch_verl_training_batch(
        batch,
        lambda rows: received.extend(rows) or "accepted",
    )

    assert result == "accepted"
    assert [row["reward"] for row in received] == [1.0, 3.0]
    assert [row["advantage"] for row in received] == [-1.0, 1.0]


def test_dispatch_passes_serializable_trajectory_fields() -> None:
    batch = build_verl_training_batch((_trajectory(2.0),), (0.0,))
    captured = dispatch_verl_training_batch(batch, lambda rows: rows)

    assert captured[0]["prompt_ids"] == [1]
    assert captured[0]["response_ids"] == [2, 3]
    assert captured[0]["response_mask"] == [1, 1]
    assert captured[0]["metadata"] == {"memory_ids": []}


def test_dispatch_rejects_missing_consumer() -> None:
    batch = build_verl_training_batch((_trajectory(1.0),), (0.0,))

    try:
        dispatch_verl_training_batch(batch, None)  # type: ignore[arg-type]
    except ValueError as error:
        assert str(error) == "consumer must be provided"
    else:
        raise AssertionError("missing consumer should be rejected")


def test_dispatch_isolates_consumer_mutation_from_source_batch() -> None:
    batch = build_verl_training_batch((_trajectory(2.0),), (0.5,))

    def mutate(rows: tuple[object, ...]) -> None:
        row = rows[0]
        assert isinstance(row, dict)
        row["reward"] = 99.0
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        metadata["memory_ids"] = ["mutated"]

    dispatch_verl_training_batch(batch, mutate)

    assert batch.trajectories[0].reward == 2.0
    assert batch.trajectories[0].metadata["memory_ids"] == []
