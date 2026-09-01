"""Tests for deterministic offline training dataset writers."""

import json

from remem.environments.base import StepResult
from remem.execution import EpisodeResult, EpisodeStep
from remem.integrations.datasets import write_grpo_jsonl, write_verl_jsonl
from remem.integrations.grpo import build_grpo_batch, build_grpo_samples
from remem.integrations.verl import encode_grpo_batch_for_verl


def _episode(reward: float) -> EpisodeResult:
    steps = (
        EpisodeStep(0, "start", "look", StepResult("middle", 0.0, False, False, {})),
        EpisodeStep(1, "middle", "finish", StepResult("goal", reward, True, False, {})),
    )
    return EpisodeResult("start", steps, reward, True, False)


def _encode(value: str) -> tuple[int, ...]:
    return tuple(ord(character) for character in value)


def test_write_grpo_jsonl_preserves_order_and_serializes_advantages(tmp_path) -> None:
    samples = build_grpo_samples(
        [_episode(1.0), _episode(3.0)],
        group_id_builder=lambda _index, _: "task-1",
    )
    batch = build_grpo_batch(samples)
    output_path = tmp_path / "nested" / "grpo.jsonl"

    write_grpo_jsonl(batch, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["reward"] for row in rows] == [1.0, 3.0]
    assert [row["advantage"] for row in rows] == [-1.0, 1.0]


def test_write_verl_jsonl_preserves_token_and_metadata_fields(tmp_path) -> None:
    samples = build_grpo_samples([_episode(2.0)])
    batch = build_grpo_batch(samples)
    verl_batch = encode_grpo_batch_for_verl(
        batch,
        encode_prompt=_encode,
        encode_completion=_encode,
    )
    output_path = tmp_path / "verl.jsonl"

    write_verl_jsonl(verl_batch, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["prompt_ids"] == list(_encode("start"))
    assert row["response_ids"] == list(_encode("look\nfinish"))
    assert row["response_mask"] == [1] * len(row["response_ids"])
    assert row["advantage"] == 0.0
    assert row["metadata"]["group_id"] == "episode-0"


def test_writers_produce_identical_bytes_for_identical_batches(tmp_path) -> None:
    batch = build_grpo_batch(build_grpo_samples([_episode(2.0)]))
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_grpo_jsonl(batch, first)
    write_grpo_jsonl(batch, second)

    assert first.read_bytes() == second.read_bytes()
