from __future__ import annotations

from pathlib import Path

import pytest

from remem.training.verl_adapter import VerlRewardAdapter
from remem.training.verl_evidence import load_verl_reward_evidence
from remem.training.verl_pipeline import VerifiedVerlRewardResult, run_verified_verl_reward_pipeline


def _samples() -> list[dict[str, object]]:
    return [
        {
            "task_reward": 1.0,
            "memory_used": True,
            "counterfactual_delta": 0.2,
        },
        {
            "task_reward": 0.0,
            "memory_used": True,
            "counterfactual_delta": -0.3,
        },
    ]


def test_verified_pipeline_persists_exact_reward_record(tmp_path: Path) -> None:
    evidence_path = tmp_path / "training" / "reward-evidence.json"

    result = run_verified_verl_reward_pipeline(_samples(), evidence_path=evidence_path)

    persisted = load_verl_reward_evidence(evidence_path)
    assert result == VerifiedVerlRewardResult(
        rewards=tuple(item.total_reward for item in persisted.rewards),
        evidence_path=evidence_path,
    )
    assert tuple(item.sample_index for item in persisted.rewards) == (0, 1)


def test_verified_pipeline_forwards_custom_adapter(tmp_path: Path) -> None:
    adapter = VerlRewardAdapter()

    result = run_verified_verl_reward_pipeline(
        _samples(),
        evidence_path=tmp_path / "reward-evidence.json",
        adapter=adapter,
    )

    assert len(result.rewards) == 2


def test_verified_result_rejects_empty_rewards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one verified"):
        VerifiedVerlRewardResult(rewards=(), evidence_path=tmp_path / "evidence.json")


def test_verified_result_requires_path() -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        VerifiedVerlRewardResult(rewards=(1.0,), evidence_path="evidence.json")  # type: ignore[arg-type]


def test_pipeline_rejects_empty_batch_before_persisting(tmp_path: Path) -> None:
    evidence_path = tmp_path / "reward-evidence.json"

    with pytest.raises(ValueError, match="trainer sample"):
        run_verified_verl_reward_pipeline([], evidence_path=evidence_path)

    assert not evidence_path.exists()
