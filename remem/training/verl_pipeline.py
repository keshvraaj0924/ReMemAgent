"""Executable reward pipeline joining verl batch scoring and persisted evidence.

The pipeline is intentionally dependency-light: external trainers own rollout and
optimization, while ReMemAgent owns deterministic reward attribution and evidence
integrity. Persisted evidence is reloaded through the fail-closed verifier before the
result is returned to a trainer integration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from remem.training.verl_adapter import VerlRewardAdapter
from remem.training.verl_batch import VerlBatchRewardResult, compute_verl_batch_rewards
from remem.training.verl_evidence import load_verl_reward_evidence, write_verl_reward_evidence


@dataclass(frozen=True, slots=True)
class VerifiedVerlRewardResult:
    """Trainer rewards whose persisted attribution evidence was verified."""

    rewards: tuple[float, ...]
    evidence_path: Path

    def __post_init__(self) -> None:
        if not self.rewards:
            raise ValueError("at least one verified scalar reward is required")
        if not isinstance(self.evidence_path, Path):
            raise TypeError("evidence_path must be a pathlib.Path")


def run_verified_verl_reward_pipeline(
    samples: Sequence[Mapping[str, Any]],
    *,
    evidence_path: str | Path,
    adapter: VerlRewardAdapter | None = None,
) -> VerifiedVerlRewardResult:
    """Score one ordered batch, persist evidence, and verify it before returning.

    This function does not perform GRPO optimization. It provides the auditable reward
    boundary an external verl trainer can call before consuming scalar rewards.
    """
    destination = Path(evidence_path)
    batch_result = compute_verl_batch_rewards(samples, adapter=adapter)
    write_verl_reward_evidence(destination, batch_result.record)
    verified_record = load_verl_reward_evidence(destination)
    _require_persisted_record_matches_batch(batch_result, verified_record)
    return VerifiedVerlRewardResult(rewards=batch_result.rewards, evidence_path=destination)


def _require_persisted_record_matches_batch(
    batch_result: VerlBatchRewardResult,
    verified_record: object,
) -> None:
    """Reject storage or verification boundaries that alter the computed record."""
    if verified_record != batch_result.record:
        raise ValueError("persisted reward evidence does not match the computed batch record")


__all__ = ["VerifiedVerlRewardResult", "run_verified_verl_reward_pipeline"]
