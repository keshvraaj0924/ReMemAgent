"""Fail-closed serialization and verification for verl reward evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from remem.training.grpo_metrics import GrpoBatchMetrics
from remem.training.verl_records import VerlBatchRewardRecord, VerlRewardRecord

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "rewards", "metrics"})
_REWARD_FIELDS = frozenset(
    {"sample_index", "task_component", "transfer_component", "memory_cost_component", "total_reward"}
)
_METRIC_FIELDS = frozenset(
    {
        "sample_count",
        "mean_total_reward",
        "mean_task_component",
        "mean_transfer_component",
        "mean_memory_cost_component",
        "positive_transfer_rate",
        "negative_transfer_rate",
    }
)
SCHEMA_VERSION = 1


def serialize_verl_reward_evidence(record: VerlBatchRewardRecord) -> dict[str, Any]:
    """Serialize a validated reward record using the versioned evidence schema."""
    if not isinstance(record, VerlBatchRewardRecord):
        raise TypeError("record must be a VerlBatchRewardRecord")
    return {
        "schema_version": SCHEMA_VERSION,
        "rewards": [reward.to_dict() for reward in record.rewards],
        "metrics": asdict(record.metrics),
    }


def verify_verl_reward_evidence(payload: dict[str, Any]) -> VerlBatchRewardRecord:
    """Reconstruct persisted reward evidence and reject schema or metric drift."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    if frozenset(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("reward evidence must contain exactly the supported top-level fields")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported reward evidence schema_version: {payload['schema_version']!r}")

    rewards_payload = payload["rewards"]
    metrics_payload = payload["metrics"]
    if not isinstance(rewards_payload, list) or not rewards_payload:
        raise ValueError("rewards must be a non-empty list")
    if not isinstance(metrics_payload, dict) or frozenset(metrics_payload) != _METRIC_FIELDS:
        raise ValueError("metrics must contain exactly the supported fields")

    rewards: list[VerlRewardRecord] = []
    for item in rewards_payload:
        if not isinstance(item, dict) or frozenset(item) != _REWARD_FIELDS:
            raise ValueError("each reward must contain exactly the supported fields")
        rewards.append(VerlRewardRecord(**item))

    metrics = GrpoBatchMetrics(**metrics_payload)
    return VerlBatchRewardRecord(rewards=tuple(rewards), metrics=metrics)


def write_verl_reward_evidence(path: str | Path, record: VerlBatchRewardRecord) -> None:
    """Atomically persist validated reward evidence as deterministic JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    payload = serialize_verl_reward_evidence(record)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_verl_reward_evidence(path: str | Path) -> VerlBatchRewardRecord:
    """Load and verify persisted reward evidence before downstream consumption."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return verify_verl_reward_evidence(payload)


__all__ = [
    "SCHEMA_VERSION",
    "load_verl_reward_evidence",
    "serialize_verl_reward_evidence",
    "verify_verl_reward_evidence",
    "write_verl_reward_evidence",
]
