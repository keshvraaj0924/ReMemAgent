"""Training integration contracts for learned ReMemAgent policies."""

from remem.training.grpo import (
    GrpoRewardBreakdown,
    GrpoRewardConfig,
    GrpoTrajectory,
    compute_grpo_reward,
    compute_grpo_reward_breakdown,
)
from remem.training.grpo_metrics import GrpoBatchMetrics, summarize_grpo_breakdowns
from remem.training.verl_adapter import VerlRewardAdapter, VerlRewardFields
from remem.training.verl_records import (
    VerlBatchRewardRecord,
    VerlRewardRecord,
    build_verl_batch_reward_record,
)
from remem.training.verl_reward import compute_score, compute_score_batched

__all__ = [
    "GrpoBatchMetrics",
    "GrpoRewardBreakdown",
    "GrpoRewardConfig",
    "GrpoTrajectory",
    "VerlBatchRewardRecord",
    "VerlRewardAdapter",
    "VerlRewardFields",
    "VerlRewardRecord",
    "build_verl_batch_reward_record",
    "compute_grpo_reward",
    "compute_grpo_reward_breakdown",
    "compute_score",
    "compute_score_batched",
    "summarize_grpo_breakdowns",
]
