"""Optional integrations for external agent-training frameworks."""

from .grpo import (
    GrpoBatch,
    GrpoSample,
    build_grpo_batch,
    build_grpo_samples,
    compute_group_relative_advantages,
)
from .verl import VerlTrajectory, encode_episode_for_verl

__all__ = [
    "GrpoBatch",
    "GrpoSample",
    "VerlTrajectory",
    "build_grpo_batch",
    "build_grpo_samples",
    "compute_group_relative_advantages",
    "encode_episode_for_verl",
]
