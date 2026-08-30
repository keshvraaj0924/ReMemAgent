"""Optional integrations for external agent-training frameworks."""

from .grpo import (
    GrpoBatch,
    GrpoSample,
    build_grpo_batch,
    build_grpo_samples,
    compute_group_relative_advantages,
)
from .verl import VerlTrainingBatch, VerlTrajectory, build_verl_training_batch, encode_episode_for_verl

__all__ = [
    "GrpoBatch",
    "GrpoSample",
    "VerlTrainingBatch",
    "VerlTrajectory",
    "build_grpo_batch",
    "build_grpo_samples",
    "compute_group_relative_advantages",
    "build_verl_training_batch",
    "encode_episode_for_verl",
]
