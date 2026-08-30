"""Optional integrations for external agent-training frameworks."""

from .grpo import GrpoSample, build_grpo_samples, compute_group_relative_advantages
from .verl import VerlTrajectory, encode_episode_for_verl

__all__ = [
    "GrpoSample",
    "VerlTrajectory",
    "build_grpo_samples",
    "compute_group_relative_advantages",
    "encode_episode_for_verl",
]
