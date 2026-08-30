"""Optional integrations for external agent-training frameworks."""

from .grpo import (
    GrpoBatch,
    GrpoSample,
    build_grpo_batch,
    build_grpo_samples,
    compute_group_relative_advantages,
)
from .verl import (
    VerlTrainingBatch,
    VerlTrajectory,
    build_verl_training_batch,
    encode_episode_for_verl,
)
from .verl_adapter import VerlTrainingConsumer, dispatch_verl_training_batch
from .verl_contract import ValidatedAgentLoopOutput, validate_agent_loop_output

__all__ = [
    "GrpoBatch",
    "GrpoSample",
    "ValidatedAgentLoopOutput",
    "VerlTrainingBatch",
    "VerlTrainingConsumer",
    "VerlTrajectory",
    "build_grpo_batch",
    "build_grpo_samples",
    "compute_group_relative_advantages",
    "build_verl_training_batch",
    "encode_episode_for_verl",
    "dispatch_verl_training_batch",
    "validate_agent_loop_output",
]
