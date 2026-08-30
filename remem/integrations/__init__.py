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
from .verl_adapter import (
    VerlTrainingConsumer,
    adapt_agent_loop_output,
    dispatch_verl_training_batch,
)
from .verl_contract import ValidatedAgentLoopOutput, validate_agent_loop_output

__all__ = [
    "GrpoBatch",
    "GrpoSample",
    "ValidatedAgentLoopOutput",
    "VerlTrainingBatch",
    "VerlTrainingConsumer",
    "VerlTrajectory",
    "adapt_agent_loop_output",
    "build_grpo_batch",
    "build_grpo_samples",
    "build_verl_training_batch",
    "compute_group_relative_advantages",
    "dispatch_verl_training_batch",
    "encode_episode_for_verl",
    "validate_agent_loop_output",
]
