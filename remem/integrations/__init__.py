"""Optional integrations for external agent-training frameworks."""

from .artifacts import (
    TRAINING_ARTIFACT_SCHEMA_VERSION,
    TrainingArtifactManifest,
    build_training_artifact_manifest,
    verify_training_artifact,
    write_training_artifact_manifest,
)
from .benchmarks import (
    BenchmarkEnvironmentFactory,
    RawEnvironmentFactory,
    load_benchmark_environment_factory,
    resolve_environment_factory,
)
from .datasets import write_grpo_jsonl, write_verl_jsonl
from .grpo import (
    GrpoBatch,
    GrpoSample,
    build_grpo_batch,
    build_grpo_samples,
    compute_group_relative_advantages,
)
from .huggingface import (
    ActionParser,
    PipelineLoader,
    PromptBuilder,
    build_huggingface_text_action_policy_factory,
)
from .official_benchmarks import (
    build_alfworld_text_environment_factory,
    build_webshop_text_environment_factory,
)
from .policies import (
    ActionPolicyFactory,
    MemoryGuidedPolicyFactory,
    PolicyContractReport,
    build_memory_guided_policy_factory,
    validate_policy_contract,
)
from .verl import (
    VerlTrainingBatch,
    VerlTrajectory,
    build_verl_training_batch,
    encode_episode_for_verl,
    encode_grpo_batch_for_verl,
)
from .verl_adapter import (
    AgentLoopRequest,
    AsyncVerlAgentLoop,
    VerlTrainingConsumer,
    adapt_agent_loop_output,
    dispatch_verl_training_batch,
    run_agent_loop,
    run_agent_loop_batch,
)
from .verl_contract import ValidatedAgentLoopOutput, validate_agent_loop_output

__all__ = [
    "ActionParser",
    "ActionPolicyFactory",
    "AgentLoopRequest",
    "AsyncVerlAgentLoop",
    "BenchmarkEnvironmentFactory",
    "GrpoBatch",
    "GrpoSample",
    "MemoryGuidedPolicyFactory",
    "PipelineLoader",
    "PolicyContractReport",
    "PromptBuilder",
    "RawEnvironmentFactory",
    "TRAINING_ARTIFACT_SCHEMA_VERSION",
    "TrainingArtifactManifest",
    "ValidatedAgentLoopOutput",
    "VerlTrainingBatch",
    "VerlTrainingConsumer",
    "VerlTrajectory",
    "adapt_agent_loop_output",
    "build_alfworld_text_environment_factory",
    "build_grpo_batch",
    "build_grpo_samples",
    "build_huggingface_text_action_policy_factory",
    "build_memory_guided_policy_factory",
    "build_training_artifact_manifest",
    "build_verl_training_batch",
    "build_webshop_text_environment_factory",
    "compute_group_relative_advantages",
    "dispatch_verl_training_batch",
    "encode_episode_for_verl",
    "encode_grpo_batch_for_verl",
    "load_benchmark_environment_factory",
    "resolve_environment_factory",
    "run_agent_loop",
    "run_agent_loop_batch",
    "validate_agent_loop_output",
    "validate_policy_contract",
    "verify_training_artifact",
    "write_grpo_jsonl",
    "write_training_artifact_manifest",
    "write_verl_jsonl",
]
