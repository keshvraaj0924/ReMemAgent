"""Optional integrations for external agent-training frameworks."""

from .grpo import GrpoSample, build_grpo_samples, compute_group_relative_advantages

__all__ = ["GrpoSample", "build_grpo_samples", "compute_group_relative_advantages"]
