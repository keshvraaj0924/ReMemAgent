"""Optional integrations for external agent-training frameworks."""

from .grpo import GrpoSample, build_grpo_samples

__all__ = ["GrpoSample", "build_grpo_samples"]
