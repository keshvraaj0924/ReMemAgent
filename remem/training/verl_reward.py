"""verl-compatible custom reward entrypoints for ReMemAgent GRPO training.

verl loads custom reward functions by file path. ReMemAgent keeps that framework-specific
surface here and delegates reward semantics to the typed, dependency-light
:class:`VerlRewardAdapter`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from remem.training.grpo import GrpoRewardConfig
from remem.training.verl_adapter import VerlRewardAdapter, VerlRewardFields

_DEFAULT_ADAPTER = VerlRewardAdapter()
_ALLOWED_REWARD_KWARGS = frozenset(
    {
        "task_weight",
        "positive_transfer_weight",
        "negative_transfer_weight",
        "memory_use_cost",
        "task_reward_field",
        "memory_used_field",
        "counterfactual_delta_field",
    }
)


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any] | None = None,
    **reward_kwargs: Any,
) -> float:
    """Compute a memory-aware reward using verl's custom reward function contract.

    ``data_source`` is accepted because it is part of verl's public reward callback
    shape. Reward attribution is intentionally based only on measured trajectory
    metadata in ``extra_info``; generated text and ground truth are not used as a
    proxy for memory transfer.

    Optional keyword arguments allow trainer configuration to override reward weights
    and metadata field names without introducing a verl dependency into the core
    reward implementation.

    Raises:
        ValueError: If ``extra_info`` is missing or lacks required measurements.
        TypeError: If trajectory measurements or reward options have invalid types.
    """
    del data_source
    if extra_info is None:
        raise ValueError("extra_info is required for ReMemAgent reward attribution")

    adapter = _build_adapter(reward_kwargs) if reward_kwargs else _DEFAULT_ADAPTER
    return adapter.from_extra_info(solution_str, ground_truth, extra_info)


def compute_score_batched(
    data_sources: Sequence[Any],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[Mapping[str, Any]],
    **reward_kwargs: Any,
) -> list[float]:
    """Compute rewards for verl's batched custom-reward contract.

    The batch boundary validates aligned cardinality before scoring so malformed trainer
    batches fail closed instead of being silently truncated by ``zip``. Each item is
    delegated to :func:`compute_score`, keeping scalar and batched reward semantics
    identical and preserving configurable reward fields and weights.
    """
    batch_size = len(data_sources)
    lengths = {
        "solution_strs": len(solution_strs),
        "ground_truths": len(ground_truths),
        "extra_infos": len(extra_infos),
    }
    mismatched = {name: size for name, size in lengths.items() if size != batch_size}
    if mismatched:
        details = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(f"verl reward batch lengths must match data_sources={batch_size}: {details}")

    return [
        compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **reward_kwargs,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        )
    ]


def _build_adapter(reward_kwargs: Mapping[str, Any]) -> VerlRewardAdapter:
    """Build an adapter from explicitly supported trainer reward options."""
    unknown_options = set(reward_kwargs) - _ALLOWED_REWARD_KWARGS
    if unknown_options:
        unknown = ", ".join(sorted(unknown_options))
        raise ValueError(f"unsupported ReMemAgent reward option(s): {unknown}")

    config = GrpoRewardConfig(
        task_reward_weight=_real_option(reward_kwargs, "task_weight", 1.0),
        positive_transfer_weight=_real_option(reward_kwargs, "positive_transfer_weight", 0.5),
        negative_transfer_weight=_real_option(reward_kwargs, "negative_transfer_weight", 1.0),
        memory_use_cost=_real_option(reward_kwargs, "memory_use_cost", 0.01),
    )
    fields = VerlRewardFields(
        task_reward=_string_option(reward_kwargs, "task_reward_field", "task_reward"),
        memory_used=_string_option(reward_kwargs, "memory_used_field", "memory_used"),
        counterfactual_delta=_string_option(
            reward_kwargs, "counterfactual_delta_field", "counterfactual_delta"
        ),
    )
    return VerlRewardAdapter(config=config, fields=fields)


def _real_option(options: Mapping[str, Any], name: str, default: float) -> float:
    """Read a numeric reward option without accepting booleans."""
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    return float(value)


def _string_option(options: Mapping[str, Any], name: str, default: str) -> str:
    """Read a trainer metadata field name."""
    value = options.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


__all__ = ["compute_score", "compute_score_batched"]
