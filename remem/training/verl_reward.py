"""verl-compatible custom reward entrypoint for ReMemAgent GRPO training.

verl loads custom reward functions by file path and calls ``compute_score`` with the
``(data_source, solution_str, ground_truth, extra_info)`` contract. ReMemAgent keeps
that framework-specific surface here and delegates reward semantics to the typed,
dependency-light :class:`VerlRewardAdapter`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remem.training.verl_adapter import VerlRewardAdapter

_DEFAULT_ADAPTER = VerlRewardAdapter()


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any] | None = None,
) -> float:
    """Compute a memory-aware reward using verl's custom reward function contract.

    ``data_source`` is accepted because it is part of verl's public reward callback
    shape. Reward attribution is intentionally based only on measured trajectory
    metadata in ``extra_info``; generated text and ground truth are not used as a
    proxy for memory transfer.

    Raises:
        ValueError: If ``extra_info`` is missing or lacks required measurements.
        TypeError: If trajectory measurements have invalid types.
    """
    del data_source
    if extra_info is None:
        raise ValueError("extra_info is required for ReMemAgent reward attribution")
    return _DEFAULT_ADAPTER.from_extra_info(solution_str, ground_truth, extra_info)


__all__ = ["compute_score"]
