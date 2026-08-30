"""Framework-facing dispatch boundary for verl-compatible training batches.

The adapter deliberately depends only on ReMemAgent's normalized batch contract.
A caller injects the external trainer/collator, keeping verl, torch, and
accelerate dependencies outside the research package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from remem.integrations.verl import VerlTrainingBatch

TrainingConsumer = Callable[[tuple[Mapping[str, object], ...]], object]
ConsumerResult = TypeVar("ConsumerResult")


def dispatch_verl_training_batch(
    batch: VerlTrainingBatch,
    consumer: Callable[[tuple[Mapping[str, object], ...]], ConsumerResult],
) -> ConsumerResult:
    """Send an immutable, ordered batch to an injected external trainer.

    ReMemAgent owns trajectory/advantage alignment; the injected consumer owns
    framework-specific collation, tensors, device placement, optimization, and
    distributed execution. Passing immutable rows prevents the adapter from
    silently reordering or mutating the research batch before handoff.
    """

    if consumer is None:
        raise ValueError("consumer must be provided")

    rows = batch.to_dicts()
    return consumer(rows)
