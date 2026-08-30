"""Framework-facing dispatch boundary for verl-compatible training batches.

The adapter deliberately depends only on ReMemAgent's normalized batch contract.
A caller injects the external trainer/collator, keeping verl, torch, and
accelerate dependencies outside the research package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TypeVar

from remem.integrations.verl import VerlTrainingBatch

ConsumerResult = TypeVar("ConsumerResult")


class VerlTrainingConsumer(Protocol[ConsumerResult]):
    """Callable protocol implemented by an external trainer or collator."""

    def __call__(
        self,
        rows: tuple[Mapping[str, object], ...],
    ) -> ConsumerResult:
        """Consume one ordered batch of serialized training rows."""


def dispatch_verl_training_batch(
    batch: VerlTrainingBatch,
    consumer: Callable[[tuple[Mapping[str, object], ...]], ConsumerResult],
) -> ConsumerResult:
    """Send an ordered batch to an injected external trainer.

    ReMemAgent owns trajectory/advantage alignment; the injected consumer owns
    framework-specific collation, tensors, device placement, optimization, and
    distributed execution. The adapter passes fresh serialized rows, so a
    consumer may mutate its received dictionaries without mutating the source
    ``VerlTrainingBatch``.
    """

    if consumer is None:
        raise ValueError("consumer must be provided")

    rows = batch.to_dicts()
    return consumer(rows)
