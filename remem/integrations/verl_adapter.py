"""Framework-facing dispatch boundaries for external verl agent-loop execution.

The adapter validates externally generated token sequences before attaching
ReMemAgent-owned reward and provenance metadata. Framework-specific generation,
model execution, batching, and optimization remain outside this package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Protocol, TypeVar

from remem.integrations.verl import VerlTrainingBatch, VerlTrajectory
from remem.integrations.verl_contract import validate_agent_loop_output

ConsumerResult_co = TypeVar("ConsumerResult_co", covariant=True)


class VerlTrainingConsumer(Protocol[ConsumerResult_co]):
    """Callable protocol implemented by an external trainer or collator."""

    def __call__(
        self,
        rows: tuple[Mapping[str, object], ...],
    ) -> ConsumerResult_co:
        """Consume one ordered batch of serialized training rows."""


class AsyncVerlAgentLoop(Protocol):
    """Async protocol for a dependency-free wrapper around ``AgentLoopBase``."""

    async def __call__(
        self,
        prompt_ids: tuple[int, ...],
    ) -> Mapping[str, Sequence[int]]:
        """Run one external agent loop and return its token-level output."""


def adapt_agent_loop_output(
    output: Mapping[str, Sequence[int]],
    *,
    reward: float,
    metadata: Mapping[str, object] | None = None,
) -> VerlTrajectory:
    """Convert a validated external agent-loop output into a trajectory.

    The token fields are validated exactly once and copied into immutable
    tuples. Reward and provenance metadata are supplied by the caller because
    the external loop owns environment execution while ReMemAgent owns the
    research record that links the outcome to memory and episode context.
    """

    if not isfinite(reward):
        raise ValueError("reward must be finite")

    validated = validate_agent_loop_output(output)
    return VerlTrajectory(
        prompt_ids=validated.prompt_ids,
        response_ids=validated.response_ids,
        response_mask=validated.response_mask,
        reward=reward,
        metadata=dict(metadata or {}),
    )


async def run_agent_loop(
    agent_loop: AsyncVerlAgentLoop,
    *,
    prompt_ids: Sequence[int],
    reward: float,
    metadata: Mapping[str, object] | None = None,
) -> VerlTrajectory:
    """Execute one external async agent loop and normalize its trajectory.

    This is the narrow integration boundary for verl's ``AgentLoopBase.run``
    coroutine. The external implementation remains responsible for model
    generation, tool/environment interaction, and reward calculation inputs.
    ReMemAgent validates the returned token contract and records provenance
    without importing verl or assuming an inference backend.
    """

    normalized_prompt_ids = tuple(prompt_ids)
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in normalized_prompt_ids):
        raise ValueError("prompt_ids must contain non-negative integer token IDs")

    output = await agent_loop(normalized_prompt_ids)
    return adapt_agent_loop_output(output, reward=reward, metadata=metadata)


def dispatch_verl_training_batch(
    batch: VerlTrainingBatch,
    consumer: VerlTrainingConsumer[ConsumerResult_co],
) -> ConsumerResult_co:
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
