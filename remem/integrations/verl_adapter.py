"""Framework-facing dispatch boundaries for external verl agent-loop execution.

The adapter validates externally generated token sequences before attaching
ReMemAgent-owned reward and provenance metadata. Framework-specific generation,
model execution, batching, and optimization remain outside this package.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol, TypeVar, cast

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


class AgentLoopOutputLike(Protocol):
    """Minimal protocol for the model-backed output returned by verl AgentLoopBase."""

    def model_dump(self) -> Mapping[str, object]:
        """Serialize the external output into a field mapping."""


class AsyncVerlAgentLoop(Protocol):
    """Protocol matching the async callable shape of a verl ``AgentLoopBase.run`` method."""

    def __call__(
        self,
        sampling_params: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[Mapping[str, object] | AgentLoopOutputLike]:
        """Run one external agent loop and return its token-level output."""


@dataclass(frozen=True, slots=True)
class AgentLoopRequest:
    """One ordered request for an external async agent loop."""

    sampling_params: Mapping[str, Any]
    reward: float
    metadata: Mapping[str, object] = field(default_factory=dict)
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def _normalize_agent_loop_output(
    output: Mapping[str, object] | AgentLoopOutputLike,
) -> Mapping[str, object]:
    """Normalize a mapping or a verl-style model output into a field mapping."""

    if isinstance(output, Mapping):
        return output

    dumped_output = output.model_dump()
    if not isinstance(dumped_output, Mapping):
        raise TypeError("agent loop model_dump() must return a mapping")
    return cast(Mapping[str, object], dumped_output)


def _merge_external_extra_fields(
    metadata: Mapping[str, object] | None,
    extra_fields: Mapping[str, object],
) -> dict[str, object]:
    """Merge verl dynamic fields without silently overwriting research metadata."""

    normalized_metadata = dict(metadata or {})
    if not extra_fields:
        return normalized_metadata
    if "verl_extra_fields" in normalized_metadata:
        raise ValueError("metadata already contains reserved key 'verl_extra_fields'")
    normalized_metadata["verl_extra_fields"] = dict(extra_fields)
    return normalized_metadata


def adapt_agent_loop_output(
    output: Mapping[str, object] | AgentLoopOutputLike,
    *,
    reward: float,
    metadata: Mapping[str, object] | None = None,
) -> VerlTrajectory:
    """Convert a validated external agent-loop output into a trajectory.

    The token fields are validated exactly once and copied into immutable
    tuples. Reward and provenance metadata are supplied by the caller because
    the external loop owns environment execution while ReMemAgent owns the
    research record that links the outcome to memory and episode context.
    verl's dynamic ``extra_fields`` are preserved under a reserved metadata key
    rather than silently discarded.
    """

    if not isfinite(reward):
        raise ValueError("reward must be finite")

    normalized_output = _normalize_agent_loop_output(output)
    validated = validate_agent_loop_output(normalized_output)
    trajectory_metadata = _merge_external_extra_fields(
        metadata,
        validated.extra_fields,
    )
    return VerlTrajectory(
        prompt_ids=validated.prompt_ids,
        response_ids=validated.response_ids,
        response_mask=validated.response_mask,
        reward=reward,
        metadata=trajectory_metadata,
    )


async def run_agent_loop(
    agent_loop: AsyncVerlAgentLoop,
    *,
    sampling_params: Mapping[str, Any],
    reward: float,
    metadata: Mapping[str, object] | None = None,
    **kwargs: Any,
) -> VerlTrajectory:
    """Execute a verl-compatible async agent loop and normalize its trajectory.

    The bridge mirrors verl's real ``AgentLoopBase.run`` boundary: sampling
    parameters are passed explicitly and dataset-specific fields are forwarded
    through ``kwargs``. ReMemAgent does not construct prompts, tokenize inputs,
    or assume a particular inference server. It only validates and records the
    token-level output after the external coroutine completes.
    """

    output = await agent_loop(sampling_params, **kwargs)
    return adapt_agent_loop_output(output, reward=reward, metadata=metadata)


async def run_agent_loop_batch(
    agent_loop: AsyncVerlAgentLoop,
    requests: Sequence[AgentLoopRequest],
    *,
    max_concurrency: int | None = None,
) -> tuple[VerlTrajectory, ...]:
    """Execute ordered agent-loop requests concurrently and preserve input order.

    Concurrency is bounded only when ``max_concurrency`` is supplied. The
    external framework remains responsible for inference-server scheduling;
    this helper only coordinates independent requests and normalizes their
    results into deterministic trajectory records.
    """

    if max_concurrency is not None and max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive when provided")

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

    async def execute(request: AgentLoopRequest) -> VerlTrajectory:
        if semaphore is None:
            return await run_agent_loop(
                agent_loop,
                sampling_params=request.sampling_params,
                reward=request.reward,
                metadata=request.metadata,
                **request.kwargs,
            )
        async with semaphore:
            return await run_agent_loop(
                agent_loop,
                sampling_params=request.sampling_params,
                reward=request.reward,
                metadata=request.metadata,
                **request.kwargs,
            )

    return tuple(await asyncio.gather(*(execute(request) for request in requests)))


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
