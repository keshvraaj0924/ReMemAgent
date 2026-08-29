"""Translate executed benchmark trajectories into episodic memories."""

from __future__ import annotations

from dataclasses import dataclass

from remem.execution import EpisodeResult

from .types import MemoryKind, MemoryRecord


@dataclass(frozen=True, slots=True)
class EpisodeMemoryAttribution:
    """Explicit outcome attribution used when recording an episode."""

    episode_success: bool


class EpisodeMemoryRecorder:
    """Create immutable-at-construction episodic records from a trajectory."""

    def record(
        self,
        episode_id: str,
        episode: EpisodeResult,
        attribution: EpisodeMemoryAttribution,
    ) -> list[MemoryRecord]:
        """Create one episodic memory per executed action.

        Success or failure is attributed at the episode level rather than
        inferred from an environment-specific reward convention. This keeps
        benchmark heuristics explicit and prevents the recorder from silently
        treating a numeric reward as a learned success signal.
        """

        normalized_episode_id = episode_id.strip()
        if not normalized_episode_id:
            raise ValueError("episode_id must not be empty")
        if not episode.steps:
            return []

        memories: list[MemoryRecord] = []
        for step in episode.steps:
            memory = MemoryRecord(
                memory_id=f"{normalized_episode_id}:step:{step.step_index}",
                state=step.observation,
                action=step.action,
                outcome=step.result.observation,
                kind=MemoryKind.EPISODIC,
                reward=step.result.reward,
                uses=1,
                successes=1 if attribution.episode_success else 0,
                failures=0 if attribution.episode_success else 1,
                metadata={
                    "episode_id": normalized_episode_id,
                    "step_index": step.step_index,
                    "episode_terminated": episode.terminated,
                    "episode_truncated": episode.truncated,
                },
            )
            memories.append(memory)
        return memories
