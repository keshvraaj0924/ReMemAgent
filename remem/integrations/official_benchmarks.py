"""Concrete factory bridges for the upstream ALFWorld and WebShop APIs.

The benchmark packages remain optional runtime dependencies. These helpers keep
imports lazy so the core package remains dependency-free while providing a
first-class, documented path to the real upstream environments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import random
import threading
from typing import Any


RawEnvironmentFactory = Callable[[int], Any]
_ALFWORLD_RANDOM_LOCK = threading.Lock()
_WEBSHOP_RANDOM_LOCK = threading.Lock()


class _SeededAlfWorldEnvironment:
    """Preserve an episode seed while adapting ALFWorld's global RNG API.

    The upstream ALFWorld text environment selects tasks through Python's module
    level ``random`` state and does not expose a seed argument on ``reset``.
    ReMemAgent therefore scopes the seed to each reset, restores the caller's
    RNG state afterwards, and serializes that small critical section so parallel
    benchmark workers cannot interleave global RNG mutations.
    """

    def __init__(self, environment: Any, seed: int) -> None:
        _validate_seed(seed)
        self._environment = environment
        self._seed = seed

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset ALFWorld deterministically without leaking global RNG state."""

        if "seed" in kwargs:
            raise TypeError("ALFWorld adapter owns the reset seed; do not pass seed explicitly")
        with _ALFWORLD_RANDOM_LOCK:
            previous_state = random.getstate()
            random.seed(self._seed)
            try:
                return self._environment.reset(*args, **kwargs)
            finally:
                random.setstate(previous_state)

    def __getattr__(self, name: str) -> Any:
        """Delegate non-reset operations to the upstream environment."""

        return getattr(self._environment, name)


class _SeededWebShopEnvironment:
    """Scope WebShop's module-level Python RNG to each benchmark reset.

    The upstream WebShop text environment exposes ``reset(session=None,
    instruction_text=None)`` rather than Gym's seeded reset contract. Its task
    selection and generated session identifiers use Python's global ``random``
    module, so a per-reset RNG scope is required for seed-level reproducibility.
    """

    def __init__(self, environment: Any, seed: int) -> None:
        _validate_seed(seed)
        self._environment = environment
        self._seed = seed

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset WebShop under the configured episode seed."""

        if "seed" in kwargs:
            raise TypeError("WebShop adapter owns the reset seed; do not pass seed explicitly")
        with _WEBSHOP_RANDOM_LOCK:
            previous_state = random.getstate()
            random.seed(self._seed)
            try:
                return self._environment.reset(*args, **kwargs)
            finally:
                random.setstate(previous_state)

    def __getattr__(self, name: str) -> Any:
        """Delegate non-reset operations to the upstream environment."""

        return getattr(self._environment, name)


def build_alfworld_text_environment_factory(
    config: Mapping[str, Any],
    *,
    env_type: str | None = None,
    train_eval: str = "eval",
    batch_size: int = 1,
) -> RawEnvironmentFactory:
    """Build a seed-aware factory for the upstream ALFWorld text environment.

    ALFWorld exposes a batch-oriented ``get_environment(...).init_env`` API.
    ReMemAgent's :class:`AlfWorldAdapter` removes the singleton batch dimension,
    while this factory owns upstream environment construction and deterministic
    episode seeding.
    """

    if batch_size != 1:
        raise ValueError("ReMemAgent's ALFWorld adapter requires batch_size=1")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not train_eval.strip():
        raise ValueError("train_eval must not be empty")

    try:
        from alfworld.agents.environment import get_environment
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld is required for the concrete ALFWorld integration; "
            "install alfworld before creating this factory"
        ) from exc

    selected_env_type = env_type or str(config.get("env", {}).get("type", "AlfredTWEnv"))
    environment_class = get_environment(selected_env_type)

    def create_environment(seed: int) -> Any:
        """Create one isolated upstream ALFWorld environment."""

        _validate_seed(seed)
        with _ALFWORLD_RANDOM_LOCK:
            previous_state = random.getstate()
            random.seed(seed)
            try:
                environment = environment_class(config, train_eval=train_eval)
                initialized_environment = environment.init_env(batch_size=1)
            finally:
                random.setstate(previous_state)
        return _SeededAlfWorldEnvironment(initialized_environment, seed)

    return create_environment


def build_webshop_text_environment_factory(
    *,
    num_products: int | None = None,
    observation_mode: str = "text",
    environment_id: str = "WebAgentTextEnv-v0",
) -> RawEnvironmentFactory:
    """Build a factory for the upstream WebShop Gym text environment.

    WebShop's simple environment is exposed as ``WebAgentTextEnv-v0`` through
    its Gym registration. The optional ``num_products`` argument mirrors the
    upstream constructor configuration used for smaller local runs.
    """

    if not observation_mode.strip():
        raise ValueError("observation_mode must not be empty")
    if not environment_id.strip():
        raise ValueError("environment_id must not be empty")
    if num_products is not None and num_products <= 0:
        raise ValueError("num_products must be positive when provided")

    try:
        import gym
    except ImportError as exc:
        raise RuntimeError(
            "Gym is required for the concrete WebShop integration; "
            "install WebShop's runtime dependencies before creating this factory"
        ) from exc

    def create_environment(seed: int) -> Any:
        """Create one upstream WebShop text environment with isolated construction RNG."""

        _validate_seed(seed)
        with _WEBSHOP_RANDOM_LOCK:
            previous_state = random.getstate()
            random.seed(seed)
            try:
                kwargs: dict[str, Any] = {"observation_mode": observation_mode}
                if num_products is not None:
                    kwargs["num_products"] = num_products
                environment = gym.make(environment_id, **kwargs)
            finally:
                random.setstate(previous_state)

        reset = getattr(environment, "reset", None)
        if not callable(reset):
            _close_if_supported(environment)
            raise TypeError("WebShop environment must expose reset()")
        return _SeededWebShopEnvironment(environment, seed)

    return create_environment


def _validate_seed(seed: int) -> None:
    """Reject non-integer and boolean seeds at the external integration boundary."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def _close_if_supported(environment: Any) -> None:
    """Close an environment when cleanup is available, preserving the primary error."""

    close = getattr(environment, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException:
        return


__all__ = [
    "RawEnvironmentFactory",
    "build_alfworld_text_environment_factory",
    "build_webshop_text_environment_factory",
]
