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
    """Preserve an episode seed while adapting ALFWorld's global RNG API."""

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
    """Scope WebShop's module-level Python RNG to each benchmark reset."""

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
    """Build a seed-aware factory for the upstream ALFWorld text environment."""

    if batch_size != 1:
        raise ValueError("ReMemAgent's ALFWorld adapter requires batch_size=1")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(train_eval, str) or not train_eval.strip():
        raise ValueError("train_eval must be a non-empty string")
    if env_type is not None and (not isinstance(env_type, str) or not env_type.strip()):
        raise ValueError("env_type must be a non-empty string when provided")
    config_env = config.get("env", {})
    if not isinstance(config_env, Mapping):
        raise TypeError("config['env'] must be a mapping when provided")

    try:
        from alfworld.agents.environment import get_environment
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld is required for the concrete ALFWorld integration; "
            "install alfworld before creating this factory"
        ) from exc

    selected_env_type = env_type or str(config_env.get("type", "AlfredTWEnv"))
    environment_class = get_environment(selected_env_type)

    def create_environment(seed: int) -> Any:
        """Create one isolated upstream ALFWorld environment."""

        _validate_seed(seed)
        with _ALFWORLD_RANDOM_LOCK:
            previous_state = random.getstate()
            random.seed(seed)
            environment: Any | None = None
            try:
                environment = environment_class(config, train_eval=train_eval)
                initialized_environment = environment.init_env(batch_size=1)
            except Exception:
                if environment is not None:
                    _close_if_supported(environment)
                raise
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
    """Build a factory for the upstream WebShop Gym text environment."""

    if not isinstance(observation_mode, str) or not observation_mode.strip():
        raise ValueError("observation_mode must be a non-empty string")
    if not isinstance(environment_id, str) or not environment_id.strip():
        raise ValueError("environment_id must be a non-empty string")
    if num_products is not None:
        if isinstance(num_products, bool) or not isinstance(num_products, int):
            raise TypeError("num_products must be an integer when provided")
        if num_products <= 0:
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
