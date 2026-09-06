"""Concrete factory bridges for the upstream ALFWorld and WebShop APIs.

The benchmark packages remain optional runtime dependencies. These helpers keep
imports lazy so the core package remains dependency-free while providing a
first-class, documented path to the real upstream environments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import copy
import random
import threading
from typing import Any, Iterator


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
        with _ALFWORLD_RANDOM_LOCK, _scoped_random_seed(self._seed):
            return self._environment.reset(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate non-reset operations to the upstream environment."""

        return getattr(self._environment, name)


class _SeededWebShopEnvironment:
    """Scope WebShop's module-level RNGs to each benchmark reset."""

    def __init__(self, environment: Any, seed: int) -> None:
        _validate_seed(seed)
        self._environment = environment
        self._seed = seed

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset WebShop under the configured episode seed."""

        if "seed" in kwargs:
            raise TypeError("WebShop adapter owns the reset seed; do not pass seed explicitly")
        with _WEBSHOP_RANDOM_LOCK, _scoped_random_seed(self._seed):
            return self._environment.reset(*args, **kwargs)

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

    selected_env_type = _resolve_alfworld_environment_type(config_env, env_type)
    config_snapshot = _copy_alfworld_config(config)
    normalized_train_eval = train_eval.strip()

    try:
        from alfworld.agents.environment import get_environment
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld is required for the concrete ALFWorld integration; "
            "install alfworld before creating this factory"
        ) from exc

    environment_class = get_environment(selected_env_type)

    def create_environment(seed: int) -> Any:
        """Create one isolated upstream ALFWorld environment."""

        _validate_seed(seed)
        environment: Any | None = None
        with _ALFWORLD_RANDOM_LOCK, _scoped_random_seed(seed):
            try:
                environment_config = _copy_alfworld_config(config_snapshot)
                environment = environment_class(
                    environment_config,
                    train_eval=normalized_train_eval,
                )
                initialized_environment = environment.init_env(batch_size=1)
            except Exception:
                if environment is not None:
                    _close_if_supported(environment)
                raise
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

    normalized_observation_mode = observation_mode.strip()
    normalized_environment_id = environment_id.strip()

    try:
        import gym
    except ImportError as exc:
        raise RuntimeError(
            "Gym is required for the concrete WebShop integration; "
            "install WebShop's runtime dependencies before creating this factory"
        ) from exc
    _validate_webshop_gym_version(gym)

    def create_environment(seed: int) -> Any:
        """Create one upstream WebShop text environment with isolated construction RNG."""

        _validate_seed(seed)
        with _WEBSHOP_RANDOM_LOCK, _scoped_random_seed(seed):
            kwargs: dict[str, Any] = {"observation_mode": normalized_observation_mode}
            if num_products is not None:
                kwargs["num_products"] = num_products
            environment = gym.make(normalized_environment_id, **kwargs)

        reset = getattr(environment, "reset", None)
        if not callable(reset):
            _close_if_supported(environment)
            raise TypeError("WebShop environment must expose reset()")
        return _SeededWebShopEnvironment(environment, seed)

    return create_environment


def _resolve_alfworld_environment_type(
    config_env: Mapping[str, Any],
    env_type: str | None,
) -> str:
    """Resolve and validate the ALFWorld environment type before imports."""

    configured_type = config_env.get("type")
    if configured_type is not None and (
        not isinstance(configured_type, str) or not configured_type.strip()
    ):
        raise ValueError("config['env']['type'] must be a non-empty string when provided")
    if env_type is not None:
        return env_type.strip()
    return configured_type.strip() if configured_type is not None else "AlfredTWEnv"


def _copy_alfworld_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Detach caller-owned ALFWorld configuration from later mutations."""

    try:
        return copy.deepcopy(dict(config))
    except (TypeError, ValueError) as exc:
        raise TypeError("config must be deep-copyable for reproducible ALFWorld runs") from exc


@contextmanager
def _scoped_random_seed(seed: int) -> Iterator[None]:
    """Scope Python and NumPy RNG state to one external benchmark operation.

    ALFWorld and WebShop are legacy integrations whose randomness is not
    consistently exposed through a modern ``reset(seed=...)`` API. The bridge
    therefore scopes the legacy global RNGs and restores their exact caller
    state afterward. NumPy remains optional so the ReMemAgent core does not
    acquire a runtime dependency merely by importing this module.
    """

    _validate_seed(seed)
    previous_python_state = random.getstate()
    numpy_module = _load_numpy_if_available()
    previous_numpy_state = numpy_module.random.get_state() if numpy_module is not None else None
    random.seed(seed)
    if numpy_module is not None:
        numpy_module.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(previous_python_state)
        if numpy_module is not None and previous_numpy_state is not None:
            numpy_module.random.set_state(previous_numpy_state)


def _load_numpy_if_available() -> Any | None:
    """Return NumPy when installed without making it a core package dependency."""

    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _validate_webshop_gym_version(gym_module: Any) -> None:
    """Reject Gym 0.24, which is known to break WebShop environment creation."""

    version = getattr(gym_module, "__version__", None)
    if not isinstance(version, str):
        return
    components = version.split(".")
    if len(components) < 2 or components[0] != "0" or components[1] != "24":
        return
    raise RuntimeError(
        "WebShop is incompatible with Gym 0.24.x because gym.make may invoke "
        "reset/step during environment construction; use a WebShop-supported "
        "Gym release such as 0.23.1 instead"
    )


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
