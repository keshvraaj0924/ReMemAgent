from typing import Any
import sys
import types

import pytest

from remem.integrations.official_benchmarks import (
    build_alfworld_text_environment_factory,
    build_webshop_text_environment_factory,
)


def test_alfworld_factory_rejects_boolean_seed_at_call_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_module = types.ModuleType("alfworld.agents.environment")
    environment_module.get_environment = lambda _env_type: _constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "alfworld", types.ModuleType("alfworld"))
    monkeypatch.setitem(sys.modules, "alfworld.agents", types.ModuleType("alfworld.agents"))
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", environment_module)

    factory = build_alfworld_text_environment_factory({})

    with pytest.raises(TypeError, match="seed must be an integer"):
        factory(True)  # type: ignore[arg-type]


def test_webshop_factory_rejects_non_integer_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    gym_module = types.ModuleType("gym")
    gym_module.make = lambda *_args, **_kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gym", gym_module)

    factory = build_webshop_text_environment_factory()

    with pytest.raises(TypeError, match="seed must be an integer"):
        factory(1.5)  # type: ignore[arg-type]


def _constructor(config: dict[str, Any], train_eval: str) -> Any:
    del config, train_eval
    return _Environment()


class _Environment:
    def init_env(self, *, batch_size: int) -> "_Environment":
        del batch_size
        return self

    def reset(self) -> str:
        return "observation"
