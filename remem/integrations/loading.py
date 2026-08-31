"""Utilities for resolving caller-owned integration callables."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast


def resolve_callable(specification: str) -> Callable[..., Any]:
    """Resolve a callable from explicit ``module:attribute`` notation.

    Attribute paths may contain dots, allowing factories nested inside modules or
    configuration namespaces to be loaded without importing third-party packages
    into the ReMemAgent core.
    """

    module_name, attribute_path = split_callable_specification(specification)
    module = importlib.import_module(module_name)
    value: Any = module
    for attribute_name in attribute_path.split("."):
        if not attribute_name.strip():
            raise ValueError(f"invalid callable specification: {specification!r}")
        try:
            value = getattr(value, attribute_name)
        except AttributeError as exc:
            raise ValueError(f"callable attribute not found: {specification!r}") from exc

    if not callable(value):
        raise TypeError(f"resolved value is not callable: {specification!r}")
    return cast(Callable[..., Any], value)


def split_callable_specification(specification: str) -> tuple[str, str]:
    """Validate and split explicit ``module:attribute`` callable notation."""

    module_name, separator, attribute_path = specification.partition(":")
    if not separator or not module_name.strip() or not attribute_path.strip():
        raise ValueError(f"invalid callable specification: {specification!r}")
    return module_name.strip(), attribute_path.strip()
