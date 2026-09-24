"""Dynamic component loading for real external benchmark executions.

Experiment entry points use explicit module:attribute references so benchmark
and policy integrations can live outside ReMemAgent without adding heavyweight
runtime dependencies to the core package.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def load_callable(component_spec: str) -> Callable[..., Any]:
    """Load and return a callable identified by module:attribute.

    Nested attributes are supported. Import and attribute errors intentionally
    propagate with their original context; malformed specifications and
    non-callable targets receive focused validation errors before execution.
    """

    if not isinstance(component_spec, str):
        raise TypeError("component_spec must be a string")

    normalized_spec = component_spec.strip()
    if not normalized_spec:
        raise ValueError("component_spec must be non-empty")

    module_name, separator, attribute_path = normalized_spec.partition(":")
    if not separator or not module_name.strip() or not attribute_path.strip():
        raise ValueError("component_spec must use the form 'module:attribute'")

    module = importlib.import_module(module_name.strip())
    target: Any = module
    for attribute_name in attribute_path.split("."):
        normalized_attribute = attribute_name.strip()
        if not normalized_attribute:
            raise ValueError("component_spec attribute path must not contain empty segments")
        target = getattr(target, normalized_attribute)

    if not callable(target):
        raise TypeError(f"component {normalized_spec!r} is not callable")
    return target


__all__ = ["load_callable"]
