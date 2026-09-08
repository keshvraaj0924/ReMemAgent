"""Optional benchmark dependency diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    """Availability of an optional Python dependency."""

    package_name: str
    available: bool
    import_name: str


def check_optional_dependency(
    package_name: str, *, import_name: str | None = None
) -> DependencyStatus:
    """Check whether an optional package can be discovered without importing it."""

    if not package_name.strip():
        raise ValueError("package_name must be non-empty")
    resolved_import_name = import_name or package_name
    if not resolved_import_name.strip():
        raise ValueError("import_name must be non-empty")
    return DependencyStatus(
        package_name=package_name,
        available=find_spec(resolved_import_name) is not None,
        import_name=resolved_import_name,
    )


def check_benchmark_dependencies() -> tuple[DependencyStatus, ...]:
    """Return dependency availability for the supported external benchmarks.

    ALFWorld is distributed as the ``alfworld`` package. WebShop is commonly
    installed from its source checkout rather than a stable PyPI package, so
    this check looks for its ``webshop`` import namespace only.
    """

    return (
        check_optional_dependency("alfworld"),
        check_optional_dependency("webshop"),
    )


__all__ = [
    "DependencyStatus",
    "check_benchmark_dependencies",
    "check_optional_dependency",
]
