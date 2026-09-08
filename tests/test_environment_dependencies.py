from __future__ import annotations

from unittest.mock import patch

import pytest

from remem.environments.dependencies import (
    DependencyStatus,
    check_benchmark_dependencies,
    check_optional_dependency,
)


def test_check_optional_dependency_reports_discovered_import() -> None:
    status = check_optional_dependency("json")

    assert status == DependencyStatus("json", True, "json")


def test_check_optional_dependency_supports_distinct_package_and_import_names() -> None:
    status = check_optional_dependency("example-package", import_name="json")

    assert status == DependencyStatus("example-package", True, "json")


def test_check_optional_dependency_reports_missing_import() -> None:
    status = check_optional_dependency("missing-package", import_name="definitely_missing_remem_dependency")

    assert status == DependencyStatus("missing-package", False, "definitely_missing_remem_dependency")


def test_check_optional_dependency_rejects_empty_names() -> None:
    with pytest.raises(ValueError, match="package_name"):
        check_optional_dependency("   ")

    with pytest.raises(ValueError, match="import_name"):
        check_optional_dependency("package", import_name="   ")


def test_benchmark_dependency_check_uses_expected_import_names() -> None:
    with patch("remem.environments.dependencies.find_spec", side_effect=lambda name: object() if name == "alfworld" else None) as find_spec:
        statuses = check_benchmark_dependencies()

    assert statuses == (
        DependencyStatus("alfworld", True, "alfworld"),
        DependencyStatus("webshop", False, "webshop"),
    )
    assert [call.args[0] for call in find_spec.call_args_list] == ["alfworld", "webshop"]
