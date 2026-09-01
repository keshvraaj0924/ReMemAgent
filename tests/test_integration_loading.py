import pytest

from remem.integrations.loading import resolve_callable, split_callable_specification


def callable_fixture() -> str:
    return "loaded"


def test_split_callable_specification_strips_whitespace() -> None:
    assert split_callable_specification(" tests.test_integration_loading : callable_fixture ") == (
        "tests.test_integration_loading",
        "callable_fixture",
    )


def test_resolve_callable_loads_nested_attribute() -> None:
    assert resolve_callable("tests.test_integration_loading:callable_fixture")() == "loaded"


@pytest.mark.parametrize("specification", ["", "module", ":attribute", "module:", "module:."])
def test_split_callable_specification_rejects_invalid_values(specification: str) -> None:
    with pytest.raises(ValueError, match="invalid callable specification"):
        split_callable_specification(specification)


def test_resolve_callable_rejects_missing_attribute() -> None:
    with pytest.raises(ValueError, match="attribute not found"):
        resolve_callable("tests.test_integration_loading:missing")
