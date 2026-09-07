"""Regression tests for normalized environment transition ownership."""

from remem.environments.base import StepResult


def test_step_result_detaches_nested_info_from_environment_owned_metadata() -> None:
    environment_info = {"metrics": {"success": True}, "tags": ["alfworld"]}

    result = StepResult("observation", 1.0, True, False, environment_info)

    environment_info["metrics"]["success"] = False
    environment_info["tags"].append("mutated")

    assert result.info == {"metrics": {"success": True}, "tags": ["alfworld"]}


def test_step_result_keeps_nested_info_independent_between_instances() -> None:
    source_info = {"metrics": {"score": 0.5}}

    first = StepResult("first", 0.0, False, False, source_info)
    second = StepResult("second", 0.0, False, False, source_info)

    first.info["metrics"]["score"] = 1.0

    assert second.info["metrics"]["score"] == 0.5
    assert source_info["metrics"]["score"] == 0.5
