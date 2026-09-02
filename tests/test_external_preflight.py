import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.external_preflight import validate_repeated_external_benchmark_runtime
from tests.test_external_benchmark import CLOSED_SEEDS


def _build_spec() -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="webshop-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=999,
    )


def test_repeated_runtime_preflight_uses_each_requested_seed() -> None:
    CLOSED_SEEDS.clear()

    reports = validate_repeated_external_benchmark_runtime(_build_spec(), [11, 17])

    assert [report.initial_observation for report in reports] == ["seed-11", "seed-17"]
    assert CLOSED_SEEDS == [11, 17]


def test_repeated_runtime_preflight_probes_optional_action_for_each_seed() -> None:
    CLOSED_SEEDS.clear()

    reports = validate_repeated_external_benchmark_runtime(
        _build_spec(),
        (3, 5),
        probe_action="look",
    )

    assert [report.step_result.reward for report in reports if report.step_result] == [1.0, 1.0]
    assert CLOSED_SEEDS == [3, 5]


def test_repeated_runtime_preflight_rejects_empty_seed_sequence() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        validate_repeated_external_benchmark_runtime(_build_spec(), [])


def test_repeated_runtime_preflight_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_repeated_external_benchmark_runtime(_build_spec(), [7, 7])
