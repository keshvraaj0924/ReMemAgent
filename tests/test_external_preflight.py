from dataclasses import replace

import pytest

from experiments.external_benchmark import ExternalBenchmarkSpec
from experiments.external_preflight import (
    run_repeated_external_benchmarks_with_preflight,
    validate_repeated_external_benchmark_runtime,
)
from experiments.runtime_provenance import (
    CLEAN_STATE,
    RUNTIME_PROVENANCE_SCHEMA_VERSION,
    RuntimeProvenance,
)
from experiments.runtime_requirements import RuntimeRequirements
from remem.benchmark import BenchmarkSuiteRunner
from tests.test_external_benchmark import CLOSED_SEEDS


def _build_spec() -> ExternalBenchmarkSpec:
    return ExternalBenchmarkSpec(
        benchmark_name="webshop-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory="tests.test_external_benchmark:make_environment",
        policy_factory="tests.test_external_benchmark:make_policy",
        success_evaluator="tests.test_external_benchmark:evaluate_success",
        seed=None,
    )


def _build_runtime_provenance(*, code_revision: str = "expected-revision") -> RuntimeProvenance:
    return RuntimeProvenance(
        schema_version=RUNTIME_PROVENANCE_SCHEMA_VERSION,
        code_revision=code_revision,
        working_tree_state=CLEAN_STATE,
        python_version="3.12.0",
        platform="test-platform",
        package_version="0.0.0-test",
        dependency_fingerprint="0" * 64,
        dependency_versions={"alfworld": "1.0.0", "transformers": "2.0.0"},
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


def test_repeated_runtime_preflight_rejects_single_seed_on_spec() -> None:
    CLOSED_SEEDS.clear()
    spec = replace(_build_spec(), seed=999)

    with pytest.raises(ValueError, match="spec.seed must be None"):
        validate_repeated_external_benchmark_runtime(spec, [11, 17])

    assert CLOSED_SEEDS == []


def test_repeated_runtime_preflight_rejects_empty_seed_sequence() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        validate_repeated_external_benchmark_runtime(_build_spec(), [])


def test_repeated_runtime_preflight_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_repeated_external_benchmark_runtime(_build_spec(), [7, 7])


def test_repeated_runtime_preflight_rejects_non_integer_seed() -> None:
    with pytest.raises(TypeError, match="only integers"):
        validate_repeated_external_benchmark_runtime(_build_spec(), [7, "11"])  # type: ignore[list-item]


def test_repeated_runtime_preflight_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError, match="only integers"):
        validate_repeated_external_benchmark_runtime(_build_spec(), [True])  # type: ignore[list-item]


def test_runtime_requirement_failure_prevents_environment_construction(monkeypatch) -> None:
    CLOSED_SEEDS.clear()
    monkeypatch.setattr(
        "experiments.external_preflight.collect_runtime_provenance",
        lambda: _build_runtime_provenance(code_revision="actual-revision"),
    )
    requirements = RuntimeRequirements(expected_code_revision="expected-revision")

    with pytest.raises(ValueError, match="runtime code revision does not match"):
        validate_repeated_external_benchmark_runtime(
            _build_spec(),
            [11, 17],
            runtime_requirements=requirements,
        )

    assert CLOSED_SEEDS == []


def test_runtime_requirements_validate_before_seed_environment_probes(monkeypatch) -> None:
    CLOSED_SEEDS.clear()
    calls: list[str] = []

    monkeypatch.setattr(
        "experiments.external_preflight.collect_runtime_provenance",
        lambda: _build_runtime_provenance(),
    )

    def record_runtime_validation(provenance, requirements) -> None:
        calls.append("runtime")

    monkeypatch.setattr(
        "experiments.external_preflight.validate_runtime_requirements",
        record_runtime_validation,
    )

    validate_repeated_external_benchmark_runtime(
        _build_spec(),
        [11, 17],
        runtime_requirements=RuntimeRequirements(
            expected_code_revision="expected-revision",
            require_clean_working_tree=True,
            dependency_versions={"alfworld": "1.0.0"},
        ),
    )

    assert calls == ["runtime"]
    assert CLOSED_SEEDS == [11, 17]


def test_repeated_runtime_preflight_rejects_invalid_runtime_requirements() -> None:
    CLOSED_SEEDS.clear()

    with pytest.raises(TypeError, match="RuntimeRequirements instance or None"):
        validate_repeated_external_benchmark_runtime(
            _build_spec(),
            [11, 17],
            runtime_requirements=object(),  # type: ignore[arg-type]
        )

    assert CLOSED_SEEDS == []


def test_preflight_failure_blocks_measured_execution(monkeypatch) -> None:
    events: list[str] = []

    def fake_preflight(*args, **kwargs) -> tuple[object, ...]:
        events.append("preflight")
        raise RuntimeError("external environment unavailable")

    def fail_if_measured(*args, **kwargs):
        pytest.fail("measured execution must not start after preflight failure")

    monkeypatch.setattr(
        "experiments.external_preflight.validate_repeated_external_benchmark_runtime",
        fake_preflight,
    )
    monkeypatch.setattr(
        "experiments.external_preflight.run_repeated_external_benchmarks",
        fail_if_measured,
    )

    with pytest.raises(RuntimeError, match="external environment unavailable"):
        run_repeated_external_benchmarks_with_preflight(_build_spec(), (11, 17))

    assert events == ["preflight"]


def test_preflight_repeated_execution_reuses_injected_runner(monkeypatch) -> None:
    runner = BenchmarkSuiteRunner()
    observed: list[BenchmarkSuiteRunner | None] = []

    monkeypatch.setattr(
        "experiments.external_preflight.validate_repeated_external_benchmark_runtime",
        lambda *args, **kwargs: (),
    )

    def fake_run_repeated(spec, seeds, *, runner=None):
        observed.append(runner)
        return tuple(object() for _ in seeds)

    monkeypatch.setattr(
        "experiments.external_preflight.run_repeated_external_benchmarks",
        fake_run_repeated,
    )

    reports = run_repeated_external_benchmarks_with_preflight(
        _build_spec(),
        (11, 17),
        runner=runner,
    )

    assert len(reports) == 2
    assert observed == [runner]


def test_measured_execution_forwards_runtime_requirements_to_preflight(monkeypatch) -> None:
    observed: list[RuntimeRequirements | None] = []
    requirements = RuntimeRequirements(expected_code_revision="expected-revision")

    def fake_preflight(*args, **kwargs) -> tuple[object, ...]:
        observed.append(kwargs.get("runtime_requirements"))
        return ()

    monkeypatch.setattr(
        "experiments.external_preflight.validate_repeated_external_benchmark_runtime",
        fake_preflight,
    )
    monkeypatch.setattr(
        "experiments.external_preflight.run_repeated_external_benchmarks",
        lambda *args, **kwargs: (),
    )

    run_repeated_external_benchmarks_with_preflight(
        _build_spec(),
        (11, 17),
        runtime_requirements=requirements,
    )

    assert observed == [requirements]
