from pathlib import Path
from typing import Any

import pytest

from experiments.benchmark_report import save_benchmark_report
from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    resolve_callable,
    run_external_benchmark,
    run_repeated_external_benchmarks,
    validate_external_benchmark,
    validate_external_benchmark_runtime,
    validate_seed_sequence,
)
from remem.benchmark import BenchmarkSuiteRunner
from remem.environments.base import StepResult
from remem.memory.store import MemoryStore


CLOSED_SEEDS: list[int] = []


class FakeEnvironment:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.closed = False

    def reset(self, **kwargs: object) -> str:
        return f"seed-{self.seed}"

    def step(self, action: str) -> StepResult:
        return StepResult(
            observation="done",
            reward=1.0,
            terminated=True,
            truncated=False,
        )

    def close(self) -> None:
        self.closed = True
        CLOSED_SEEDS.append(self.seed)


class BrokenCloseEnvironment(FakeEnvironment):
    def close(self) -> None:
        CLOSED_SEEDS.append(self.seed)
        raise RuntimeError("close failed")


def make_environment(seed: int) -> FakeEnvironment:
    return FakeEnvironment(seed)


def make_broken_close_environment(seed: int) -> BrokenCloseEnvironment:
    return BrokenCloseEnvironment(seed)


def make_policy(seed: int, store: MemoryStore):
    del store
    return lambda state: f"act-{seed}"


def make_invalid_policy(seed: int, store: MemoryStore):
    del seed, store
    return lambda state: ""


def evaluate_success(episode) -> bool:
    return episode.total_reward > 0


def test_external_benchmark_spec_rejects_invalid_episode_count() -> None:
    with pytest.raises(ValueError, match="episode_count"):
        _build_spec(episode_count=-1)


def test_external_benchmark_spec_rejects_invalid_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        _build_spec(max_steps=0)


def test_external_benchmark_spec_rejects_invalid_seed_type() -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        _build_spec(seed="7")


def test_external_benchmark_spec_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        _build_spec(seed=True)


def test_external_benchmark_spec_rejects_invalid_minimum_trust() -> None:
    with pytest.raises(ValueError, match="minimum_trust"):
        _build_spec(minimum_trust=1.1)


def test_external_benchmark_spec_rejects_invalid_callable_field() -> None:
    with pytest.raises(ValueError, match="policy_factory"):
        _build_spec(policy_factory="not-a-callable-spec")


def test_validate_seed_sequence_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        validate_seed_sequence(())


def test_validate_seed_sequence_rejects_non_integer_seed() -> None:
    with pytest.raises(TypeError, match="only integers"):
        validate_seed_sequence((7, "11"))  # type: ignore[arg-type]


def test_validate_seed_sequence_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError, match="only integers"):
        validate_seed_sequence((True,))  # type: ignore[arg-type]


def test_validate_seed_sequence_rejects_duplicate_seed() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_seed_sequence((7, 7))


def test_validate_seed_sequence_preserves_order() -> None:
    assert validate_seed_sequence((23, 7, -1)) == (23, 7, -1)


def test_resolve_callable_supports_nested_attributes() -> None:
    resolved = resolve_callable("tests.test_external_benchmark:make_environment")
    assert resolved.__name__ == "make_environment"
    assert resolved.__module__ == "tests.test_external_benchmark"


def test_resolve_callable_rejects_malformed_specification() -> None:
    with pytest.raises(ValueError, match="invalid callable specification"):
        resolve_callable("tests.test_external_benchmark")


def test_resolve_callable_rejects_non_callable_attribute() -> None:
    with pytest.raises(TypeError, match="not callable"):
        resolve_callable("tests.test_external_benchmark:CLOSED_SEEDS")


def test_validate_external_benchmark_resolves_all_callables_without_running() -> None:
    validate_external_benchmark(_build_spec(seed=19))


def test_validate_external_benchmark_rejects_unresolvable_callable() -> None:
    spec = _build_spec(success_evaluator="tests.test_external_benchmark:missing_evaluator")

    with pytest.raises(ValueError, match="callable attribute not found"):
        validate_external_benchmark(spec)


def test_validate_external_benchmark_runtime_probes_reset() -> None:
    CLOSED_SEEDS.clear()

    report = validate_external_benchmark_runtime(_build_spec(seed=19))

    assert report.initial_observation == "seed-19"
    assert report.step_result is None
    assert CLOSED_SEEDS == [19]


def test_validate_external_benchmark_runtime_probes_one_step() -> None:
    CLOSED_SEEDS.clear()

    report = validate_external_benchmark_runtime(_build_spec(seed=23), probe_action="look")

    assert report.initial_observation == "seed-23"
    assert report.step_result is not None
    assert report.step_result.observation == "done"
    assert report.step_result.reward == 1.0
    assert report.step_result.done
    assert CLOSED_SEEDS == [23]


def test_validate_external_benchmark_runtime_closes_environment_when_policy_probe_fails() -> None:
    CLOSED_SEEDS.clear()
    spec = _build_spec(policy_factory="tests.test_external_benchmark:make_invalid_policy")

    with pytest.raises(ValueError, match="non-empty string action"):
        validate_external_benchmark_runtime(spec)

    assert CLOSED_SEEDS == [7]


def test_validate_external_benchmark_runtime_preserves_probe_failure_when_close_fails() -> None:
    CLOSED_SEEDS.clear()
    spec = _build_spec(
        environment_factory="tests.test_external_benchmark:make_broken_close_environment",
        policy_factory="tests.test_external_benchmark:make_invalid_policy",
    )

    with pytest.raises(ValueError, match="non-empty string action"):
        validate_external_benchmark_runtime(spec)

    assert CLOSED_SEEDS == [7]


def test_validate_external_benchmark_runtime_rejects_invalid_policy_action() -> None:
    spec = _build_spec(policy_factory="tests.test_external_benchmark:make_invalid_policy")

    with pytest.raises(ValueError, match="non-empty string action"):
        validate_external_benchmark_runtime(spec)


def test_run_external_benchmark_passes_seed_to_factories() -> None:
    report = run_external_benchmark(_build_spec(seed=100, episode_count=2))

    assert report.seed == 100
    assert [episode.episode.initial_observation for episode in report.episodes] == [
        "seed-100",
        "seed-101",
    ]
    assert [episode.episode.steps[0].action for episode in report.episodes] == [
        "act-100",
        "act-101",
    ]


def test_run_repeated_external_benchmarks_rejects_invalid_seed_type() -> None:
    with pytest.raises(TypeError, match="only integers"):
        run_repeated_external_benchmarks(_build_spec(), (7, "11"))  # type: ignore[arg-type]


def test_run_repeated_external_benchmarks_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError, match="only integers"):
        run_repeated_external_benchmarks(_build_spec(), (True,))  # type: ignore[arg-type]


def test_run_repeated_external_benchmarks_preserves_seed_order() -> None:
    reports = run_repeated_external_benchmarks(_build_spec(), (23, 7))

    assert [report.seed for report in reports] == [23, 7]


def test_run_external_benchmark_records_callable_provenance() -> None:
    report = run_external_benchmark(_build_spec(seed=7, minimum_trust=0.65))

    assert report.configuration is not None
    assert report.configuration.benchmark_name == "webshop-smoke"
    assert report.configuration.episode_count == 1
    assert report.configuration.max_steps == 1
    assert report.configuration.seed == 7
    assert report.configuration.environment_factory == (
        "tests.test_external_benchmark:make_environment"
    )
    assert report.configuration.policy_factory == "tests.test_external_benchmark:make_policy"
    assert report.configuration.success_evaluator == "tests.test_external_benchmark:evaluate_success"
    assert report.configuration.transfer_success_evaluator is None
    assert report.configuration.minimum_trust == 0.65


def test_save_benchmark_report_writes_measured_json(tmp_path: Path) -> None:
    report = run_external_benchmark(_build_spec(seed=7), runner=BenchmarkSuiteRunner())

    output = save_benchmark_report(report, tmp_path / "report.json")
    payload = output.read_text(encoding="utf-8")

    assert '"benchmark_name": "webshop-smoke"' in payload
    assert '"seed": 7' in payload
    assert '"minimum_trust": 0.0' in payload
    assert '"environment_factory": "tests.test_external_benchmark:make_environment"' in payload
    assert '"episode_success": true' in payload


def _build_spec(**overrides: Any) -> ExternalBenchmarkSpec:
    values: dict[str, Any] = {
        "benchmark_name": "webshop-smoke",
        "episode_count": 1,
        "max_steps": 1,
        "environment_factory": "tests.test_external_benchmark:make_environment",
        "policy_factory": "tests.test_external_benchmark:make_policy",
        "success_evaluator": "tests.test_external_benchmark:evaluate_success",
        "seed": 7,
    }
    values.update(overrides)
    return ExternalBenchmarkSpec(**values)
