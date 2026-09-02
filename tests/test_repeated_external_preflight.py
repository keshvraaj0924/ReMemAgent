"""Regression coverage for fail-fast repeated benchmark launch orchestration."""

from experiments import external_preflight
from experiments.external_benchmark import ExternalBenchmarkSpec


def _spec() -> ExternalBenchmarkSpec:
    """Build a dependency-free specification for orchestration tests."""

    return ExternalBenchmarkSpec(
        benchmark_name="alfworld-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory="experiments.smoke_benchmark:build_environment",
        policy_factory=None,
        action_policy_factory="experiments.smoke_benchmark:build_action_policy",
        success_evaluator="experiments.smoke_benchmark:is_success",
    )


def test_repeated_launch_preflights_all_seeds_before_execution(monkeypatch) -> None:
    """Measured execution starts only after every seed passes preflight."""

    events: list[tuple[str, tuple[int, ...]]] = []

    def fake_validate(spec, seeds, *, probe_action=None):
        selected_seeds = tuple(seeds)
        events.append(("preflight", selected_seeds))
        return tuple(object() for _ in selected_seeds)

    def fake_run(spec, seeds):
        selected_seeds = tuple(seeds)
        events.append(("run", selected_seeds))
        return (object(),)

    monkeypatch.setattr(
        external_preflight,
        "validate_repeated_external_benchmark_runtime",
        fake_validate,
    )
    monkeypatch.setattr(
        external_preflight,
        "run_repeated_external_benchmarks",
        fake_run,
    )

    result = external_preflight.run_repeated_external_benchmarks_with_preflight(
        _spec(),
        [11, 17, 23],
        probe_action="look",
    )

    assert len(result) == 1
    assert events == [("preflight", (11, 17, 23)), ("run", (11, 17, 23))]


def test_repeated_launch_does_not_execute_when_preflight_fails(monkeypatch) -> None:
    """A failed seed probe prevents all measured execution."""

    run_called = False

    def fake_validate(spec, seeds, *, probe_action=None):
        raise RuntimeError("seed 17 is not loadable")

    def fake_run(spec, seeds):
        nonlocal run_called
        run_called = True
        return ()

    monkeypatch.setattr(
        external_preflight,
        "validate_repeated_external_benchmark_runtime",
        fake_validate,
    )
    monkeypatch.setattr(
        external_preflight,
        "run_repeated_external_benchmarks",
        fake_run,
    )

    try:
        external_preflight.run_repeated_external_benchmarks_with_preflight(
            _spec(),
            [11, 17, 23],
        )
    except RuntimeError as exc:
        assert str(exc) == "seed 17 is not loadable"
    else:
        raise AssertionError("expected preflight failure")

    assert not run_called
