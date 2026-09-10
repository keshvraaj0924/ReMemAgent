from remem.benchmark import BenchmarkRunConfiguration, BenchmarkRunReport, BenchmarkSuiteRunner
from remem.benchmark_artifacts import (
    benchmark_run_artifact,
    validate_benchmark_run_artifact,
)
from remem.environments.base import StepResult


class FakeEnvironment:
    def reset(self, **kwargs: object) -> str:
        return "state"

    def step(self, action: str) -> StepResult:
        return StepResult(
            observation="done",
            reward=1.0,
            terminated=True,
            truncated=False,
        )

    def close(self) -> None:
        pass


def _build_report() -> BenchmarkRunReport:
    configuration = BenchmarkRunConfiguration(
        benchmark_name="artifact-smoke",
        episode_count=1,
        max_steps=1,
        seed=11,
        environment_factory="tests.fake:environment",
        policy_factory="tests.fake:policy",
        success_evaluator="tests.fake:success",
    )
    return BenchmarkSuiteRunner().run(
        benchmark_name="artifact-smoke",
        episode_count=1,
        max_steps=1,
        environment_factory=lambda index: FakeEnvironment(),
        policy_factory=lambda index, store: lambda state: "act",
        success_evaluator=lambda episode: True,
        seed=11,
        configuration=configuration,
    )


def test_benchmark_run_artifact_embeds_configuration_manifest() -> None:
    report = _build_report()

    artifact = benchmark_run_artifact(report)

    assert "configuration_manifest" in artifact
    manifest = artifact["configuration_manifest"]
    assert manifest["schema_version"] == 1
    assert manifest["payload"]["configuration"]["benchmark_name"] == "artifact-smoke"


def test_benchmark_run_artifact_validation_accepts_matching_manifest() -> None:
    report = _build_report()
    artifact = benchmark_run_artifact(report)

    validate_benchmark_run_artifact(artifact, report)


def test_benchmark_run_artifact_validation_rejects_tampered_manifest() -> None:
    report = _build_report()
    artifact = benchmark_run_artifact(report)
    manifest = artifact["configuration_manifest"]
    manifest["configuration_digest"] = "0" * 64

    try:
        validate_benchmark_run_artifact(artifact, report)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered benchmark artifact must be rejected")


def test_benchmark_run_artifact_requires_configuration_provenance() -> None:
    report = BenchmarkRunReport(
        benchmark_name="unprovenanced-smoke",
        episodes=(),
        final_memory_count=0,
    )

    try:
        benchmark_run_artifact(report)
    except ValueError as exc:
        assert "configuration provenance" in str(exc)
    else:
        raise AssertionError("artifact creation must require configuration provenance")
