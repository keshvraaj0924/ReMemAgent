from remem.benchmark import BenchmarkRunConfiguration
from remem.reproducibility import benchmark_configuration_digest


def make_configuration(**overrides: object) -> BenchmarkRunConfiguration:
    values: dict[str, object] = {
        "benchmark_name": "alfworld-smoke",
        "episode_count": 8,
        "max_steps": 12,
        "seed": 41,
        "environment_factory": "package.module:make_environment",
        "policy_factory": "package.module:make_policy",
        "success_evaluator": "package.module:evaluate_success",
        "transfer_success_evaluator": "package.module:evaluate_transfer",
        "minimum_trust": 0.25,
    }
    values.update(overrides)
    return BenchmarkRunConfiguration(**values)


def test_benchmark_configuration_digest_is_deterministic() -> None:
    configuration = make_configuration()

    assert benchmark_configuration_digest(configuration) == benchmark_configuration_digest(
        make_configuration()
    )
    assert len(benchmark_configuration_digest(configuration)) == 64


def test_benchmark_configuration_digest_changes_when_configuration_changes() -> None:
    configuration = make_configuration()
    changed_configuration = make_configuration(seed=42)

    assert benchmark_configuration_digest(configuration) != benchmark_configuration_digest(
        changed_configuration
    )


def test_benchmark_configuration_digest_includes_provenance_fields() -> None:
    configuration = make_configuration()
    changed_provenance = make_configuration(policy_factory="other.module:make_policy")

    assert benchmark_configuration_digest(configuration) != benchmark_configuration_digest(
        changed_provenance
    )
