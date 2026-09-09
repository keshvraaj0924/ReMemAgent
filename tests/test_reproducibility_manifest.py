from remem.benchmark import BenchmarkRunConfiguration
from remem.reproducibility_manifest import (
    benchmark_configuration_manifest,
    validate_benchmark_configuration_manifest,
)


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


def test_manifest_contains_versioned_payload_and_digest() -> None:
    manifest = benchmark_configuration_manifest(make_configuration())

    assert set(manifest) == {"schema_version", "configuration_digest", "payload"}
    assert manifest["payload"]["schema_version"] == manifest["schema_version"]
    assert len(manifest["configuration_digest"]) == 64


def test_manifest_validation_accepts_matching_configuration() -> None:
    configuration = make_configuration()

    validate_benchmark_configuration_manifest(
        benchmark_configuration_manifest(configuration),
        configuration,
    )


def test_manifest_validation_rejects_changed_configuration() -> None:
    configuration = make_configuration()
    manifest = benchmark_configuration_manifest(configuration)

    try:
        validate_benchmark_configuration_manifest(manifest, make_configuration(seed=42))
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("expected changed configuration to invalidate manifest")


def test_manifest_validation_rejects_unknown_top_level_fields() -> None:
    configuration = make_configuration()
    manifest = benchmark_configuration_manifest(configuration)
    manifest["unexpected"] = "value"

    try:
        validate_benchmark_configuration_manifest(manifest, configuration)
    except ValueError as error:
        assert "unexpected schema" in str(error)
    else:
        raise AssertionError("expected unknown manifest field to be rejected")
