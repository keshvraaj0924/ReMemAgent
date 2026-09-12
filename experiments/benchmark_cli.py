"""Command-line entry point for caller-owned external benchmark experiments."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from experiments.benchmark_distribution_config import (
    BenchmarkDistributionConfig,
    build_benchmark_distribution_config,
)
from experiments.benchmark_observability_bundle import persist_benchmark_observability_bundle
from experiments.benchmark_observability_session import BenchmarkObservabilitySession
from experiments.benchmark_report import save_benchmark_report, save_repeated_benchmark_reports
from experiments.benchmark_statistics import summarize_benchmark_reports
from experiments.controlled_benchmark_artifacts import (
    save_controlled_benchmark_result,
    save_controlled_repeated_benchmark_result,
)
from experiments.controlled_external_benchmark import (
    run_controlled_external_benchmark,
    run_controlled_repeated_external_benchmarks,
)
from experiments.external_benchmark import (
    ExternalBenchmarkSpec,
    run_external_benchmark,
    run_repeated_external_benchmarks,
    validate_external_benchmark,
    validate_external_benchmark_runtime,
    validate_seed_sequence,
)
from experiments.external_preflight import (
    run_repeated_external_benchmarks_with_preflight,
    validate_controlled_external_benchmark_runtime,
    validate_repeated_external_benchmark_runtime,
)
from experiments.runtime_provenance import collect_runtime_provenance
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutRequirement
from remem.benchmark import BenchmarkSuiteRunner
from remem.observability import ObservationSnapshot
from remem.observability_distribution_artifacts import DistributionObservationSnapshot

DEFAULT_OUTPUT_PATH = Path("artifacts/benchmark.json")


def parse_args() -> argparse.Namespace:
    """Parse arguments for a caller-owned benchmark experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", help="Comma-separated independent integer seeds")
    parser.add_argument("--environment-factory", required=True)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--policy-factory")
    policy_group.add_argument("--action-policy-factory")
    parser.add_argument("--minimum-trust", type=float, default=0.0)
    parser.add_argument("--success-evaluator", required=True)
    parser.add_argument("--transfer-success-evaluator")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path for the exact-byte benchmark artifact integrity manifest",
    )
    parser.add_argument(
        "--observability-output",
        type=Path,
        help="Optional path for the deterministic benchmark observability snapshot",
    )
    parser.add_argument(
        "--distribution-output",
        type=Path,
        help="Optional path for the fixed-bucket benchmark duration distribution sidecar",
    )
    parser.add_argument(
        "--episode-duration-buckets",
        help=(
            "Comma-separated finite, non-negative, strictly increasing episode-duration "
            "bucket upper bounds in seconds; requires --distribution-output"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing an existing benchmark report, manifest, observability snapshot, "
            "or distribution sidecar"
        ),
    )
    preflight_group = parser.add_mutually_exclusive_group()
    preflight_group.add_argument(
        "--preflight",
        action="store_true",
        help="Resolve configured callables and exit without constructing an environment",
    )
    preflight_group.add_argument(
        "--runtime-preflight",
        action="store_true",
        help="Construct the configured environment and validate its normalized runtime contract",
    )
    preflight_group.add_argument(
        "--repeated-runtime-preflight",
        action="store_true",
        help="Run the runtime contract probe independently for every seed in --seeds",
    )
    parser.add_argument(
        "--preflight-before-run",
        action="store_true",
        help="Runtime-preflight the configured environment before measured execution",
    )
    parser.add_argument(
        "--probe-action",
        help="Optional concrete action used by runtime preflight for one step probe",
    )
    parser.add_argument(
        "--require-code-revision",
        help="Require the exact repository revision before runtime preflight or measurement",
    )
    parser.add_argument(
        "--require-clean-working-tree",
        action="store_true",
        help="Require a clean repository working tree before runtime preflight or measurement",
    )
    parser.add_argument(
        "--require-dependency-version",
        action="append",
        metavar="PACKAGE==VERSION",
        help="Require an exact installed package version; may be specified multiple times",
    )
    parser.add_argument(
        "--source-checkout",
        action="append",
        metavar="NAME=PATH",
        help="Declare a named source checkout path; may be specified multiple times",
    )
    parser.add_argument(
        "--require-source-revision",
        action="append",
        metavar="NAME=REVISION",
        help="Require an exact Git revision for a named source checkout",
    )
    parser.add_argument(
        "--allow-dirty-source-checkout",
        action="append",
        metavar="NAME",
        help="Allow one declared source checkout to have local modifications",
    )
    return parser.parse_args()


def main() -> int:
    """Execute one or more measured benchmark runs and save JSON output."""

    arguments = parse_args()
    spec = ExternalBenchmarkSpec(
        benchmark_name=arguments.benchmark,
        episode_count=arguments.episodes,
        max_steps=arguments.max_steps,
        environment_factory=arguments.environment_factory,
        policy_factory=getattr(arguments, "policy_factory", None),
        action_policy_factory=getattr(arguments, "action_policy_factory", None),
        minimum_trust=getattr(arguments, "minimum_trust", 0.0),
        success_evaluator=arguments.success_evaluator,
        transfer_success_evaluator=getattr(arguments, "transfer_success_evaluator", None),
        seed=getattr(arguments, "seed", None),
    )
    runtime_requirements = _build_runtime_requirements(arguments)
    source_checkout_paths, source_checkout_requirements = _build_source_checkout_contract(arguments)
    source_controlled = source_checkout_paths is not None
    if getattr(arguments, "preflight", False):
        _reject_preflight_only_conflicts(arguments, manifest=True, before_run=True)
        if runtime_requirements is not None:
            raise ValueError(
                "runtime requirements require --runtime-preflight, "
                "--repeated-runtime-preflight, or a measured benchmark run"
            )
        if source_controlled:
            raise ValueError(
                "source-checkout requirements require --runtime-preflight, "
                "--repeated-runtime-preflight, or a measured benchmark run"
            )
        validate_external_benchmark(spec)
        print("benchmark callable preflight succeeded")
        return 0
    if getattr(arguments, "repeated_runtime_preflight", False):
        _reject_preflight_only_conflicts(arguments, manifest=True, before_run=True)
        if getattr(arguments, "seeds", None) is None:
            raise ValueError("--repeated-runtime-preflight requires --seeds")
        seeds = _parse_seeds(arguments.seeds)
        if source_controlled:
            validate_repeated_external_benchmark_runtime(
                spec,
                seeds or (),
                probe_action=getattr(arguments, "probe_action", None),
                runtime_requirements=runtime_requirements,
                source_checkout_paths=source_checkout_paths,
                source_checkout_requirements=source_checkout_requirements,
            )
        elif runtime_requirements is not None:
            validate_repeated_external_benchmark_runtime(
                spec,
                seeds or (),
                probe_action=getattr(arguments, "probe_action", None),
                runtime_requirements=runtime_requirements,
            )
        else:
            validate_repeated_external_benchmark_runtime(
                spec,
                seeds or (),
                probe_action=getattr(arguments, "probe_action", None),
            )
        print(f"benchmark repeated runtime preflight succeeded ({len(seeds or ())} seeds)")
        return 0
    if getattr(arguments, "runtime_preflight", False):
        _reject_preflight_only_conflicts(arguments, manifest=True, before_run=True)
        if source_controlled:
            preflight_report = validate_controlled_external_benchmark_runtime(
                spec,
                probe_action=getattr(arguments, "probe_action", None),
                runtime_requirements=runtime_requirements,
                source_checkout_paths=source_checkout_paths,
                source_checkout_requirements=source_checkout_requirements,
            )
        elif runtime_requirements is not None:
            preflight_report = validate_controlled_external_benchmark_runtime(
                spec,
                probe_action=getattr(arguments, "probe_action", None),
                runtime_requirements=runtime_requirements,
            )
        else:
            preflight_report = validate_external_benchmark_runtime(
                spec,
                probe_action=getattr(arguments, "probe_action", None),
            )
        mode = "step" if preflight_report.step_result is not None else "reset"
        print(f"benchmark runtime preflight succeeded ({mode} probe)")
        return 0

    probe_action = getattr(arguments, "probe_action", None)
    if (
        probe_action is not None
        and not getattr(arguments, "preflight_before_run", False)
        and runtime_requirements is None
        and not source_controlled
    ):
        raise ValueError("--probe-action requires a runtime preflight or --preflight-before-run")

    overwrite = getattr(arguments, "overwrite", False)
    output_path = _prepare_output_path(arguments.output, overwrite=overwrite)
    manifest_path = getattr(arguments, "manifest", None)
    selected_manifest_path = _prepare_manifest_path(
        output_path,
        manifest_path,
        overwrite=overwrite,
    )
    observability_path = _prepare_optional_artifact_path(
        getattr(arguments, "observability_output", None),
        artifact_name="observability snapshot",
        overwrite=overwrite,
        reserved_paths=(output_path, selected_manifest_path),
    )
    distribution_config = build_benchmark_distribution_config(
        output_path=getattr(arguments, "distribution_output", None),
        episode_duration_buckets=getattr(arguments, "episode_duration_buckets", None),
    )
    if distribution_config is not None:
        distribution_path = _prepare_optional_artifact_path(
            distribution_config.output_path,
            artifact_name="distribution observability sidecar",
            overwrite=overwrite,
            reserved_paths=(output_path, selected_manifest_path, observability_path),
        )
        if distribution_path is None:
            raise AssertionError("validated distribution configuration must have an output path")
        distribution_config = BenchmarkDistributionConfig(
            output_path=distribution_path,
            episode_duration_upper_bounds=distribution_config.episode_duration_upper_bounds,
        )
    observability_session = BenchmarkObservabilitySession(
        observability_output_path=observability_path,
        distribution_config=distribution_config,
    )
    seeds = _parse_seeds(getattr(arguments, "seeds", None))
    observation_collector = observability_session.create_collector()
    benchmark_runner = (
        BenchmarkSuiteRunner(observation_collector=observation_collector)
        if observation_collector is not None
        else None
    )
    controlled_execution = runtime_requirements is not None or source_controlled

    if seeds is None:
        if controlled_execution:
            controlled_result = run_controlled_external_benchmark(
                spec,
                probe_action=probe_action,
                runner=benchmark_runner,
                runtime_requirements=runtime_requirements,
                source_checkout_paths=source_checkout_paths,
                source_checkout_requirements=source_checkout_requirements,
            )

            def report_writer(temporary_path: Path) -> object:
                return save_controlled_benchmark_result(
                    controlled_result,
                    temporary_path,
                    runtime_requirements=runtime_requirements,
                    source_checkout_requirements=source_checkout_requirements,
                )

        else:
            if getattr(arguments, "preflight_before_run", False):
                validate_external_benchmark_runtime(
                    spec,
                    probe_action=probe_action,
                )
            report = run_external_benchmark(spec, runner=benchmark_runner)
            runtime_provenance = collect_runtime_provenance(environment=os.environ).to_dict()

            def report_writer(temporary_path: Path) -> object:
                return save_benchmark_report(
                    report,
                    temporary_path,
                    runtime_provenance=runtime_provenance,
                )

    else:
        if controlled_execution:
            controlled_result_repeated = run_controlled_repeated_external_benchmarks(
                spec,
                seeds,
                probe_action=probe_action,
                runner=benchmark_runner,
                runtime_requirements=runtime_requirements,
                source_checkout_paths=source_checkout_paths,
                source_checkout_requirements=source_checkout_requirements,
            )
            statistics = summarize_benchmark_reports(controlled_result_repeated.reports).to_dict()

            def report_writer(temporary_path: Path) -> object:
                return save_controlled_repeated_benchmark_result(
                    controlled_result_repeated,
                    temporary_path,
                    runtime_requirements=runtime_requirements,
                    source_checkout_requirements=source_checkout_requirements,
                    statistics=statistics,
                )

        else:
            if getattr(arguments, "preflight_before_run", False):
                reports = run_repeated_external_benchmarks_with_preflight(
                    spec,
                    seeds,
                    probe_action=probe_action,
                    runner=benchmark_runner,
                )
            else:
                reports = run_repeated_external_benchmarks(
                    spec,
                    seeds,
                    runner=benchmark_runner,
                )
            statistics = summarize_benchmark_reports(reports).to_dict()
            runtime_provenance = collect_runtime_provenance(environment=os.environ).to_dict()

            def report_writer(temporary_path: Path) -> object:
                return save_repeated_benchmark_reports(
                    reports,
                    temporary_path,
                    runtime_provenance=runtime_provenance,
                    statistics=statistics,
                )

    observability_snapshots = observability_session.freeze(observation_collector)
    distribution_path = (
        distribution_config.output_path if distribution_config is not None else None
    )
    output_path = _persist_benchmark_bundle(
        output_path,
        overwrite=overwrite,
        writer=report_writer,
        manifest_path=selected_manifest_path,
        observability_path=observability_path,
        observation_snapshot=observability_snapshots.observation_snapshot,
        distribution_path=distribution_path,
        distribution_snapshot=observability_snapshots.distribution_snapshot,
    )

    if selected_manifest_path is not None:
        print(f"saved benchmark artifact manifest: {selected_manifest_path}")
    if observability_path is not None:
        print(f"saved benchmark observability snapshot: {observability_path}")
    if distribution_path is not None:
        print(f"saved benchmark duration distribution sidecar: {distribution_path}")
    print(f"saved benchmark report: {output_path}")
    return 0


def _build_runtime_requirements(arguments: argparse.Namespace) -> RuntimeRequirements | None:
    """Build fail-closed runtime requirements from optional CLI arguments."""

    expected_revision = getattr(arguments, "require_code_revision", None)
    require_clean_working_tree = getattr(arguments, "require_clean_working_tree", False)
    dependency_values = getattr(arguments, "require_dependency_version", None) or ()
    dependency_versions = _parse_dependency_requirements(dependency_values)
    if expected_revision is None and not require_clean_working_tree and not dependency_versions:
        return None
    return RuntimeRequirements(
        expected_code_revision=expected_revision,
        require_clean_working_tree=require_clean_working_tree,
        dependency_versions=dependency_versions,
    )


def _build_source_checkout_contract(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Path] | None, dict[str, SourceCheckoutRequirement] | None]:
    """Build a complete named source-checkout admission contract from CLI arguments."""

    path_values = getattr(arguments, "source_checkout", None) or ()
    revision_values = getattr(arguments, "require_source_revision", None) or ()
    allow_dirty_values = getattr(arguments, "allow_dirty_source_checkout", None) or ()
    if not path_values and not revision_values and not allow_dirty_values:
        return None, None

    checkout_paths = {
        name: Path(path)
        for name, path in _parse_named_values(path_values, option="--source-checkout").items()
    }
    revisions = _parse_named_values(revision_values, option="--require-source-revision")
    allow_dirty = _parse_unique_names(
        allow_dirty_values,
        option="--allow-dirty-source-checkout",
    )

    path_names = {name.casefold(): name for name in checkout_paths}
    revision_names = {name.casefold(): name for name in revisions}
    if set(path_names) != set(revision_names):
        raise ValueError(
            "--source-checkout and --require-source-revision must declare the same names"
        )
    unknown_dirty_names = allow_dirty - set(path_names)
    if unknown_dirty_names:
        names = ", ".join(sorted(unknown_dirty_names))
        raise ValueError(f"--allow-dirty-source-checkout names must be declared sources: {names}")

    requirements: dict[str, SourceCheckoutRequirement] = {}
    for normalized_name in sorted(path_names):
        display_name = path_names[normalized_name]
        revision_name = revision_names[normalized_name]
        requirements[display_name] = SourceCheckoutRequirement(
            expected_revision=revisions[revision_name],
            require_clean_working_tree=normalized_name not in allow_dirty,
        )
    return checkout_paths, requirements


def _parse_named_values(values: Sequence[str], *, option: str) -> dict[str, str]:
    """Parse repeatable ``NAME=VALUE`` arguments with case-insensitive unique names."""

    parsed: dict[str, str] = {}
    normalized_names: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{option} values must be strings")
        name, separator, assigned_value = value.partition("=")
        name = name.strip()
        assigned_value = assigned_value.strip()
        if not separator or not name or not assigned_value:
            raise ValueError(f"{option} must use NAME=VALUE")
        normalized_name = name.casefold()
        if normalized_name in normalized_names:
            raise ValueError(f"{option} names must be unique ignoring case")
        normalized_names.add(normalized_name)
        parsed[name] = assigned_value
    return parsed


def _parse_unique_names(values: Sequence[str], *, option: str) -> set[str]:
    """Normalize repeatable source names while rejecting duplicates."""

    parsed: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{option} values must be strings")
        name = value.strip()
        if not name:
            raise ValueError(f"{option} names must not be empty")
        normalized_name = name.casefold()
        if normalized_name in parsed:
            raise ValueError(f"{option} names must be unique ignoring case")
        parsed.add(normalized_name)
    return parsed


def _parse_dependency_requirements(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated exact dependency pins in ``PACKAGE==VERSION`` form."""

    parsed: dict[str, str] = {}
    normalized_names: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("--require-dependency-version values must be strings")
        package_name, separator, package_version = value.partition("==")
        package_name = package_name.strip()
        package_version = package_version.strip()
        if not separator or not package_name or not package_version:
            raise ValueError("--require-dependency-version must use PACKAGE==VERSION")
        normalized_name = package_name.lower()
        if normalized_name in normalized_names:
            raise ValueError(
                "--require-dependency-version package names must be unique ignoring case"
            )
        normalized_names.add(normalized_name)
        parsed[package_name] = package_version
    return parsed


def _persist_benchmark_bundle(
    output_path: Path,
    *,
    overwrite: bool,
    writer: Callable[[Path], object],
    manifest_path: Path | None = None,
    observability_path: Path | None = None,
    observation_snapshot: ObservationSnapshot | None = None,
    distribution_path: Path | None = None,
    distribution_snapshot: DistributionObservationSnapshot | None = None,
) -> Path:
    """Publish one benchmark artifact bundle through the shared transactional path."""

    return persist_benchmark_observability_bundle(
        output_path,
        overwrite=overwrite,
        report_writer=writer,
        manifest_path=manifest_path,
        observability_path=observability_path,
        observation_snapshot=observation_snapshot,
        distribution_path=distribution_path,
        distribution_snapshot=distribution_snapshot,
    )


def _persist_benchmark_report(
    output_path: Path,
    *,
    overwrite: bool,
    writer: Callable[[Path], object],
) -> Path:
    """Publish a fully written benchmark report without a check-then-replace race."""

    return _persist_benchmark_bundle(
        output_path,
        overwrite=overwrite,
        writer=writer,
    )


def _prepare_output_path(path: Path, *, overwrite: bool) -> Path:
    """Reject accidental artifact replacement unless explicitly requested."""

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"benchmark artifact already exists: {path}; pass --overwrite to replace it"
        )
    return path


def _prepare_optional_artifact_path(
    path: Path | None,
    *,
    artifact_name: str,
    overwrite: bool,
    reserved_paths: tuple[Path | None, ...] = (),
) -> Path | None:
    """Validate an optional artifact destination before measured execution."""

    if path is None:
        return None
    resolved_path = path.resolve()
    for reserved_path in reserved_paths:
        if reserved_path is not None and resolved_path == reserved_path.resolve():
            raise ValueError(f"{artifact_name} must use a different path from another artifact")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{artifact_name} already exists: {path}; pass --overwrite to replace it"
        )
    return path


def _prepare_manifest_path(
    output_path: Path,
    manifest_path: Path | None,
    *,
    overwrite: bool,
) -> Path | None:
    """Validate the optional manifest destination before measured execution."""

    if manifest_path is None:
        return None
    if manifest_path.resolve() == output_path.resolve():
        raise ValueError("--manifest must use a different path from --output")
    return _prepare_output_path(manifest_path, overwrite=overwrite)


def _reject_preflight_only_conflicts(
    arguments: argparse.Namespace,
    *,
    manifest: bool,
    before_run: bool,
) -> None:
    """Reject options that only make sense for measured execution."""

    if manifest and getattr(arguments, "manifest", None) is not None:
        raise ValueError("--manifest requires a measured benchmark run")
    if getattr(arguments, "observability_output", None) is not None:
        raise ValueError("--observability-output requires a measured benchmark run")
    if getattr(arguments, "distribution_output", None) is not None:
        raise ValueError("--distribution-output requires a measured benchmark run")
    if getattr(arguments, "episode_duration_buckets", None) is not None:
        raise ValueError("--episode-duration-buckets requires a measured benchmark run")
    if before_run and getattr(arguments, "preflight_before_run", False):
        raise ValueError("--preflight-before-run requires a measured benchmark run")


def _parse_seeds(value: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated seed list, rejecting malformed or duplicate values."""

    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise ValueError("--seeds must contain comma-separated integers")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--seeds must contain comma-separated integers") from exc
    return validate_seed_sequence(seeds)


if __name__ == "__main__":
    raise SystemExit(main())
