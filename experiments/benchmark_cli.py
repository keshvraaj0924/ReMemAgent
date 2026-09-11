"""Command-line entry point for caller-owned external benchmark experiments."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from experiments.artifact_bundle import (
    PreparedArtifact,
    create_private_artifact_path,
    publish_artifact_bundle,
)
from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.benchmark_report import save_benchmark_report, save_repeated_benchmark_reports
from experiments.benchmark_statistics import summarize_benchmark_reports
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
    validate_repeated_external_benchmark_runtime,
)
from experiments.runtime_provenance import collect_runtime_provenance
from remem.benchmark import BenchmarkSuiteRunner
from remem.observability import (
    ObservationCollector,
    ObservationSnapshot,
    write_observation_snapshot,
)

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
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing benchmark report, manifest, or observability snapshot",
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
    if getattr(arguments, "preflight", False):
        _reject_preflight_only_conflicts(arguments, manifest=True, before_run=True)
        validate_external_benchmark(spec)
        print("benchmark callable preflight succeeded")
        return 0
    if getattr(arguments, "repeated_runtime_preflight", False):
        _reject_preflight_only_conflicts(arguments, manifest=True, before_run=True)
        if getattr(arguments, "seeds", None) is None:
            raise ValueError("--repeated-runtime-preflight requires --seeds")
        seeds = _parse_seeds(arguments.seeds)
        validate_repeated_external_benchmark_runtime(
            spec,
            seeds or (),
            probe_action=getattr(arguments, "probe_action", None),
        )
        print(f"benchmark repeated runtime preflight succeeded ({len(seeds or ())} seeds)")
        return 0
    if getattr(arguments, "runtime_preflight", False):
        _reject_preflight_only_conflicts(arguments, manifest=True, before_run=True)
        preflight_report = validate_external_benchmark_runtime(
            spec,
            probe_action=getattr(arguments, "probe_action", None),
        )
        mode = "step" if preflight_report.step_result is not None else "reset"
        print(f"benchmark runtime preflight succeeded ({mode} probe)")
        return 0

    probe_action = getattr(arguments, "probe_action", None)
    if probe_action is not None and not getattr(arguments, "preflight_before_run", False):
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
    runtime_provenance = collect_runtime_provenance(environment=os.environ).to_dict()
    seeds = _parse_seeds(getattr(arguments, "seeds", None))
    observation_collector = ObservationCollector() if observability_path is not None else None
    benchmark_runner = (
        BenchmarkSuiteRunner(observation_collector=observation_collector)
        if observation_collector is not None
        else None
    )
    if seeds is None:
        if getattr(arguments, "preflight_before_run", False):
            validate_external_benchmark_runtime(
                spec,
                probe_action=probe_action,
            )
        report = run_external_benchmark(spec, runner=benchmark_runner)

        def report_writer(temporary_path: Path) -> object:
            return save_benchmark_report(
                report,
                temporary_path,
                runtime_provenance=runtime_provenance,
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

        def report_writer(temporary_path: Path) -> object:
            return save_repeated_benchmark_reports(
                reports,
                temporary_path,
                runtime_provenance=runtime_provenance,
                statistics=statistics,
            )

    observation_snapshot = (
        observation_collector.snapshot() if observation_collector is not None else None
    )
    output_path = _persist_benchmark_bundle(
        output_path,
        overwrite=overwrite,
        writer=report_writer,
        manifest_path=selected_manifest_path,
        observability_path=observability_path,
        observation_snapshot=observation_snapshot,
    )

    if selected_manifest_path is not None:
        print(f"saved benchmark artifact manifest: {selected_manifest_path}")
    if observability_path is not None:
        print(f"saved benchmark observability snapshot: {observability_path}")
    print(f"saved benchmark report: {output_path}")
    return 0


def _persist_benchmark_bundle(
    output_path: Path,
    *,
    overwrite: bool,
    writer: Callable[[Path], object],
    manifest_path: Path | None = None,
    observability_path: Path | None = None,
    observation_snapshot: ObservationSnapshot | None = None,
) -> Path:
    """Stage and publish one benchmark artifact bundle with rollback semantics."""

    if (observability_path is None) != (observation_snapshot is None):
        raise ValueError(
            "observability_path and observation_snapshot must either both be provided or both omitted"
        )

    prepared_artifacts: list[PreparedArtifact] = []
    report_temporary_path = create_private_artifact_path(output_path)
    report_artifact = PreparedArtifact(report_temporary_path, output_path)
    prepared_artifacts.append(report_artifact)
    try:
        writer(report_temporary_path)

        ordered_artifacts: list[PreparedArtifact] = []
        if observability_path is not None and observation_snapshot is not None:
            observability_temporary_path = create_private_artifact_path(observability_path)
            observability_artifact = PreparedArtifact(
                observability_temporary_path,
                observability_path,
            )
            prepared_artifacts.append(observability_artifact)
            write_observation_snapshot(
                observability_temporary_path,
                observation_snapshot,
                overwrite=True,
            )
            ordered_artifacts.append(observability_artifact)

        if manifest_path is not None:
            manifest_temporary_path = create_private_artifact_path(manifest_path)
            manifest_artifact = PreparedArtifact(manifest_temporary_path, manifest_path)
            prepared_artifacts.append(manifest_artifact)
            save_benchmark_artifact_manifest(
                report_temporary_path,
                manifest_temporary_path,
                overwrite=True,
            )
            ordered_artifacts.append(manifest_artifact)

        ordered_artifacts.append(report_artifact)
        publish_artifact_bundle(tuple(ordered_artifacts), overwrite=overwrite)
    except BaseException:
        for artifact in prepared_artifacts:
            artifact.temporary_path.unlink(missing_ok=True)
        raise
    return output_path


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
