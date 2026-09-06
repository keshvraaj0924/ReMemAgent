"""Command-line entry point for caller-owned external benchmark experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing benchmark report or integrity manifest",
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
        report = validate_external_benchmark_runtime(
            spec,
            probe_action=getattr(arguments, "probe_action", None),
        )
        mode = "step" if report.step_result is not None else "reset"
        print(f"benchmark runtime preflight succeeded ({mode} probe)")
        return 0

    probe_action = getattr(arguments, "probe_action", None)
    if probe_action is not None and not getattr(arguments, "preflight_before_run", False):
        raise ValueError("--probe-action requires a runtime preflight or --preflight-before-run")

    output_path = _prepare_output_path(
        arguments.output,
        overwrite=getattr(arguments, "overwrite", False),
    )
    manifest_path = getattr(arguments, "manifest", None)
    selected_manifest_path = _prepare_manifest_path(
        output_path,
        manifest_path,
        overwrite=getattr(arguments, "overwrite", False),
    )
    runtime_provenance = collect_runtime_provenance(environment=os.environ).to_dict()
    seeds = _parse_seeds(getattr(arguments, "seeds", None))
    if seeds is None:
        if getattr(arguments, "preflight_before_run", False):
            validate_external_benchmark_runtime(
                spec,
                probe_action=probe_action,
            )
        report = run_external_benchmark(spec)
        output_path = save_benchmark_report(
            report,
            output_path,
            runtime_provenance=runtime_provenance,
        )
    else:
        if getattr(arguments, "preflight_before_run", False):
            reports = run_repeated_external_benchmarks_with_preflight(
                spec,
                seeds,
                probe_action=probe_action,
            )
        else:
            reports = run_repeated_external_benchmarks(spec, seeds)
        statistics = summarize_benchmark_reports(reports).to_dict()
        output_path = save_repeated_benchmark_reports(
            reports,
            output_path,
            runtime_provenance=runtime_provenance,
            statistics=statistics,
        )

    if selected_manifest_path is not None:
        manifest_output = save_benchmark_artifact_manifest(output_path, selected_manifest_path)
        print(f"saved benchmark artifact manifest: {manifest_output}")
    print(f"saved benchmark report: {output_path}")
    return 0


def _prepare_output_path(path: Path, *, overwrite: bool) -> Path:
    """Reject accidental artifact replacement unless explicitly requested."""

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"benchmark artifact already exists: {path}; pass --overwrite to replace it"
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
