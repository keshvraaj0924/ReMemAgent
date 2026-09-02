"""Command-line entry point for externally supplied benchmark experiments."""

from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--probe-action",
        help="Optional concrete action used by --runtime-preflight for one step probe",
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
        policy_factory=arguments.policy_factory,
        action_policy_factory=arguments.action_policy_factory,
        minimum_trust=arguments.minimum_trust,
        success_evaluator=arguments.success_evaluator,
        transfer_success_evaluator=arguments.transfer_success_evaluator,
        seed=arguments.seed,
    )
    if getattr(arguments, "preflight", False):
        if getattr(arguments, "probe_action", None) is not None:
            raise ValueError("--probe-action requires --runtime-preflight")
        if getattr(arguments, "manifest", None) is not None:
            raise ValueError("--manifest requires a measured benchmark run")
        validate_external_benchmark(spec)
        print("benchmark callable preflight succeeded")
        return 0
    if getattr(arguments, "runtime_preflight", False):
        if getattr(arguments, "manifest", None) is not None:
            raise ValueError("--manifest requires a measured benchmark run")
        report = validate_external_benchmark_runtime(
            spec,
            probe_action=getattr(arguments, "probe_action", None),
        )
        mode = "step" if report.step_result is not None else "reset"
        print(f"benchmark runtime preflight succeeded ({mode} probe)")
        return 0
    if getattr(arguments, "probe_action", None) is not None:
        raise ValueError("--probe-action requires --runtime-preflight")

    runtime_provenance = collect_runtime_provenance().to_dict()

    seeds = _parse_seeds(getattr(arguments, "seeds", None))
    if seeds is None:
        report = run_external_benchmark(spec)
        output_path = save_benchmark_report(
            report,
            arguments.output,
            runtime_provenance=runtime_provenance,
        )
    else:
        reports = run_repeated_external_benchmarks(spec, seeds)
        statistics = summarize_benchmark_reports(reports).to_dict()
        output_path = save_repeated_benchmark_reports(
            reports,
            arguments.output,
            runtime_provenance=runtime_provenance,
            statistics=statistics,
        )

    manifest_path = getattr(arguments, "manifest", None)
    if manifest_path is not None:
        manifest_output = save_benchmark_artifact_manifest(output_path, manifest_path)
        print(f"saved benchmark artifact manifest: {manifest_output}")
    print(f"saved benchmark report: {output_path}")
    return 0


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
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integers")
    return seeds


if __name__ == "__main__":
    raise SystemExit(main())
