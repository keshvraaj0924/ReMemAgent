"""Command-line entry point for paired external benchmark experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.benchmark_report import save_paired_benchmark_result
from experiments.external_benchmark import ExternalBenchmarkSpec, validate_seed_sequence
from experiments.paired_benchmark import run_paired_external_benchmarks_with_preflight
from experiments.runtime_provenance import collect_runtime_provenance

DEFAULT_OUTPUT_PATH = Path("artifacts/paired-benchmark.json")


def parse_args() -> argparse.Namespace:
    """Parse arguments for a paired baseline-versus-treatment experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated independent integer seeds")
    parser.add_argument("--environment-factory", required=True)
    parser.add_argument("--success-evaluator", required=True)
    parser.add_argument("--transfer-success-evaluator")
    _add_policy_arguments(parser, "baseline")
    _add_policy_arguments(parser, "treatment")
    parser.add_argument("--minimum-trust", type=float, default=0.0)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--treatment-label", default="treatment")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--probe-action")
    return parser.parse_args()


def main() -> int:
    """Execute a paired experiment and persist its measured reports."""

    arguments = parse_args()
    try:
        seeds = _parse_seeds(arguments.seeds)
        baseline_spec = _build_spec(
            arguments,
            policy_factory=arguments.baseline_policy_factory,
            action_policy_factory=arguments.baseline_action_policy_factory,
        )
        treatment_spec = _build_spec(
            arguments,
            policy_factory=arguments.treatment_policy_factory,
            action_policy_factory=arguments.treatment_action_policy_factory,
        )
        result = run_paired_external_benchmarks_with_preflight(
            baseline_spec,
            treatment_spec,
            seeds,
            baseline_label=arguments.baseline_label,
            treatment_label=arguments.treatment_label,
            probe_action=arguments.probe_action,
        )
        runtime_provenance = collect_runtime_provenance(environment=os.environ).to_dict()
        output_path = save_paired_benchmark_result(
            result.baseline_reports,
            result.treatment_reports,
            result.comparison,
            arguments.output,
            runtime_provenance=runtime_provenance,
        )
        if arguments.manifest is not None:
            manifest_path = save_benchmark_artifact_manifest(output_path, arguments.manifest)
            print(f"saved paired benchmark artifact manifest: {manifest_path}")
        print(f"saved paired benchmark report: {output_path}")
        return 0
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


def _add_policy_arguments(parser: argparse.ArgumentParser, condition: str) -> None:
    """Add exactly-one policy factory arguments for a benchmark condition."""

    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(f"--{condition}-policy-factory", dest=f"{condition}_policy_factory")
    policy_group.add_argument(
        f"--{condition}-action-policy-factory",
        dest=f"{condition}_action_policy_factory",
    )


def _build_spec(
    arguments: argparse.Namespace,
    *,
    policy_factory: str | None,
    action_policy_factory: str | None,
) -> ExternalBenchmarkSpec:
    """Build one condition specification from shared CLI configuration."""

    return ExternalBenchmarkSpec(
        benchmark_name=arguments.benchmark,
        episode_count=arguments.episodes,
        max_steps=arguments.max_steps,
        environment_factory=arguments.environment_factory,
        policy_factory=policy_factory,
        action_policy_factory=action_policy_factory,
        success_evaluator=arguments.success_evaluator,
        transfer_success_evaluator=arguments.transfer_success_evaluator,
        minimum_trust=arguments.minimum_trust,
        seed=None,
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated independent seed sequence."""

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
