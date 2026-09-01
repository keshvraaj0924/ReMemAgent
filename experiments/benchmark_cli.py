"""Command-line entry point for externally supplied benchmark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.benchmark_report import save_benchmark_report
from experiments.external_benchmark import ExternalBenchmarkSpec, run_external_benchmark
from experiments.runtime_provenance import collect_runtime_provenance

DEFAULT_OUTPUT_PATH = Path("artifacts/benchmark.json")


def parse_args() -> argparse.Namespace:
    """Parse arguments for a caller-owned benchmark experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--environment-factory", required=True)
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--policy-factory")
    policy_group.add_argument("--action-policy-factory")
    parser.add_argument("--minimum-trust", type=float, default=0.0)
    parser.add_argument("--success-evaluator", required=True)
    parser.add_argument("--transfer-success-evaluator")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Execute the measured suite and save its JSON report."""

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
    report = run_external_benchmark(spec)
    runtime_provenance = collect_runtime_provenance().to_dict()
    output_path = save_benchmark_report(
        report,
        arguments.output,
        runtime_provenance=runtime_provenance,
    )
    print(f"saved benchmark report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
