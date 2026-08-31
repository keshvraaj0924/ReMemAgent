"""Command-line entry point for externally supplied benchmark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.benchmark_report import save_benchmark_report
from experiments.external_benchmark import load_callable, load_typed_callable, run_external_benchmark
from remem.memory.attribution import TransferSuccessEvaluator
from remem.services import SuccessEvaluator

DEFAULT_OUTPUT_PATH = Path("artifacts/benchmark.json")


def parse_args() -> argparse.Namespace:
    """Parse arguments for a caller-owned benchmark experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--environment-factory", required=True)
    parser.add_argument("--policy-factory", required=True)
    parser.add_argument("--success-evaluator", required=True)
    parser.add_argument("--transfer-success-evaluator")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Load experiment callables, execute the measured suite, and save its report."""

    arguments = parse_args()
    environment_factory = load_callable(arguments.environment_factory)
    policy_factory = load_callable(arguments.policy_factory)
    success_evaluator = load_typed_callable(arguments.success_evaluator)
    transfer_success_evaluator: TransferSuccessEvaluator | None = None
    if arguments.transfer_success_evaluator:
        transfer_success_evaluator = load_typed_callable(arguments.transfer_success_evaluator)

    report = run_external_benchmark(
        benchmark_name=arguments.benchmark,
        episode_count=arguments.episodes,
        max_steps=arguments.max_steps,
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        success_evaluator=success_evaluator,
        transfer_success_evaluator=transfer_success_evaluator,
    )
    output_path = save_benchmark_report(report, arguments.output)
    print(f"saved benchmark report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
