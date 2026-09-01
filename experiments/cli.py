"""Command-line entry point for reproducible synthetic ablation experiments."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from experiments.runner import ExperimentConfig, run_reproducible_ablation, save_report
from experiments.synthetic_negative_transfer import BenchmarkCase

DEFAULT_OUTPUT_PATH = Path("artifacts/ablation.json")
DEFAULT_CASE_COUNT = 20


def build_synthetic_cases(rng: random.Random, case_count: int) -> list[BenchmarkCase]:
    """Generate deterministic matched cases for a local research smoke experiment."""

    if case_count <= 0:
        raise ValueError("case_count must be positive")

    cases: list[BenchmarkCase] = []
    for index in range(case_count):
        memory_utility = rng.uniform(0.2, 0.9)
        self_reasoning_utility = rng.uniform(0.2, 0.9)
        cases.append(
            BenchmarkCase(
                case_id=f"synthetic_{index:04d}",
                utility_with_memory=memory_utility,
                utility_without_memory=self_reasoning_utility,
            )
        )
    return cases


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the reproducible experiment runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-delta", type=float, default=0.05)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Run the configured synthetic ablation and persist its report."""

    arguments = parse_args()
    config = ExperimentConfig(seed=arguments.seed, minimum_delta=arguments.minimum_delta)
    report = run_reproducible_ablation(
        lambda rng: build_synthetic_cases(rng, arguments.cases),
        config,
    )
    output_path = save_report(report, arguments.output)
    print(f"saved experiment report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
