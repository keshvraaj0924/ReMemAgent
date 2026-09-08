"""CLI for checking optional external benchmark dependencies."""

from __future__ import annotations

import argparse

from remem.environments.dependencies import check_benchmark_dependencies

SUPPORTED_BENCHMARKS = ("alfworld", "webshop")


def parse_args() -> argparse.Namespace:
    """Parse environment dependency diagnostic arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        choices=SUPPORTED_BENCHMARKS,
        action="append",
        metavar="BENCHMARK",
        help="Require a benchmark dependency to be available; repeat for multiple benchmarks",
    )
    return parser.parse_args()


def main() -> int:
    """Print dependency status and fail when explicitly required dependencies are missing."""

    arguments = parse_args()
    statuses = {status.package_name: status for status in check_benchmark_dependencies()}
    required = tuple(arguments.require or SUPPORTED_BENCHMARKS)

    for benchmark_name in SUPPORTED_BENCHMARKS:
        status = statuses[benchmark_name]
        state = "available" if status.available else "missing"
        print(f"{benchmark_name}: {state} (import={status.import_name})")

    missing_required = tuple(name for name in required if not statuses[name].available)
    if missing_required:
        print("missing required benchmark dependencies: " + ", ".join(missing_required))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
