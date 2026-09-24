from pathlib import Path

from experiments.synthetic_benchmark_evidence import load_verified_synthetic_benchmark_evidence
from experiments.synthetic_negative_transfer import main as run_benchmark_cli


CANONICAL_CASES = Path("experiments/data/synthetic_negative_transfer_cases.json")


def test_canonical_synthetic_evaluation_round_trips_verified_evidence(tmp_path) -> None:
    output_path = tmp_path / "synthetic-evidence.json"

    exit_code = run_benchmark_cli(
        [
            "--cases",
            str(CANONICAL_CASES),
            "--output",
            str(output_path),
            "--minimum-delta",
            "0.05",
        ]
    )

    assert exit_code == 0
    verified = load_verified_synthetic_benchmark_evidence(output_path)
    assert verified.result.total_cases == 8
    assert verified.result.memory_selected + verified.result.self_reasoning_selected == 8
    assert verified.result.negative_transfer_cases == 3
    assert verified.result.selected_negative_transfer_cases == 0
    assert verified.result.avoided_negative_transfer_cases == 3
    assert verified.result.routing_regret >= 0.0
