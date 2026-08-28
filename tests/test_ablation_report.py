from pathlib import Path

from experiments.ablation_report import results_to_records, write_results_json
from experiments.ablations import AblationResult, AblationStrategy


def _sample_result() -> AblationResult:
    return AblationResult(
        strategy=AblationStrategy.COUNTERFACTUAL,
        total_cases=2,
        selected_memory=1,
        mean_utility=0.85,
        negative_transfer_cases=1,
        selected_negative_transfer_cases=0,
        routing_regret=0.0,
    )


def test_results_to_records_uses_serializable_strategy_value() -> None:
    records = results_to_records([_sample_result()])

    assert records == [
        {
            "mean_utility": 0.85,
            "negative_transfer_cases": 1,
            "negative_transfer_rate": 0.0,
            "routing_regret": 0.0,
            "selected_memory": 1,
            "selected_negative_transfer_cases": 0,
            "strategy": "counterfactual",
            "total_cases": 2,
        }
    ]


def test_write_results_json_creates_parent_directory_and_stable_json(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "results.json"

    write_results_json([_sample_result()], output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == (
        '{\n'
        '  "results": [\n'
        '    {\n'
        '      "mean_utility": 0.85,\n'
        '      "negative_transfer_cases": 1,\n'
        '      "negative_transfer_rate": 0.0,\n'
        '      "routing_regret": 0.0,\n'
        '      "selected_memory": 1,\n'
        '      "selected_negative_transfer_cases": 0,\n'
        '      "strategy": "counterfactual",\n'
        '      "total_cases": 2\n'
        '    }\n'
        '  ]\n'
        '}\n'
    )
