from experiments.memory_transfer import record_transfer_outcomes
from experiments.synthetic_negative_transfer import BenchmarkCase, run_benchmark
from remem.memory.store import MemoryStore
from remem.memory.types import MemoryRecord
from remem.routing.counterfactual import CounterfactualRouter


def test_record_transfer_outcomes_updates_only_explicit_memory_transfers() -> None:
    store = MemoryStore(
        [
            MemoryRecord(memory_id="memory_a", state="state a"),
            MemoryRecord(memory_id="memory_b", state="state b"),
        ]
    )
    result = run_benchmark(
        [
            BenchmarkCase("success", 0.9, 0.7, memory_id="memory_a", transfer_success=True),
            BenchmarkCase("failure", 0.8, 0.7, memory_id="memory_b", transfer_success=False),
            BenchmarkCase("unmeasured", 0.9, 0.7, memory_id="memory_a"),
        ],
        CounterfactualRouter(minimum_delta=-0.5),
    )

    recorded_count = record_transfer_outcomes(result, store)

    assert recorded_count == 2
    memory_a = store.get("memory_a")
    memory_b = store.get("memory_b")
    assert memory_a is not None
    assert memory_b is not None
    assert memory_a.transfer_attempts == 1
    assert memory_a.transfer_successes == 1
    assert memory_b.transfer_attempts == 1
    assert memory_b.transfer_successes == 0


def test_record_transfer_outcomes_ignores_self_reasoning_and_unmeasured_cases() -> None:
    store = MemoryStore([MemoryRecord(memory_id="memory_a", state="state a")])
    result = run_benchmark(
        [
            BenchmarkCase("avoided", 0.4, 0.8, memory_id="memory_a", transfer_success=True),
            BenchmarkCase("unmeasured", 0.9, 0.7, memory_id="memory_a"),
        ],
        CounterfactualRouter(minimum_delta=0.05),
    )

    assert record_transfer_outcomes(result, store) == 0
    memory = store.get("memory_a")
    assert memory is not None
    assert memory.transfer_attempts == 0


def test_record_transfer_outcomes_rejects_unknown_memory_identity() -> None:
    store = MemoryStore()
    result = run_benchmark(
        [BenchmarkCase("unknown", 0.9, 0.7, memory_id="missing", transfer_success=True)],
        CounterfactualRouter(minimum_delta=-0.5),
    )

    try:
        record_transfer_outcomes(result, store)
    except KeyError as error:
        assert str(error) == "\"Memory 'missing' does not exist\""
    else:
        raise AssertionError("Expected unknown memory identity to be rejected")
