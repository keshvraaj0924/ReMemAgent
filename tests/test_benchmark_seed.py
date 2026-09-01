from remem.benchmark import BenchmarkSuiteRunner
from remem.environments.base import StepResult
from remem.execution import Policy
from remem.memory.store import MemoryStore


class SeedEnvironment:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def reset(self, **kwargs: object) -> str:
        return f"state-{self.seed}"

    def step(self, action: str) -> StepResult:
        return StepResult("done", float(self.seed), True, False)

    def close(self) -> None:
        return None


def test_seed_contract_is_forwarded_to_environment_and_policy_factories() -> None:
    environment_seeds: list[int] = []
    policy_seeds: list[int] = []

    def environment_factory(seed: int) -> SeedEnvironment:
        environment_seeds.append(seed)
        return SeedEnvironment(seed)

    def policy_factory(seed: int, store: MemoryStore) -> Policy:
        policy_seeds.append(seed)
        return lambda state: "act"

    report = BenchmarkSuiteRunner().run(
        benchmark_name="seeded-smoke",
        episode_count=3,
        max_steps=1,
        environment_factory=environment_factory,
        policy_factory=policy_factory,
        success_evaluator=lambda episode: True,
        seed=100,
    )

    assert environment_seeds == [100, 101, 102]
    assert policy_seeds == [100, 101, 102]
    assert [episode.episode_id for episode in report.episodes] == [
        "seeded-smoke:0",
        "seeded-smoke:1",
        "seeded-smoke:2",
    ]
    assert report.seed == 100


def test_unseeded_benchmark_preserves_episode_index_factory_contract() -> None:
    observed_seeds: list[int] = []

    report = BenchmarkSuiteRunner().run(
        benchmark_name="unseeded-smoke",
        episode_count=2,
        max_steps=1,
        environment_factory=lambda seed: (
            observed_seeds.append(seed) or SeedEnvironment(seed)
        ),
        policy_factory=lambda seed, store: (lambda state: "act"),
        success_evaluator=lambda episode: True,
    )

    assert observed_seeds == [0, 1]
    assert report.seed is None
