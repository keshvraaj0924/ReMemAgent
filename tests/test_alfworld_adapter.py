import math

import pytest

from remem.environments.alfworld import AlfWorldAdapter


class FakeAlfWorld:
    def __init__(self, step_result, *, reset_result=None):
        self.step_result = step_result
        self.reset_result = (
            (["initial observation"], {"ignored": [1]}) if reset_result is None else reset_result
        )

    def reset(self):
        return self.reset_result

    def step(self, actions):
        assert actions == ["look"]
        return self.step_result

    def close(self):
        pass


def test_alfworld_adapter_normalizes_valid_five_value_step() -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld((["next"], [1.0], [True], [False], {"score": [1]})))

    assert adapter.reset() == "initial observation"
    result = adapter.step("look")

    assert result.observation == "next"
    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info == {"score": 1}


def test_alfworld_adapter_rejects_non_text_reset_observation() -> None:
    adapter = AlfWorldAdapter(
        FakeAlfWorld(
            (["next"], [0.0], [False], [False], {}),
            reset_result=([None], {}),
        )
    )

    with pytest.raises(TypeError, match="ALFWorld observation must be a string"):
        adapter.reset()


def test_alfworld_adapter_rejects_non_text_step_observation() -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld(([None], [0.0], [False], [False], {})))

    with pytest.raises(TypeError, match="ALFWorld observation must be a string"):
        adapter.step("look")


def test_alfworld_adapter_rejects_non_boolean_terminal_flags() -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld((["next"], [0.0], ["false"], [False], {})))

    with pytest.raises(TypeError, match="terminated flag must be a boolean"):
        adapter.step("look")


def test_alfworld_adapter_rejects_boolean_reward() -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld((["next"], [True], [False], [False], {})))

    with pytest.raises(TypeError, match="reward must be a finite numeric value"):
        adapter.step("look")


@pytest.mark.parametrize("reward", [math.nan, math.inf, -math.inf])
def test_alfworld_adapter_rejects_non_finite_reward(reward: float) -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld((["next"], [reward], [False], [False], {})))

    with pytest.raises(ValueError, match="reward must be finite"):
        adapter.step("look")


def test_alfworld_adapter_rejects_non_mapping_info() -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld((["next"], [0.0], [False], [False], None)))

    with pytest.raises(TypeError, match="ALFWorld info must be a mapping"):
        adapter.step("look")


def test_alfworld_adapter_rejects_non_string_info_keys() -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld((["next"], [0.0], [False], [False], {1: ["bad"]})))

    with pytest.raises(TypeError, match="ALFWorld info keys must be strings"):
        adapter.step("look")


def test_alfworld_adapter_supports_legacy_four_value_step() -> None:
    adapter = AlfWorldAdapter(FakeAlfWorld((["next"], [0.5], [True], {"done": [True]})))

    result = adapter.step("look")

    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 0.5
