"""Regression tests for the supported training integration surface."""

import remem.training as training


def test_training_package_exports_validated_integration_contracts() -> None:
    expected_exports = {
        "GrpoBatchMetrics",
        "GrpoRewardBreakdown",
        "GrpoRewardConfig",
        "GrpoTrajectory",
        "VerlBatchRewardRecord",
        "VerlRewardAdapter",
        "VerlRewardFields",
        "VerlRewardRecord",
        "build_verl_batch_reward_record",
        "compute_grpo_reward",
        "compute_grpo_reward_breakdown",
        "compute_score",
        "compute_score_batched",
        "summarize_grpo_breakdowns",
    }

    assert set(training.__all__) == expected_exports
    for export_name in expected_exports:
        assert getattr(training, export_name) is not None


def test_public_verl_reward_entrypoint_uses_canonical_reward_semantics() -> None:
    reward = training.compute_score(
        data_source="remem/synthetic",
        solution_str="ignored by transfer attribution",
        ground_truth="ignored by transfer attribution",
        extra_info={
            "task_reward": 1.0,
            "memory_used": True,
            "counterfactual_delta": 0.4,
        },
    )

    assert reward == 1.19
