from remem.routing.counterfactual import CounterfactualRouter


def test_router_uses_memory_when_expected_benefit_is_positive() -> None:
    score, decision = CounterfactualRouter().route(
        evaluate_with_memory=lambda: 0.9,
        evaluate_without_memory=lambda: 0.6,
    )

    assert score.delta == 0.30000000000000004
    assert decision.route == "memory"
    assert decision.expected_delta > 0


def test_router_rejects_memory_when_it_causes_negative_transfer() -> None:
    score, decision = CounterfactualRouter().route(
        evaluate_with_memory=lambda: 0.4,
        evaluate_without_memory=lambda: 0.8,
    )

    assert score.delta < 0
    assert decision.route == "self_reasoning"


def test_router_supports_explicit_margin() -> None:
    _, decision = CounterfactualRouter(minimum_delta=0.2).route(
        evaluate_with_memory=lambda: 0.7,
        evaluate_without_memory=lambda: 0.6,
    )

    assert decision.route == "self_reasoning"
