"""CLI bridge that binds paired measurement to persisted research admission evidence.

The primary paired CLI predates persisted readiness evidence and frozen experiment
plans. This bridge preserves its full argument contract while adding fail-closed
measurement admission for both artifacts. Bridge-only options are removed before
delegating to the existing parser, and controlled runner/persistence boundaries
are replaced only for that invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import experiments.paired_benchmark_cli as paired_cli
from experiments.evidence_bound_paired_benchmark import (
    run_evidence_bound_paired_external_benchmarks,
)
from experiments.preflight_evidence import (
    PREFLIGHT_EVIDENCE_PROVENANCE_KEY,
    verify_controlled_paired_preflight_evidence,
)
from experiments.research_experiment_plan import (
    JsonScalar,
    ResearchExperimentPlan,
    load_research_experiment_plan,
)

READINESS_EVIDENCE_OPTION = "--require-preflight-evidence"
EXPERIMENT_PLAN_OPTION = "--require-experiment-plan"
MODEL_IDENTITY_OPTION = "--model-identity"
EXPERIMENT_PARAMETER_OPTION = "--experiment-parameter"
EXPERIMENT_PLAN_PROVENANCE_KEY = "research_experiment_plan_sha256"
MODEL_IDENTITY_PROVENANCE_KEY = "research_model_identity"
EXPERIMENT_PARAMETERS_PROVENANCE_KEY = "research_experiment_parameters"
SOURCE_CHECKOUT_OPTION = "--source-checkout"
SOURCE_REVISION_OPTION = "--require-source-revision"


def main() -> int:
    """Run paired measurement with optional readiness and frozen-plan admission."""

    original_argv = list(sys.argv)
    try:
        evidence_path, argv_without_evidence = _extract_readiness_evidence_path(original_argv)
        plan_path, argv_without_plan = _extract_experiment_plan_path(argv_without_evidence)
        model_identity, argv_without_model = _extract_model_identity(argv_without_plan)
        experiment_parameters, delegated_argv = _extract_experiment_parameters(argv_without_model)
        if evidence_path is None and plan_path is None:
            _reject_unbound_model_configuration(model_identity, experiment_parameters)
            return paired_cli.main()
        if plan_path is not None and evidence_path is None:
            raise SystemExit(
                f"error: {EXPERIMENT_PLAN_OPTION} requires {READINESS_EVIDENCE_OPTION} "
                "for measured controlled execution"
            )
        if plan_path is None:
            _reject_unbound_model_configuration(model_identity, experiment_parameters)
        if evidence_path is None:  # pragma: no cover - guarded above
            raise RuntimeError("readiness evidence path unexpectedly missing")
        if _contains_option(delegated_argv, "--preflight-only"):
            raise SystemExit(
                f"error: {READINESS_EVIDENCE_OPTION} is for measured execution, not --preflight-only"
            )
        _require_controlled_source_contract(delegated_argv)
        readiness_evidence = _load_readiness_evidence(evidence_path)
        verify_controlled_paired_preflight_evidence(readiness_evidence)
        evidence_sha256 = _readiness_evidence_sha256(readiness_evidence)

        plan_sha256: str | None = None
        if plan_path is not None:
            plan = _load_experiment_plan(plan_path)
            arguments = _parse_delegated_arguments(delegated_argv)
            _validate_experiment_plan_contract(plan, arguments)
            _validate_experiment_model_contract(plan, model_identity, experiment_parameters)
            plan_sha256 = plan.sha256

        original_runner = paired_cli.run_controlled_paired_external_benchmarks
        original_saver = paired_cli.save_paired_execution_result

        def evidence_bound_runner(*args: Any, **kwargs: Any):
            return run_evidence_bound_paired_external_benchmarks(
                readiness_evidence,
                *args,
                **kwargs,
            )

        def evidence_bound_saver(*args: Any, **kwargs: Any):
            runtime_provenance = dict(kwargs.get("runtime_provenance") or {})
            _reserve_provenance_key(runtime_provenance, PREFLIGHT_EVIDENCE_PROVENANCE_KEY)
            runtime_provenance[PREFLIGHT_EVIDENCE_PROVENANCE_KEY] = evidence_sha256
            if plan_sha256 is not None:
                _reserve_provenance_key(runtime_provenance, EXPERIMENT_PLAN_PROVENANCE_KEY)
                runtime_provenance[EXPERIMENT_PLAN_PROVENANCE_KEY] = plan_sha256
                _reserve_provenance_key(runtime_provenance, MODEL_IDENTITY_PROVENANCE_KEY)
                _reserve_provenance_key(runtime_provenance, EXPERIMENT_PARAMETERS_PROVENANCE_KEY)
                runtime_provenance[MODEL_IDENTITY_PROVENANCE_KEY] = model_identity
                runtime_provenance[EXPERIMENT_PARAMETERS_PROVENANCE_KEY] = dict(
                    sorted(experiment_parameters.items())
                )
            kwargs["runtime_provenance"] = runtime_provenance
            return original_saver(*args, **kwargs)

        paired_cli.run_controlled_paired_external_benchmarks = evidence_bound_runner
        paired_cli.save_paired_execution_result = evidence_bound_saver
        sys.argv = delegated_argv
        try:
            return paired_cli.main()
        finally:
            paired_cli.run_controlled_paired_external_benchmarks = original_runner
            paired_cli.save_paired_execution_result = original_saver
    finally:
        sys.argv = original_argv


def _extract_readiness_evidence_path(argv: Sequence[str]) -> tuple[Path | None, list[str]]:
    """Remove the readiness-evidence option from argv and return its path exactly once."""

    return _extract_path_option(argv, READINESS_EVIDENCE_OPTION)


def _extract_experiment_plan_path(argv: Sequence[str]) -> tuple[Path | None, list[str]]:
    """Remove the frozen-plan option from argv and return its path exactly once."""

    return _extract_path_option(argv, EXPERIMENT_PLAN_OPTION)


def _extract_model_identity(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    """Remove one explicit model/checkpoint identity from bridge-owned arguments."""

    selected_identity: str | None = None
    delegated = [argv[0]] if argv else []
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == MODEL_IDENTITY_OPTION:
            if selected_identity is not None:
                raise SystemExit(f"error: {MODEL_IDENTITY_OPTION} may be specified only once")
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit(f"error: {MODEL_IDENTITY_OPTION} requires a value")
            selected_identity = _non_empty_cli_value(argv[index + 1], MODEL_IDENTITY_OPTION)
            index += 2
            continue
        prefix = f"{MODEL_IDENTITY_OPTION}="
        if argument.startswith(prefix):
            if selected_identity is not None:
                raise SystemExit(f"error: {MODEL_IDENTITY_OPTION} may be specified only once")
            selected_identity = _non_empty_cli_value(argument[len(prefix) :], MODEL_IDENTITY_OPTION)
            index += 1
            continue
        delegated.append(argument)
        index += 1
    return selected_identity, delegated


def _extract_experiment_parameters(
    argv: Sequence[str],
) -> tuple[dict[str, JsonScalar], list[str]]:
    """Remove repeatable ``NAME=JSON_SCALAR`` experiment parameters from argv."""

    parameters: dict[str, JsonScalar] = {}
    delegated = [argv[0]] if argv else []
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == EXPERIMENT_PARAMETER_OPTION:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit(f"error: {EXPERIMENT_PARAMETER_OPTION} requires NAME=JSON_SCALAR")
            _add_experiment_parameter(parameters, argv[index + 1])
            index += 2
            continue
        prefix = f"{EXPERIMENT_PARAMETER_OPTION}="
        if argument.startswith(prefix):
            _add_experiment_parameter(parameters, argument[len(prefix) :])
            index += 1
            continue
        delegated.append(argument)
        index += 1
    return parameters, delegated


def _add_experiment_parameter(parameters: dict[str, JsonScalar], declaration: str) -> None:
    """Parse and add one deterministic scalar parameter declaration."""

    name, separator, raw_value = declaration.partition("=")
    normalized_name = name.strip()
    if not separator or not normalized_name or not raw_value:
        raise SystemExit(f"error: {EXPERIMENT_PARAMETER_OPTION} requires NAME=JSON_SCALAR")
    if normalized_name in parameters:
        raise SystemExit(
            f"error: duplicate {EXPERIMENT_PARAMETER_OPTION} name: {normalized_name!r}"
        )
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"error: invalid JSON scalar for {EXPERIMENT_PARAMETER_OPTION} {normalized_name!r}: {exc}"
        ) from exc
    if isinstance(value, (dict, list)):
        raise SystemExit(
            f"error: {EXPERIMENT_PARAMETER_OPTION} {normalized_name!r} must be a JSON scalar"
        )
    parameters[normalized_name] = value


def _non_empty_cli_value(value: str, option: str) -> str:
    """Return a trimmed non-empty bridge-owned CLI value."""

    normalized = value.strip()
    if not normalized:
        raise SystemExit(f"error: {option} requires a non-empty value")
    return normalized


def _reject_unbound_model_configuration(
    model_identity: str | None,
    parameters: Mapping[str, JsonScalar],
) -> None:
    """Prevent bridge-owned model declarations from being silently ignored without a plan."""

    if model_identity is not None or parameters:
        raise SystemExit(
            f"error: {MODEL_IDENTITY_OPTION} and {EXPERIMENT_PARAMETER_OPTION} require "
            f"{EXPERIMENT_PLAN_OPTION}"
        )


def _extract_path_option(argv: Sequence[str], option: str) -> tuple[Path | None, list[str]]:
    """Remove one bridge-owned path option while rejecting ambiguous duplicates."""

    delegated = [argv[0]] if argv else []
    selected_path: Path | None = None
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == option:
            if selected_path is not None:
                raise SystemExit(f"error: {option} may be specified only once")
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit(f"error: {option} requires a path")
            selected_path = Path(argv[index + 1])
            index += 2
            continue
        prefix = f"{option}="
        if argument.startswith(prefix):
            if selected_path is not None:
                raise SystemExit(f"error: {option} may be specified only once")
            value = argument[len(prefix) :]
            if not value:
                raise SystemExit(f"error: {option} requires a path")
            selected_path = Path(value)
            index += 1
            continue
        delegated.append(argument)
        index += 1
    return selected_path, delegated


def _parse_delegated_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the exact delegated paired CLI configuration without mutating caller argv."""

    original_argv = list(sys.argv)
    try:
        sys.argv = list(argv)
        return paired_cli.parse_args()
    finally:
        sys.argv = original_argv


def _validate_experiment_plan_contract(
    plan: ResearchExperimentPlan,
    arguments: argparse.Namespace,
) -> None:
    """Fail closed when measured paired CLI configuration drifts from a frozen plan."""

    if not getattr(arguments, "strict_reproducibility", False):
        raise ValueError(f"{EXPERIMENT_PLAN_OPTION} requires --strict-reproducibility")

    observed_scalars = {
        "benchmark_name": arguments.benchmark,
        "episode_count": arguments.episodes,
        "max_steps": arguments.max_steps,
        "environment_factory": arguments.environment_factory,
        "success_evaluator": arguments.success_evaluator,
        "transfer_success_evaluator": arguments.transfer_success_evaluator,
        "baseline_policy_factory": arguments.baseline_policy_factory,
        "baseline_action_policy_factory": arguments.baseline_action_policy_factory,
        "treatment_policy_factory": arguments.treatment_policy_factory,
        "treatment_action_policy_factory": arguments.treatment_action_policy_factory,
        "minimum_trust": arguments.minimum_trust,
        "baseline_label": arguments.baseline_label,
        "treatment_label": arguments.treatment_label,
    }
    for field_name, observed in observed_scalars.items():
        expected = getattr(plan, field_name)
        if observed != expected:
            raise ValueError(
                f"research experiment plan mismatch for {field_name}: "
                f"expected {expected!r}, got {observed!r}"
            )

    seeds = paired_cli._parse_seeds(arguments.seeds)
    if seeds != plan.seeds:
        raise ValueError(
            f"research experiment plan mismatch for seeds: expected {plan.seeds!r}, got {seeds!r}"
        )

    runtime_requirements = paired_cli._build_runtime_requirements(arguments)
    if runtime_requirements is None:
        raise ValueError("research experiment plan requires explicit runtime requirements")
    if runtime_requirements.expected_code_revision != plan.remem_revision:
        raise ValueError(
            "research experiment plan ReMemAgent revision does not match --require-code-revision"
        )
    if not runtime_requirements.require_clean_working_tree:
        raise ValueError("research experiment plan requires --require-clean-working-tree")
    _require_normalized_mapping_match(
        expected=dict(plan.dependency_versions),
        observed=dict(runtime_requirements.dependency_versions),
        field_name="dependency_versions",
    )

    _, source_requirements = paired_cli._build_source_checkout_contract(arguments)
    if source_requirements is None:
        raise ValueError("research experiment plan requires declared source checkouts")
    if any(
        not requirement.require_clean_working_tree for requirement in source_requirements.values()
    ):
        raise ValueError("research experiment plan requires clean source checkouts")
    observed_source_revisions = {
        name: requirement.expected_revision for name, requirement in source_requirements.items()
    }
    _require_normalized_mapping_match(
        expected=dict(plan.source_revisions),
        observed=observed_source_revisions,
        field_name="source_revisions",
    )


def _validate_experiment_model_contract(
    plan: ResearchExperimentPlan,
    model_identity: str | None,
    parameters: Mapping[str, JsonScalar],
) -> None:
    """Require exact model/checkpoint and scalar parameter agreement with the frozen plan."""

    if model_identity != plan.model_identity:
        raise ValueError(
            "research experiment plan mismatch for model_identity: "
            f"expected {plan.model_identity!r}, got {model_identity!r}"
        )
    expected_parameters = dict(plan.parameters)
    if dict(parameters) != expected_parameters:
        raise ValueError(
            "research experiment plan mismatch for parameters: "
            f"expected {expected_parameters!r}, got {dict(parameters)!r}"
        )


def _require_normalized_mapping_match(
    *,
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    field_name: str,
) -> None:
    """Compare named research pins case-insensitively while retaining exact values."""

    normalized_expected = {name.strip().casefold(): value for name, value in expected.items()}
    normalized_observed = {name.strip().casefold(): value for name, value in observed.items()}
    if normalized_expected != normalized_observed:
        raise ValueError(
            f"research experiment plan mismatch for {field_name}: "
            f"expected {dict(expected)!r}, got {dict(observed)!r}"
        )


def _contains_option(argv: Sequence[str], option: str) -> bool:
    """Return whether argv contains an option in separated or ``--name=value`` form."""

    prefix = f"{option}="
    return any(argument == option or argument.startswith(prefix) for argument in argv[1:])


def _require_controlled_source_contract(argv: Sequence[str]) -> None:
    """Fail closed unless evidence-bound measurement will use controlled admission."""

    if not _contains_option(argv, SOURCE_CHECKOUT_OPTION) or not _contains_option(
        argv, SOURCE_REVISION_OPTION
    ):
        raise SystemExit(
            "error: --require-preflight-evidence requires declared source checkouts "
            "and exact source revisions"
        )


def _load_readiness_evidence(path: Path) -> Mapping[str, Any]:
    """Load a readiness evidence JSON object without collecting mutable runtime state."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"error: unable to read readiness evidence {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid readiness evidence JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit("error: readiness evidence root must be a JSON object")
    return payload


def _load_experiment_plan(path: Path) -> ResearchExperimentPlan:
    """Load one frozen plan without collecting or inferring mutable runtime state."""

    try:
        return load_research_experiment_plan(path)
    except OSError as exc:
        raise SystemExit(f"error: unable to read research experiment plan {path}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"error: invalid research experiment plan {path}: {exc}") from exc


def _readiness_evidence_sha256(payload: Mapping[str, Any]) -> str:
    """Return the already-verified readiness digest used to bind persisted measurement."""

    evidence_sha256 = payload.get("evidence_sha256")
    if not isinstance(evidence_sha256, str):
        raise ValueError("verified preflight evidence must contain evidence_sha256")
    return evidence_sha256


def _reserve_provenance_key(runtime_provenance: Mapping[str, Any], key: str) -> None:
    """Reject caller-owned values for provenance fields reserved by this bridge."""

    if key in runtime_provenance:
        raise ValueError(f"runtime_provenance reserves {key!r} for evidence-bound execution")


__all__ = [
    "EXPERIMENT_PARAMETER_OPTION",
    "EXPERIMENT_PARAMETERS_PROVENANCE_KEY",
    "EXPERIMENT_PLAN_OPTION",
    "EXPERIMENT_PLAN_PROVENANCE_KEY",
    "MODEL_IDENTITY_OPTION",
    "MODEL_IDENTITY_PROVENANCE_KEY",
    "PREFLIGHT_EVIDENCE_PROVENANCE_KEY",
    "READINESS_EVIDENCE_OPTION",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
