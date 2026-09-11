"""Command-line entry point for paired external benchmark experiments."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.benchmark_manifest import save_benchmark_artifact_manifest
from experiments.external_benchmark import ExternalBenchmarkSpec, validate_seed_sequence
from experiments.paired_artifacts import save_paired_execution_result
from experiments.paired_benchmark import run_paired_external_benchmarks_with_preflight
from experiments.paired_source_preflight import run_controlled_paired_external_benchmarks
from experiments.runtime_provenance import collect_runtime_provenance
from experiments.runtime_requirements import RuntimeRequirements
from experiments.source_checkouts import SourceCheckoutProvenance, SourceCheckoutRequirement

DEFAULT_OUTPUT_PATH = Path("artifacts/paired-benchmark.json")
PAIRED_PREFLIGHT_STATUS_KEY = "paired_runtime_preflight"
PAIRED_PREFLIGHT_PROBE_ACTION_KEY = "paired_runtime_preflight_probe_action"
RESET_ONLY_PREFLIGHT_VALUE = "reset-only"


def parse_args() -> argparse.Namespace:
    """Parse arguments for a paired baseline-versus-treatment experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated independent integer seeds")
    parser.add_argument("--environment-factory", required=True)
    parser.add_argument("--success-evaluator", required=True)
    parser.add_argument("--transfer-success-evaluator")
    _add_policy_arguments(parser, "baseline")
    _add_policy_arguments(parser, "treatment")
    parser.add_argument("--minimum-trust", type=float, default=0.0)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--treatment-label", default="treatment")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing paired benchmark report or integrity manifest",
    )
    parser.add_argument("--probe-action")
    parser.add_argument(
        "--require-code-revision",
        help="Require the exact repository revision before paired preflight or measurement",
    )
    parser.add_argument(
        "--require-clean-working-tree",
        action="store_true",
        help="Require a clean repository working tree before paired preflight or measurement",
    )
    parser.add_argument(
        "--require-dependency-version",
        action="append",
        metavar="PACKAGE==VERSION",
        help="Require an exact installed package version; may be specified multiple times",
    )
    parser.add_argument(
        "--source-checkout",
        action="append",
        metavar="NAME=PATH",
        help="Declare a named source checkout path; may be specified multiple times",
    )
    parser.add_argument(
        "--require-source-revision",
        action="append",
        metavar="NAME=REVISION",
        help="Require an exact Git revision for a named source checkout",
    )
    parser.add_argument(
        "--allow-dirty-source-checkout",
        action="append",
        metavar="NAME",
        help="Allow one declared source checkout to have local modifications",
    )
    return parser.parse_args()


def main() -> int:
    """Execute a paired experiment and persist its measured reports."""

    arguments = parse_args()
    try:
        seeds = _parse_seeds(arguments.seeds)
        baseline_spec = _build_spec(
            arguments,
            policy_factory=arguments.baseline_policy_factory,
            action_policy_factory=arguments.baseline_action_policy_factory,
        )
        treatment_spec = _build_spec(
            arguments,
            policy_factory=arguments.treatment_policy_factory,
            action_policy_factory=arguments.treatment_action_policy_factory,
        )
        runtime_requirements = _build_runtime_requirements(arguments)
        source_checkout_paths, source_checkout_requirements = _build_source_checkout_contract(
            arguments
        )
        _validate_artifact_destinations(arguments.output, arguments.manifest)
        output_path = _prepare_output_path(arguments.output, overwrite=arguments.overwrite)
        manifest_path = (
            _prepare_output_path(arguments.manifest, overwrite=arguments.overwrite)
            if arguments.manifest is not None
            else None
        )

        source_checkout_provenance: Mapping[str, SourceCheckoutProvenance] | None = None
        if source_checkout_paths is not None and source_checkout_requirements is not None:
            controlled_result = run_controlled_paired_external_benchmarks(
                baseline_spec,
                treatment_spec,
                seeds,
                baseline_label=arguments.baseline_label,
                treatment_label=arguments.treatment_label,
                probe_action=arguments.probe_action,
                runtime_requirements=runtime_requirements or RuntimeRequirements(),
                source_checkout_paths=source_checkout_paths,
                source_checkout_requirements=source_checkout_requirements,
            )
            result = controlled_result.paired_result
            runtime_provenance = controlled_result.runtime_provenance.to_dict()
            source_checkout_provenance = controlled_result.source_checkout_provenance
        else:
            result = run_paired_external_benchmarks_with_preflight(
                baseline_spec,
                treatment_spec,
                seeds,
                baseline_label=arguments.baseline_label,
                treatment_label=arguments.treatment_label,
                probe_action=arguments.probe_action,
                runtime_requirements=runtime_requirements,
            )
            if result.runtime_provenance is not None:
                runtime_provenance = result.runtime_provenance.to_dict()
            else:
                runtime_provenance = collect_runtime_provenance(environment=os.environ).to_dict()

        runtime_provenance.update(_paired_preflight_provenance(arguments.probe_action))
        output_path = save_paired_execution_result(
            result,
            output_path,
            runtime_provenance=runtime_provenance,
            runtime_requirements=runtime_requirements,
            source_checkout_provenance=source_checkout_provenance,
            source_checkout_requirements=source_checkout_requirements,
            overwrite=arguments.overwrite,
        )
        if manifest_path is not None:
            manifest_output = save_benchmark_artifact_manifest(output_path, manifest_path)
            print(f"saved paired benchmark artifact manifest: {manifest_output}")
        print(f"saved paired benchmark report: {output_path}")
        return 0
    except (TypeError, ValueError, FileExistsError) as exc:
        raise SystemExit(f"error: {exc}") from exc


def _add_policy_arguments(parser: argparse.ArgumentParser, condition: str) -> None:
    """Add exactly-one policy factory arguments for a benchmark condition."""

    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(f"--{condition}-policy-factory", dest=f"{condition}_policy_factory")
    policy_group.add_argument(
        f"--{condition}-action-policy-factory",
        dest=f"{condition}_action_policy_factory",
    )


def _build_spec(
    arguments: argparse.Namespace,
    *,
    policy_factory: str | None,
    action_policy_factory: str | None,
) -> ExternalBenchmarkSpec:
    """Build one condition specification from shared CLI configuration."""

    return ExternalBenchmarkSpec(
        benchmark_name=arguments.benchmark,
        episode_count=arguments.episodes,
        max_steps=arguments.max_steps,
        environment_factory=arguments.environment_factory,
        policy_factory=policy_factory,
        action_policy_factory=action_policy_factory,
        success_evaluator=arguments.success_evaluator,
        transfer_success_evaluator=arguments.transfer_success_evaluator,
        minimum_trust=arguments.minimum_trust,
        seed=None,
    )


def _build_runtime_requirements(arguments: argparse.Namespace) -> RuntimeRequirements | None:
    """Build fail-closed runtime requirements from optional paired CLI arguments."""

    expected_revision = getattr(arguments, "require_code_revision", None)
    require_clean_working_tree = getattr(arguments, "require_clean_working_tree", False)
    dependency_values = getattr(arguments, "require_dependency_version", None) or ()
    dependency_versions = _parse_dependency_requirements(dependency_values)
    if expected_revision is None and not require_clean_working_tree and not dependency_versions:
        return None
    return RuntimeRequirements(
        expected_code_revision=expected_revision,
        require_clean_working_tree=require_clean_working_tree,
        dependency_versions=dependency_versions,
    )


def _build_source_checkout_contract(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Path] | None, dict[str, SourceCheckoutRequirement] | None]:
    """Build a complete named source-checkout admission contract from CLI arguments."""

    path_values = getattr(arguments, "source_checkout", None) or ()
    revision_values = getattr(arguments, "require_source_revision", None) or ()
    allow_dirty_values = getattr(arguments, "allow_dirty_source_checkout", None) or ()
    if not path_values and not revision_values and not allow_dirty_values:
        return None, None

    checkout_paths = {
        name: Path(path)
        for name, path in _parse_named_values(path_values, option="--source-checkout").items()
    }
    revisions = _parse_named_values(
        revision_values,
        option="--require-source-revision",
    )
    allow_dirty = _parse_unique_names(
        allow_dirty_values,
        option="--allow-dirty-source-checkout",
    )

    path_names = {name.casefold(): name for name in checkout_paths}
    revision_names = {name.casefold(): name for name in revisions}
    if set(path_names) != set(revision_names):
        raise ValueError(
            "--source-checkout and --require-source-revision must declare the same names"
        )
    unknown_dirty_names = allow_dirty - set(path_names)
    if unknown_dirty_names:
        names = ", ".join(sorted(unknown_dirty_names))
        raise ValueError(f"--allow-dirty-source-checkout names must be declared sources: {names}")

    requirements: dict[str, SourceCheckoutRequirement] = {}
    for normalized_name in sorted(path_names):
        display_name = path_names[normalized_name]
        revision_name = revision_names[normalized_name]
        requirements[display_name] = SourceCheckoutRequirement(
            expected_revision=revisions[revision_name],
            require_clean_working_tree=normalized_name not in allow_dirty,
        )
    return checkout_paths, requirements


def _parse_named_values(values: Sequence[str], *, option: str) -> dict[str, str]:
    """Parse repeatable ``NAME=VALUE`` arguments with case-insensitive unique names."""

    parsed: dict[str, str] = {}
    normalized_names: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{option} values must be strings")
        name, separator, assigned_value = value.partition("=")
        name = name.strip()
        assigned_value = assigned_value.strip()
        if not separator or not name or not assigned_value:
            raise ValueError(f"{option} must use NAME=VALUE")
        normalized_name = name.casefold()
        if normalized_name in normalized_names:
            raise ValueError(f"{option} names must be unique ignoring case")
        normalized_names.add(normalized_name)
        parsed[name] = assigned_value
    return parsed


def _parse_unique_names(values: Sequence[str], *, option: str) -> set[str]:
    """Normalize repeatable source names while rejecting duplicates."""

    parsed: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{option} values must be strings")
        name = value.strip()
        if not name:
            raise ValueError(f"{option} names must not be empty")
        normalized_name = name.casefold()
        if normalized_name in parsed:
            raise ValueError(f"{option} names must be unique ignoring case")
        parsed.add(normalized_name)
    return parsed


def _parse_dependency_requirements(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated exact dependency pins in ``PACKAGE==VERSION`` form."""

    parsed: dict[str, str] = {}
    normalized_names: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("--require-dependency-version values must be strings")
        package_name, separator, package_version = value.partition("==")
        package_name = package_name.strip()
        package_version = package_version.strip()
        if not separator or not package_name or not package_version:
            raise ValueError("--require-dependency-version must use PACKAGE==VERSION")
        normalized_name = package_name.lower()
        if normalized_name in normalized_names:
            raise ValueError(
                "--require-dependency-version package names must be unique ignoring case"
            )
        normalized_names.add(normalized_name)
        parsed[package_name] = package_version
    return parsed


def _parse_seeds(value: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated independent seed sequence."""

    parts = tuple(part.strip() for part in value.split(","))
    if not parts or any(not part for part in parts):
        raise ValueError("--seeds must contain comma-separated integers")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--seeds must contain comma-separated integers") from exc
    return validate_seed_sequence(seeds)


def _prepare_output_path(path: Path, *, overwrite: bool) -> Path:
    """Reject accidental artifact replacement unless explicitly requested."""

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"benchmark artifact already exists: {path}; pass --overwrite to replace it"
        )
    return path


def _validate_artifact_destinations(output_path: Path, manifest_path: Path | None) -> None:
    """Reject report and manifest paths that resolve to the same artifact."""

    if manifest_path is None:
        return
    if output_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise ValueError("--manifest must resolve to a different path than --output")


def _paired_preflight_provenance(probe_action: str | None) -> dict[str, str]:
    """Describe the runtime preflight that completed before paired measurement.

    The paired CLI always executes runtime preflight before measured episodes. This
    metadata is added only after that call returns successfully, so persisted CLI
    artifacts can distinguish reset-only preflight from a concrete one-step probe.
    """

    return {
        PAIRED_PREFLIGHT_STATUS_KEY: "completed",
        PAIRED_PREFLIGHT_PROBE_ACTION_KEY: (
            probe_action if probe_action is not None else RESET_ONLY_PREFLIGHT_VALUE
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
