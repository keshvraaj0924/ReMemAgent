"""Verify a persisted benchmark report against its integrity and identity contracts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from experiments.benchmark_distribution_config import BENCHMARK_EPISODE_DURATION_METRIC
from experiments.benchmark_manifest import (
    load_benchmark_artifact_manifest,
    verify_benchmark_artifact,
)
from experiments.controlled_benchmark_artifacts import (
    validate_persisted_controlled_benchmark_artifact,
)
from experiments.evidence_bound_paired_cli import EXPERIMENT_PLAN_PROVENANCE_KEY
from experiments.paired_artifacts import validate_persisted_paired_artifact
from experiments.preflight_evidence import (
    PREFLIGHT_EVIDENCE_PROVENANCE_KEY,
    verify_controlled_paired_preflight_evidence,
)
from experiments.research_experiment_plan import (
    RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION,
    verify_research_experiment_plan,
)
from remem.benchmark_artifacts import validate_persisted_benchmark_artifact
from remem.observability import OBSERVATION_SNAPSHOT_SCHEMA_VERSION, ObservationSnapshot
from remem.observability_distribution_artifacts import (
    DISTRIBUTION_OBSERVATION_SCHEMA_VERSION,
    read_distribution_observation_snapshot,
)

_BUNDLE_DIGEST_DOMAIN = b"remem-benchmark-bundle-v1\0"
_OBSERVABILITY_BUNDLE_DIGEST_DOMAIN = b"remem-benchmark-observability-bundle-v1\0"
_BENCHMARK_EPISODES_COMPLETED_METRIC = "benchmark.episodes.completed"
_LEGACY_BENCHMARK_EPISODE_COMPLETED_METRIC = "benchmark.episode.completed"


@dataclass(frozen=True, slots=True)
class BenchmarkVerificationResult:
    """Machine-readable attestation for one successfully verified artifact."""

    schema_version: int
    byte_count: int
    sha256: str
    benchmark_name: str | None = None
    configuration_fingerprint: str | None = None
    experiment_identity: str | None = None
    preflight_evidence_sha256: str | None = None
    experiment_plan_schema_version: int | None = None
    experiment_plan_sha256: str | None = None
    experiment_plan_byte_count: int | None = None
    experiment_plan_file_sha256: str | None = None
    experiment_plan_name: str | None = None
    experiment_plan_remem_revision: str | None = None
    observability_schema_version: int | None = None
    observability_byte_count: int | None = None
    observability_sha256: str | None = None
    distribution_schema_version: int | None = None
    distribution_byte_count: int | None = None
    distribution_sha256: str | None = None
    bundle_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "benchmark_name": self.benchmark_name,
            "configuration_fingerprint": self.configuration_fingerprint,
            "experiment_identity": self.experiment_identity,
            "preflight_evidence_sha256": self.preflight_evidence_sha256,
            "observability_schema_version": self.observability_schema_version,
            "observability_byte_count": self.observability_byte_count,
            "observability_sha256": self.observability_sha256,
            "distribution_schema_version": self.distribution_schema_version,
            "distribution_byte_count": self.distribution_byte_count,
            "distribution_sha256": self.distribution_sha256,
            "bundle_sha256": self.bundle_sha256,
        }
        if self.experiment_plan_sha256 is not None:
            payload.update(
                {
                    "experiment_plan_schema_version": self.experiment_plan_schema_version,
                    "experiment_plan_sha256": self.experiment_plan_sha256,
                    "experiment_plan_byte_count": self.experiment_plan_byte_count,
                    "experiment_plan_file_sha256": self.experiment_plan_file_sha256,
                    "experiment_plan_name": self.experiment_plan_name,
                    "experiment_plan_remem_revision": self.experiment_plan_remem_revision,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class _SidecarVerification:
    """Exact-byte integrity metadata plus comparable benchmark measurements."""

    schema_version: int
    byte_count: int
    sha256: str
    episode_duration_total: float | None = None
    episode_count: float | None = None


@dataclass(frozen=True, slots=True)
class _ExperimentPlanVerification:
    """Validated frozen-plan identity and exact retained-file metadata."""

    schema_version: int
    canonical_sha256: str
    byte_count: int
    file_sha256: str
    experiment_name: str
    remem_revision: str


def parse_args() -> argparse.Namespace:
    """Parse the report and optional integrity/readiness evidence paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Persisted benchmark JSON artifact")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Integrity manifest; defaults to <report>.manifest.json",
    )
    parser.add_argument(
        "--preflight-evidence",
        type=Path,
        help=(
            "Persisted readiness evidence required to verify an evidence-bound measured artifact"
        ),
    )
    parser.add_argument(
        "--experiment-plan",
        type=Path,
        help=(
            "Persisted frozen research experiment plan required to verify a plan-bound "
            "measured artifact"
        ),
    )
    parser.add_argument(
        "--observability-sidecar",
        type=Path,
        help=(
            "Optional persisted aggregate observability sidecar to validate and bind into "
            "the verification attestation"
        ),
    )
    parser.add_argument(
        "--distribution-sidecar",
        type=Path,
        help=(
            "Optional persisted duration-distribution sidecar to validate and bind into "
            "the verification attestation"
        ),
    )
    parser.add_argument(
        "--attestation-output",
        type=Path,
        help=(
            "Optional output path for the canonical verification attestation; refuses to "
            "overwrite an existing file"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit the verified artifact attestation as canonical JSON",
    )
    return parser.parse_args()


def verify_report_artifact(
    report_path: Path,
    manifest_path: Path | None = None,
    preflight_evidence_path: Path | None = None,
    distribution_sidecar_path: Path | None = None,
    observability_sidecar_path: Path | None = None,
    experiment_plan_path: Path | None = None,
) -> BenchmarkVerificationResult:
    """Verify report bytes, identities, admission evidence, and optional sidecars."""

    selected_manifest_path = manifest_path or report_path.with_suffix(
        report_path.suffix + ".manifest.json"
    )
    manifest = load_benchmark_artifact_manifest(selected_manifest_path)
    verify_benchmark_artifact(report_path, manifest)
    payload = _load_report_payload(report_path)
    validate_persisted_benchmark_artifact(payload)
    validate_persisted_controlled_benchmark_artifact(payload)
    validate_persisted_paired_artifact(payload)
    benchmark_name = _optional_attested_string(payload, "benchmark_name")
    configuration_fingerprint = _optional_attested_string(
        payload,
        "configuration_fingerprint",
    )
    experiment_identity = _optional_attested_string(payload, "experiment_identity")
    preflight_evidence_sha256 = _verify_preflight_evidence_binding(
        payload,
        preflight_evidence_path,
    )
    experiment_plan = _verify_experiment_plan_binding(payload, experiment_plan_path)
    observability = _verify_observability_sidecar(observability_sidecar_path)
    distribution = _verify_distribution_sidecar(distribution_sidecar_path)
    _verify_sidecar_measurement_consistency(observability, distribution)
    bundle_sha256 = _select_bundle_digest(
        manifest.sha256,
        observability,
        distribution,
    )
    return BenchmarkVerificationResult(
        schema_version=manifest.schema_version,
        byte_count=manifest.byte_count,
        sha256=manifest.sha256,
        benchmark_name=benchmark_name,
        configuration_fingerprint=configuration_fingerprint,
        experiment_identity=experiment_identity,
        preflight_evidence_sha256=preflight_evidence_sha256,
        experiment_plan_schema_version=(
            experiment_plan.schema_version if experiment_plan is not None else None
        ),
        experiment_plan_sha256=(
            experiment_plan.canonical_sha256 if experiment_plan is not None else None
        ),
        experiment_plan_byte_count=(
            experiment_plan.byte_count if experiment_plan is not None else None
        ),
        experiment_plan_file_sha256=(
            experiment_plan.file_sha256 if experiment_plan is not None else None
        ),
        experiment_plan_name=(
            experiment_plan.experiment_name if experiment_plan is not None else None
        ),
        experiment_plan_remem_revision=(
            experiment_plan.remem_revision if experiment_plan is not None else None
        ),
        observability_schema_version=(
            observability.schema_version if observability is not None else None
        ),
        observability_byte_count=(observability.byte_count if observability is not None else None),
        observability_sha256=observability.sha256 if observability is not None else None,
        distribution_schema_version=(
            distribution.schema_version if distribution is not None else None
        ),
        distribution_byte_count=(distribution.byte_count if distribution is not None else None),
        distribution_sha256=distribution.sha256 if distribution is not None else None,
        bundle_sha256=bundle_sha256,
    )


def write_verification_attestation(
    path: str | Path,
    result: BenchmarkVerificationResult,
) -> None:
    """Atomically persist one canonical verification result without overwriting evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_verification_json(result)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"verification attestation already exists: {destination}"
                ) from exc
            temporary_path.unlink()
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def _canonical_verification_json(result: BenchmarkVerificationResult) -> str:
    """Return the exact canonical JSON representation used by stdout and persistence."""

    return (
        json.dumps(
            result.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _optional_attested_string(payload: Mapping[str, Any], key: str) -> str | None:
    """Return an optional top-level identity field after enforcing its string contract."""

    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark artifact {key} must be a non-empty string when present")
    return value


def _load_report_payload(report_path: Path) -> Mapping[str, Any]:
    """Load a persisted benchmark JSON object after byte-integrity verification."""

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("benchmark artifact must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark artifact root must be a JSON object")
    return payload


def _verify_observability_sidecar(
    observability_sidecar_path: Path | None,
) -> _SidecarVerification | None:
    """Validate and hash one retained aggregate-observability sidecar."""

    if observability_sidecar_path is None:
        return None
    raw_bytes = observability_sidecar_path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("observability sidecar must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("observability sidecar root must be a JSON object")
    snapshot = ObservationSnapshot.from_dict(payload)
    return _SidecarVerification(
        schema_version=OBSERVATION_SNAPSHOT_SCHEMA_VERSION,
        byte_count=len(raw_bytes),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        episode_duration_total=snapshot.durations_seconds.get(BENCHMARK_EPISODE_DURATION_METRIC),
        episode_count=_benchmark_completed_episode_count(snapshot),
    )


def _benchmark_completed_episode_count(snapshot: ObservationSnapshot) -> float | None:
    """Resolve the canonical completed count while accepting one historical alias."""

    canonical_count = snapshot.counters.get(_BENCHMARK_EPISODES_COMPLETED_METRIC)
    legacy_count = snapshot.counters.get(_LEGACY_BENCHMARK_EPISODE_COMPLETED_METRIC)
    if canonical_count is not None and legacy_count is not None and canonical_count != legacy_count:
        raise ValueError("observability sidecar has conflicting completed episode counters")
    return canonical_count if canonical_count is not None else legacy_count


def _verify_distribution_sidecar(
    distribution_sidecar_path: Path | None,
) -> _SidecarVerification | None:
    """Validate and hash one retained distribution sidecar without mutating it."""

    if distribution_sidecar_path is None:
        return None
    raw_bytes = distribution_sidecar_path.read_bytes()
    snapshot = read_distribution_observation_snapshot(distribution_sidecar_path)
    histogram = snapshot.duration_histograms.get(BENCHMARK_EPISODE_DURATION_METRIC)
    if histogram is None:
        raise ValueError("distribution sidecar is missing benchmark episode duration histogram")
    return _SidecarVerification(
        schema_version=DISTRIBUTION_OBSERVATION_SCHEMA_VERSION,
        byte_count=len(raw_bytes),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        episode_duration_total=histogram.total,
        episode_count=float(histogram.count),
    )


def _verify_sidecar_measurement_consistency(
    observability: _SidecarVerification | None,
    distribution: _SidecarVerification | None,
) -> None:
    """Reject sidecars that are individually valid but describe different measurements."""

    if observability is None or distribution is None:
        return
    if observability.episode_duration_total is None:
        raise ValueError("observability sidecar is missing benchmark episode duration aggregate")
    if observability.episode_count is None:
        raise ValueError("observability sidecar is missing benchmark episodes completed counter")
    if distribution.episode_duration_total is None or distribution.episode_count is None:
        raise ValueError("distribution sidecar is missing benchmark duration measurements")
    if observability.episode_count != distribution.episode_count:
        raise ValueError(
            "observability and distribution sidecars disagree on completed episode count"
        )
    if not math.isclose(
        observability.episode_duration_total,
        distribution.episode_duration_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "observability and distribution sidecars disagree on episode duration total"
        )


def _select_bundle_digest(
    report_sha256: str,
    observability: _SidecarVerification | None,
    distribution: _SidecarVerification | None,
) -> str | None:
    """Select a backward-compatible bundle digest for the supplied validated sidecars."""

    if observability is not None:
        return _build_observability_bundle_digest(
            report_sha256,
            observability.sha256,
            distribution.sha256 if distribution is not None else None,
        )
    if distribution is not None:
        return _build_bundle_digest(report_sha256, distribution.sha256)
    return None


def _build_bundle_digest(report_sha256: str, distribution_sha256: str) -> str:
    """Bind exact report and distribution digests using the established v1 contract."""

    digest = hashlib.sha256()
    digest.update(_BUNDLE_DIGEST_DOMAIN)
    digest.update(bytes.fromhex(report_sha256))
    digest.update(bytes.fromhex(distribution_sha256))
    return digest.hexdigest()


def _build_observability_bundle_digest(
    report_sha256: str,
    observability_sha256: str,
    distribution_sha256: str | None,
) -> str:
    """Bind report, aggregate telemetry, and optional distributions unambiguously."""

    digest = hashlib.sha256()
    digest.update(_OBSERVABILITY_BUNDLE_DIGEST_DOMAIN)
    digest.update(bytes.fromhex(report_sha256))
    digest.update(bytes.fromhex(observability_sha256))
    if distribution_sha256 is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        digest.update(bytes.fromhex(distribution_sha256))
    return digest.hexdigest()


def _verify_preflight_evidence_binding(
    payload: Mapping[str, Any],
    preflight_evidence_path: Path | None,
) -> str | None:
    """Verify an artifact's readiness digest against the retained evidence object.

    Evidence-bound measured artifacts fail closed unless the original readiness JSON
    is supplied. Legacy artifacts without the readiness digest remain independently
    verifiable and reject an unrelated evidence argument rather than silently
    implying a binding that was never persisted.
    """

    runtime_provenance = payload.get("runtime_provenance")
    stored_digest = (
        runtime_provenance.get(PREFLIGHT_EVIDENCE_PROVENANCE_KEY)
        if isinstance(runtime_provenance, Mapping)
        else None
    )

    if stored_digest is None and preflight_evidence_path is None:
        return None
    if stored_digest is None:
        raise ValueError("benchmark artifact is not bound to preflight evidence")
    if not isinstance(stored_digest, str):
        raise ValueError("preflight evidence provenance digest must be a string")
    if preflight_evidence_path is None:
        raise ValueError(
            "evidence-bound benchmark artifact requires --preflight-evidence for verification"
        )

    evidence = _load_preflight_evidence(preflight_evidence_path)
    verify_controlled_paired_preflight_evidence(evidence)
    evidence_digest = evidence.get("evidence_sha256")
    if not isinstance(evidence_digest, str):
        raise ValueError("verified preflight evidence must contain evidence_sha256")
    if not hmac.compare_digest(stored_digest, evidence_digest):
        raise ValueError("benchmark artifact readiness digest does not match preflight evidence")
    return evidence_digest


def _verify_experiment_plan_binding(
    payload: Mapping[str, Any],
    experiment_plan_path: Path | None,
) -> _ExperimentPlanVerification | None:
    """Verify a report's frozen-plan digest against the exact retained plan file."""

    runtime_provenance = payload.get("runtime_provenance")
    stored_digest = (
        runtime_provenance.get(EXPERIMENT_PLAN_PROVENANCE_KEY)
        if isinstance(runtime_provenance, Mapping)
        else None
    )

    if stored_digest is None and experiment_plan_path is None:
        return None
    if stored_digest is None:
        raise ValueError("benchmark artifact is not bound to a research experiment plan")
    if not isinstance(stored_digest, str):
        raise ValueError("research experiment plan provenance digest must be a string")
    if experiment_plan_path is None:
        raise ValueError(
            "plan-bound benchmark artifact requires --experiment-plan for verification"
        )

    raw_bytes = experiment_plan_path.read_bytes()
    plan = verify_research_experiment_plan(
        experiment_plan_path,
        expected_sha256=stored_digest,
    )
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError("plan-bound benchmark artifact requires runtime provenance")
    code_revision = runtime_provenance.get("code_revision")
    if not isinstance(code_revision, str) or not code_revision:
        raise ValueError(
            "plan-bound benchmark artifact runtime provenance must contain code_revision"
        )
    if code_revision != plan.remem_revision:
        raise ValueError(
            "benchmark artifact code revision does not match research experiment plan"
        )

    return _ExperimentPlanVerification(
        schema_version=RESEARCH_EXPERIMENT_PLAN_SCHEMA_VERSION,
        canonical_sha256=plan.sha256,
        byte_count=len(raw_bytes),
        file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        experiment_name=plan.experiment_name,
        remem_revision=plan.remem_revision,
    )


def _load_preflight_evidence(path: Path) -> Mapping[str, Any]:
    """Load retained readiness evidence without recollecting mutable runtime state."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("preflight evidence must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("preflight evidence root must be a JSON object")
    return payload


def main() -> int:
    """Verify a benchmark report and return a process exit status."""

    arguments = parse_args()
    try:
        result = verify_report_artifact(
            report_path=arguments.report,
            manifest_path=arguments.manifest,
            preflight_evidence_path=arguments.preflight_evidence,
            distribution_sidecar_path=arguments.distribution_sidecar,
            observability_sidecar_path=arguments.observability_sidecar,
            experiment_plan_path=arguments.experiment_plan,
        )
        if arguments.attestation_output is not None:
            write_verification_attestation(arguments.attestation_output, result)
    except (OSError, TypeError, ValueError) as error:
        print(f"benchmark artifact verification failed: {error}", file=sys.stderr)
        return 1

    if arguments.json_output:
        print(_canonical_verification_json(result), end="")
    else:
        # Preserve the established CLI success prefix for callers that parse it.
        # Configuration identity is still verified by verify_report_artifact when present.
        print(f"benchmark artifact integrity verified: {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
