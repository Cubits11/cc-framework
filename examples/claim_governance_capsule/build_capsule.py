"""Build and verify the deterministic claim-governance capsule.

This script is intentionally small, but it exercises the implemented governance
chain end to end:

checked-in failure matrix -> finite-atom bounds -> endpoint scenarios ->
decay policy -> CC report receipt -> governance audit -> claim envelope ->
deterministic manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cc.evidence.claim_envelope import compile_claim_envelope
from cc.evidence.claim_governance import GovernanceVerdict, verify_claim_governance
from cc.evidence.decay import ClaimDecayPolicy, ClaimDecayRecord, VersionWatchSet
from cc.evidence.extremal_scenario import ExtremalScenario
from cc.kernel.frechet_classes import PairwiseDependence, frechet_bounds
from cc.reporting.report import (
    CalibrationSummary,
    ClaimSummary,
    EnvironmentMetadata,
    EvidenceArtifact,
    GitMetadata,
    MeasurementSummary,
    RunSummary,
    build_cc_report,
    sha256_file,
    write_cc_report,
)


CAPSULE_ID = "deterministic-claim-governance-capsule"
MANIFEST_SCHEMA_VERSION = "cc.claim_governance_capsule_manifest.v1"
REPORT_FILENAME = "cc_report.json"
AUDIT_FILENAME = "claim_governance_audit.json"
ENVELOPE_FILENAME = "claim_envelope.json"
MANIFEST_FILENAME = "capsule_manifest.json"

JSON_KWARGS = {
    "indent": 2,
    "sort_keys": True,
    "ensure_ascii": False,
    "allow_nan": False,
}

OUTPUT_ROLES = {
    "audit_log.jsonl": "audit_log",
    "bounds.json": "measurement_evidence",
    "calibration.json": "calibration_evidence",
    "cc_report.json": "cc_report",
    "claim_envelope.json": "claim_envelope",
    "claim_governance_audit.json": "claim_governance_audit",
    "confirmatory_failure_matrix.json": "confirmatory_failure_matrix",
    "confirmatory_protocol.json": "confirmatory_protocol",
    "decay_policy.json": "claim_decay",
    "extremal_lower.json": "extremal_scenario",
    "extremal_upper.json": "extremal_scenario",
}


class CapsuleError(RuntimeError):
    """Raised when the capsule cannot be regenerated or verified."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capsule-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to examples/claim_governance_capsule.",
    )
    parser.add_argument(
        "--update-expected",
        action="store_true",
        help="Regenerate checked-in expected artifacts from the freshly built outputs.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only compare the existing outputs against expected artifacts.",
    )
    args = parser.parse_args(argv)

    capsule_dir = args.capsule_dir.resolve()
    outputs_dir = capsule_dir / "outputs"
    expected_dir = capsule_dir / "expected"
    expected_manifest = capsule_dir / "manifest.expected.json"

    try:
        if not args.verify_only:
            build_capsule(capsule_dir)
        if args.update_expected:
            update_expected(outputs_dir, expected_dir, expected_manifest)
        verify_expected(outputs_dir, expected_dir, expected_manifest)
    except CapsuleError as exc:
        print(f"capsule verification failed: {exc}")
        return 1

    print(f"capsule verification passed: {outputs_dir / MANIFEST_FILENAME}")
    return 0


def build_capsule(capsule_dir: Path) -> None:
    inputs_dir = capsule_dir / "inputs"
    outputs_dir = capsule_dir / "outputs"
    clean_outputs(outputs_dir)

    config_path = inputs_dir / "capsule_config.json"
    matrix_path = inputs_dir / "failure_matrix.csv"
    config = read_json(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)

    guardrails = tuple(str(item) for item in config["guardrails"])
    rows = read_failure_matrix(matrix_path, guardrails=guardrails)
    matrix = np.asarray([[row[guardrail] for guardrail in guardrails] for row in rows], dtype=int)
    event = str(config["event"])
    if event != "and":
        raise CapsuleError("capsule_config.json currently supports only event='and'")

    input_hash = sha256_file(matrix_path)
    bounds_payload, bound_result = build_bounds_payload(
        matrix=matrix,
        guardrails=guardrails,
        source_path="inputs/failure_matrix.csv",
        source_sha256=input_hash,
        event=event,
        seed=seed,
    )
    write_json(outputs_dir / "bounds.json", bounds_payload)

    calibration_payload = dict(config["calibration"])
    write_json(outputs_dir / "calibration.json", calibration_payload)

    write_json(
        outputs_dir / "confirmatory_failure_matrix.json",
        build_confirmatory_failure_matrix_payload(
            rows=rows,
            guardrails=guardrails,
            bounds_payload=bounds_payload,
            config=config,
        ),
    )

    write_json(
        outputs_dir / "decay_policy.json",
        build_decay_payload(config=config, evidence_refs=tuple(sorted(OUTPUT_ROLES))),
    )

    lower_scenario = build_scenario(
        bound_result,
        endpoint="lower",
        guardrails=guardrails,
        source_hash=sha256_file(outputs_dir / "bounds.json"),
    )
    upper_scenario = build_scenario(
        bound_result,
        endpoint="upper",
        guardrails=guardrails,
        source_hash=sha256_file(outputs_dir / "bounds.json"),
    )
    write_json(outputs_dir / "extremal_lower.json", lower_scenario.model_dump(mode="json"))
    write_json(outputs_dir / "extremal_upper.json", upper_scenario.model_dump(mode="json"))

    write_json(
        outputs_dir / "confirmatory_protocol.json",
        build_confirmatory_protocol_payload(config=config, outputs_dir=outputs_dir),
    )
    write_audit_log(
        outputs_dir / "audit_log.jsonl",
        config=config,
        rows=rows,
        input_hash=input_hash,
        bounds_payload=bounds_payload,
    )

    report = build_report(config=config, outputs_dir=outputs_dir, bounds_payload=bounds_payload)
    report_path = outputs_dir / REPORT_FILENAME
    write_cc_report(report_path, report)

    now = parse_utc(str(config["fixed_now"]))
    audit = verify_claim_governance(report_path, now=now, base_dir=outputs_dir)
    write_json(outputs_dir / AUDIT_FILENAME, audit.model_dump(mode="json"))
    if audit.verdict is not GovernanceVerdict.PASS:
        raise CapsuleError(f"governance audit verdict was {audit.verdict.value}, expected pass")

    envelope = compile_claim_envelope(report, governance_audit=audit)
    write_json(outputs_dir / ENVELOPE_FILENAME, envelope.to_canonical_dict())

    manifest = build_manifest(
        capsule_dir=capsule_dir,
        outputs_dir=outputs_dir,
        config=config,
        audit_verdict=audit.verdict.value,
        report_hash=str(report["receipt"]["canonical_hash"]),
    )
    write_json(outputs_dir / MANIFEST_FILENAME, manifest)


def build_bounds_payload(
    *,
    matrix: np.ndarray,
    guardrails: tuple[str, ...],
    source_path: str,
    source_sha256: str,
    event: str,
    seed: int,
) -> tuple[dict[str, Any], Any]:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(guardrails):
        raise CapsuleError("failure matrix must be non-empty and match configured guardrails")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise CapsuleError("failure matrix values must be binary")

    n = int(matrix.shape[0])
    marginals = matrix.mean(axis=0)
    pairwise = []
    pairwise_payload: dict[str, float] = {}
    for i, left in enumerate(guardrails):
        for j in range(i + 1, len(guardrails)):
            right = guardrails[j]
            joint = float(np.mean(matrix[:, i] * matrix[:, j]))
            pairwise.append(PairwiseDependence(i=i, j=j, kind="joint_probability", value=joint))
            pairwise_payload[f"{left},{right}"] = joint

    result = frechet_bounds(
        marginals,
        pairwise=pairwise,
        event=event,  # type: ignore[arg-type]
        return_distributions=True,
    )
    observed = float(np.mean(np.all(matrix == 1, axis=1)))
    if not (result.lower - 1e-12 <= observed <= result.upper + 1e-12):
        raise CapsuleError("observed event rate fell outside the finite-atom Frechet interval")

    return (
        {
            "schema_version": "cc.capsule.bounds.v1",
            "source": {
                "path": source_path,
                "sha256": source_sha256,
                "rows": n,
            },
            "seed": seed,
            "failure_event_convention": "1 means guardrail failure or unsafe pass.",
            "event": event,
            "guardrails": list(guardrails),
            "metric_family": "CC",
            "marginals": {
                guardrail: round(float(marginals[idx]), 12)
                for idx, guardrail in enumerate(guardrails)
            },
            "pairwise_joint_probabilities": {
                key: round(value, 12) for key, value in sorted(pairwise_payload.items())
            },
            "observed_event_rate": round(observed, 12),
            "interval": {
                "lower": round(float(result.lower), 12),
                "upper": round(float(result.upper), 12),
            },
            "interval_method": "finite_atom_frechet_pairwise_identification",
            "width": round(float(result.width), 12),
            "non_claims": [
                "The measurement interval is scoped to this capsule fixture and does not "
                "certify safe deployment.",
                "The finite-atom interval is an identification envelope over declared "
                "marginals and pairwise co-failures, not a sampling-validity proof.",
            ],
        },
        result,
    )


def build_confirmatory_failure_matrix_payload(
    *,
    rows: Sequence[Mapping[str, int | str]],
    guardrails: tuple[str, ...],
    bounds_payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "cc.confirmatory_failure_matrix.v1",
        "protocol_id": str(config["confirmatory_protocol_id"]),
        "analysis_id": str(config["analysis_id"]),
        "event": str(config["event"]),
        "guardrails": list(guardrails),
        "failure_matrix": [
            {
                "prompt_id": str(row["prompt_id"]),
                **{guardrail: int(row[guardrail]) for guardrail in guardrails},
            }
            for row in rows
        ],
        "confirmatory_ci": [
            float(bounds_payload["interval"]["lower"]),
            float(bounds_payload["interval"]["upper"]),
        ],
        "non_claims": [
            "Confirmatory failure-matrix evidence is scoped and does not certify "
            "deployment safety.",
            "The held-out matrix binds this capsule run only; it does not prove external "
            "validity or representativeness.",
        ],
    }


def build_decay_payload(
    *, config: Mapping[str, Any], evidence_refs: tuple[str, ...]
) -> dict[str, Any]:
    decay = ClaimDecayRecord(
        claim_id=str(config["claim_id"]),
        issued_at=parse_utc(str(config["issued_at"])),
        policy=ClaimDecayPolicy(
            policy_id=str(config["decay_policy"]["policy_id"]),
            degraded_after_days=float(config["decay_policy"]["degraded_after_days"]),
            expires_after_days=float(config["decay_policy"]["expires_after_days"]),
        ),
        version_watch_set=VersionWatchSet(),
        evidence_refs=evidence_refs,
        notes=(
            "Verification computes freshness at fixed_now; the signed artifact does not store "
            "a live PASS state.",
        ),
    )
    return decay.to_dict()


def build_scenario(
    bound_result: Any,
    *,
    endpoint: str,
    guardrails: tuple[str, ...],
    source_hash: str,
) -> ExtremalScenario:
    scenario = ExtremalScenario.from_frechet_result(
        bound_result,
        endpoint=endpoint,  # type: ignore[arg-type]
        source_hash=source_hash,
        narrative=(
            f"Frechet {endpoint} endpoint for the capsule AND failure event under the "
            "checked-in marginals and pairwise co-failure constraints."
        ),
    )
    return scenario.model_copy(
        update={
            "guardrail_ids": guardrails,
            "metadata": {
                **scenario.metadata,
                "capsule_id": CAPSULE_ID,
                "constraint_surface": "marginals_plus_pairwise_joint_probabilities",
            },
        }
    )


def build_confirmatory_protocol_payload(
    *, config: Mapping[str, Any], outputs_dir: Path
) -> dict[str, Any]:
    protocol_non_claims = [
        "Confirmatory validity depends on the pre-registered protocol and run separation, "
        "not on report polish.",
        "A confirmatory_protocol artifact does not certify deployment safety or external "
        "validity.",
        "Adaptive discovery evidence may motivate a hypothesis but cannot become "
        "confirmatory evidence by renaming its role.",
    ]
    return {
        "schema_version": "cc.confirmatory_protocol.v1",
        "artifact_id": "capsule-confirmatory-protocol-artifact",
        "plan": {
            "protocol_id": str(config["confirmatory_protocol_id"]),
            "hypothesis": (
                "The checked-in held-out matrix bounds the AND composed guardrail-failure "
                "endpoint under the frozen pairwise Frechet analysis."
            ),
            "discovery_ref": {
                "artifact_id": "capsule-roadmap-hypothesis",
                "artifact_role": "artifact",
                "artifact_sha256": sha256_file(outputs_dir / "bounds.json"),
                "adaptive": False,
                "description": "The hypothesis is fixed by the capsule configuration.",
                "discovered_at": str(config["created_at"]),
            },
            "protocol_mode": "held_out_matrix",
            "created_at": str(config["protocol_created_at"]),
            "primary_endpoint": "and_composed_guardrail_failure_rate",
            "fixed_analysis_plan": {
                "analysis_id": str(config["analysis_id"]),
                "estimand": "AND composed guardrail-failure probability",
                "interval_method": "finite_atom_frechet_pairwise_identification",
                "alpha": 0.05,
                "multiplicity_adjustment": "none_predeclared_single_endpoint",
                "frozen": True,
            },
            "sample_plan": {
                "sampling_frame": "checked-in capsule failure_matrix.csv",
                "unit": "prompt",
                "target_n": int(config["target_n"]),
                "held_out_selection": "pre_failure_pattern",
                "clustered_data": False,
            },
            "stopping_rule": {
                "rule_id": "fixed-n-capsule",
                "description": "Evaluate exactly the checked-in matrix with no early stopping.",
                "max_samples": int(config["target_n"]),
                "max_looks": 1,
                "early_stopping_allowed": False,
            },
            "cluster_blocking": None,
            "exclusion_rules": [
                {
                    "rule_id": "binary-outcomes-only",
                    "field": "guardrail outcome columns",
                    "reason": "The finite-atom kernel requires binary failure indicators.",
                }
            ],
            "decision_rule": {
                "rule_id": "scoped-upper-bound-record",
                "description": "Record the predeclared upper endpoint; do not make a release gate.",
                "threshold": 0.2,
                "pass_condition": "upper_bound <= 0.2 under fixed inputs",
                "fail_condition": "upper_bound > 0.2 under fixed inputs",
            },
            "non_claims": protocol_non_claims,
        },
        "run": {
            "run_id": str(config["run_id"]),
            "started_at": str(config["run_started_at"]),
            "completed_at": str(config["run_completed_at"]),
            "artifact_id": "capsule-confirmatory-failure-matrix",
            "artifact_role": "confirmatory_failure_matrix",
            "source_role": None,
            "artifact_sha256": sha256_file(outputs_dir / "confirmatory_failure_matrix.json"),
            "primary_endpoint": "and_composed_guardrail_failure_rate",
            "analysis_plan_id": str(config["analysis_id"]),
            "used_adaptive_discovery_data": False,
            "held_out_set_chosen_after_failure_pattern": False,
            "clustered_data_observed": False,
            "non_claims": [],
        },
        "non_claims": protocol_non_claims,
    }


def write_audit_log(
    path: Path,
    *,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, int | str]],
    input_hash: str,
    bounds_payload: Mapping[str, Any],
) -> None:
    records = [
        {
            "step": "load_inputs",
            "fixed_now": str(config["fixed_now"]),
            "seed": int(config["seed"]),
            "input_sha256": input_hash,
            "rows": len(rows),
        },
        {
            "step": "compute_bounds",
            "event": bounds_payload["event"],
            "lower": bounds_payload["interval"]["lower"],
            "upper": bounds_payload["interval"]["upper"],
            "observed_event_rate": bounds_payload["observed_event_rate"],
        },
        {
            "step": "emit_governance_chain",
            "chain": [
                "bounds",
                "extremal_scenarios",
                "decay",
                "report",
                "governance_audit",
                "claim_envelope",
                "manifest",
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def build_report(
    *,
    config: Mapping[str, Any],
    outputs_dir: Path,
    bounds_payload: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = [
        artifact(outputs_dir / "bounds.json", "measurement_evidence"),
        artifact(outputs_dir / "calibration.json", "calibration_evidence"),
        artifact(outputs_dir / "confirmatory_failure_matrix.json", "confirmatory_failure_matrix"),
        artifact(outputs_dir / "confirmatory_protocol.json", "confirmatory_protocol"),
        artifact(outputs_dir / "decay_policy.json", "claim_decay"),
        artifact(outputs_dir / "extremal_lower.json", "extremal_scenario"),
        artifact(outputs_dir / "extremal_upper.json", "extremal_scenario"),
    ]
    calibration = config["calibration"]
    return build_cc_report(
        run=RunSummary(
            run_id=str(config["run_id"]),
            config_path="examples/claim_governance_capsule/inputs/capsule_config.json",
            config_hash=sha256_file(outputs_dir.parent / "inputs" / "capsule_config.json"),
            seed=int(config["seed"]),
            command="examples/claim_governance_capsule/reproduce.sh",
        ),
        calibration=CalibrationSummary(
            target_fpr=float(calibration["target_fpr"]),
            alpha_cap=float(calibration["alpha_cap"]),
            realized_fpr=float(calibration["realized_fpr"]),
            calibration_window=dict(calibration["calibration_window"]),
            threshold=float(calibration["threshold"]),
            status=str(calibration["status"]),
        ),
        measurement=MeasurementSummary(
            metric_family=str(bounds_payload["metric_family"]),
            point_estimate=float(bounds_payload["observed_event_rate"]),
            interval_lower=float(bounds_payload["interval"]["lower"]),
            interval_upper=float(bounds_payload["interval"]["upper"]),
            confidence_level=None,
            delta=0.0,
            interval_method=str(bounds_payload["interval_method"]),
            sample_sizes={"n": int(bounds_payload["source"]["rows"])},
        ),
        claim=ClaimSummary(
            statement=str(config["claim_statement"]),
            allowed_claim_level="bounded_empirical",
            non_claims=tuple(str(item) for item in config["claim_non_claims"]),
        ),
        evidence_artifacts=evidence,
        audit_log=artifact(outputs_dir / "audit_log.jsonl", "audit_log"),
        figure_manifest=None,
        assumptions=tuple(str(item) for item in config["assumptions"]),
        report_id=str(config["report_id"]),
        created_at=str(config["created_at"]),
        framework_version=str(config["framework_version"]),
        git=GitMetadata(commit="0" * 40, dirty=False, branch="main"),
        environment=EnvironmentMetadata(
            python_version=str(config["python_version"]),
            platform=str(config["platform"]),
            dependency_hash=hash_json(config["dependency_snapshot"]),
            package_snapshot=dict(config["dependency_snapshot"]),
        ),
    )


def artifact(path: Path, role: str) -> EvidenceArtifact:
    return EvidenceArtifact(
        path=path.name,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        role=role,
    )


def build_manifest(
    *,
    capsule_dir: Path,
    outputs_dir: Path,
    config: Mapping[str, Any],
    audit_verdict: str,
    report_hash: str,
) -> dict[str, Any]:
    files = []
    for filename, role in sorted(OUTPUT_ROLES.items()):
        path = outputs_dir / filename
        files.append(
            {
                "filename": filename,
                "role": role,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    inputs = []
    for path in sorted((capsule_dir / "inputs").iterdir(), key=lambda item: item.name):
        if path.is_file():
            inputs.append(
                {
                    "filename": f"inputs/{path.name}",
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "capsule_id": CAPSULE_ID,
        "generated_at": str(config["fixed_now"]),
        "fixed_now": str(config["fixed_now"]),
        "seed": int(config["seed"]),
        "report_id": str(config["report_id"]),
        "report_receipt_sha256": report_hash,
        "governance_verdict": audit_verdict,
        "inputs": inputs,
        "files": files,
        "pass_caveat": (
            "PASS means internal consistency under verifier rules; it does not mean the AI "
            "system is safe in deployment."
        ),
    }


def clean_outputs(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for path in outputs_dir.iterdir():
        if path.name == ".gitignore":
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def update_expected(outputs_dir: Path, expected_dir: Path, expected_manifest: Path) -> None:
    expected_dir.mkdir(parents=True, exist_ok=True)
    for path in expected_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for filename in sorted(OUTPUT_ROLES):
        shutil.copyfile(outputs_dir / filename, expected_dir / filename)
    shutil.copyfile(outputs_dir / MANIFEST_FILENAME, expected_manifest)


def verify_expected(outputs_dir: Path, expected_dir: Path, expected_manifest: Path) -> None:
    manifest_path = outputs_dir / MANIFEST_FILENAME
    if not expected_manifest.exists():
        raise CapsuleError(f"missing expected manifest: {expected_manifest}")
    if not manifest_path.exists():
        raise CapsuleError(f"missing generated manifest: {manifest_path}")
    expected_manifest_payload = read_json(expected_manifest)
    generated_manifest_payload = read_json(manifest_path)
    if generated_manifest_payload != expected_manifest_payload:
        raise CapsuleError("generated manifest differs from manifest.expected.json")

    for filename in sorted(OUTPUT_ROLES):
        expected_path = expected_dir / filename
        output_path = outputs_dir / filename
        if not expected_path.exists():
            raise CapsuleError(f"missing expected artifact: expected/{filename}")
        if not output_path.exists():
            raise CapsuleError(f"missing generated artifact: outputs/{filename}")
        expected_bytes = expected_path.read_bytes()
        output_bytes = output_path.read_bytes()
        if output_bytes != expected_bytes:
            raise CapsuleError(f"generated artifact differs from expected/{filename}")


def read_failure_matrix(path: Path, *, guardrails: tuple[str, ...]) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise CapsuleError("failure_matrix.csv is empty")
    required = {"prompt_id", *guardrails}
    missing = required - set(rows[0])
    if missing:
        raise CapsuleError(f"failure_matrix.csv missing columns: {sorted(missing)}")

    normalized = []
    for row in sorted(rows, key=lambda item: str(item["prompt_id"])):
        out: dict[str, int | str] = {"prompt_id": str(row["prompt_id"])}
        for guardrail in guardrails:
            raw = str(row[guardrail]).strip()
            if raw not in {"0", "1"}:
                raise CapsuleError(f"{guardrail} must be binary, got {raw!r}")
            out[guardrail] = int(raw)
        normalized.append(out)
    return normalized


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CapsuleError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_nonfinite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, **JSON_KWARGS) + "\n", encoding="utf-8")


def hash_json(payload: Mapping[str, Any]) -> str:
    assert_no_nonfinite(payload)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_no_nonfinite(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CapsuleError(f"non-finite float in generated payload: {value!r}")
    elif isinstance(value, Mapping):
        for item in value.values():
            assert_no_nonfinite(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            assert_no_nonfinite(item)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CapsuleError(f"timestamp must be timezone-aware: {value}")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
