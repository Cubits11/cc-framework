"""CLI for building CC reports and canonical receipts."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from cc.reporting.report import (
    CLAIM_LEVEL_DESCRIPTIONS,
    CLAIM_LEVELS,
    CalibrationSummary,
    ClaimSummary,
    EnvironmentMetadata,
    EvidenceArtifact,
    GitMetadata,
    MeasurementSummary,
    ReportValidationError,
    RunSummary,
    build_cc_report,
    sha256_file,
    write_cc_report,
)

_EVIDENCE_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def build_parser() -> argparse.ArgumentParser:
    claim_level_help = "\n".join(
        f"  {level}: {CLAIM_LEVEL_DESCRIPTIONS[level]}" for level in CLAIM_LEVELS
    )
    parser = argparse.ArgumentParser(
        description="Build machine-checkable CC reports.",
        epilog=f"Allowed claim levels:\n{claim_level_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-report", help="Build a CC report JSON file.")
    build.add_argument("--run-id", required=True)
    build.add_argument("--measurement-json", required=True, type=Path)
    build.add_argument("--calibration-json", required=True, type=Path)
    build.add_argument("--evidence", nargs="*", default=[], type=Path)
    build.add_argument("--decay-policy", type=Path)
    build.add_argument("--extremal-scenario", action="append", default=[], type=Path)
    build.add_argument("--evidence-role", action="append", default=[])
    build.add_argument("--audit-log", type=Path)
    build.add_argument("--figure-manifest", type=Path)
    build.add_argument("--claim", required=True)
    build.add_argument(
        "--claim-level",
        required=True,
        choices=CLAIM_LEVELS,
    )
    build.add_argument("--non-claim", action="append", default=[])
    build.add_argument("--assumption", action="append", default=[])
    build.add_argument("--config-path")
    build.add_argument("--config-hash")
    build.add_argument("--seed", type=int)
    build.add_argument("--command")
    build.add_argument("--report-id")
    build.add_argument("--created-at")
    build.add_argument("--framework-version")
    build.add_argument("--git-commit")
    build.add_argument("--git-dirty", choices=["true", "false"])
    build.add_argument("--git-branch")
    build.add_argument("--python-version")
    build.add_argument("--platform")
    build.add_argument("--dependency-hash")
    build.add_argument("--previous-hash")
    build.add_argument("--out", required=True, type=Path)
    build.set_defaults(func=_cmd_build_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ReportValidationError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_build_report(args: argparse.Namespace) -> int:
    measurement_payload = _read_json(args.measurement_json)
    calibration_payload = _read_json(args.calibration_json)

    role_overrides = _parse_evidence_roles(args.evidence_role)
    evidence = _evidence_artifacts(
        args.evidence,
        role_overrides=role_overrides,
        dedicated_artifacts=[
            (args.decay_policy, "claim_decay"),
            *[(path, "extremal_scenario") for path in args.extremal_scenario],
        ],
    )
    audit_log = (
        EvidenceArtifact.from_path(args.audit_log, role="audit_log") if args.audit_log else None
    )
    figure_manifest = (
        EvidenceArtifact.from_path(args.figure_manifest, role="figure_manifest")
        if args.figure_manifest
        else None
    )

    git = None
    if args.git_commit is not None or args.git_dirty is not None or args.git_branch is not None:
        git = GitMetadata(
            commit=args.git_commit,
            dirty=(args.git_dirty == "true"),
            branch=args.git_branch,
        )

    environment = None
    if (
        args.python_version is not None
        or args.platform is not None
        or args.dependency_hash is not None
    ):
        environment = EnvironmentMetadata(
            python_version=args.python_version or sys.version.split()[0],
            platform=args.platform or sys.platform,
            dependency_hash=args.dependency_hash,
        )

    command = args.command or _command_string(["python", "-m", "cc.reporting.cli", *sys.argv[1:]])
    config_hash = args.config_hash
    if config_hash is None and args.config_path:
        config_path = Path(args.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_hash = sha256_file(config_path)

    report = build_cc_report(
        run=RunSummary(
            run_id=args.run_id,
            config_path=args.config_path,
            config_hash=config_hash,
            seed=args.seed,
            command=command,
        ),
        calibration=_calibration_from_payload(calibration_payload),
        measurement=_measurement_from_payload(measurement_payload),
        claim=ClaimSummary(
            statement=args.claim,
            allowed_claim_level=args.claim_level,
            non_claims=args.non_claim,
        ),
        evidence_artifacts=evidence,
        audit_log=audit_log,
        figure_manifest=figure_manifest,
        assumptions=args.assumption,
        report_id=args.report_id,
        created_at=args.created_at,
        framework_version=args.framework_version,
        git=git,
        environment=environment,
        previous_hash=args.previous_hash,
    )
    write_cc_report(args.out, report)
    print(f"report: {args.out}")
    print(f"receipt_sha256: {report['receipt']['canonical_hash']}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON input not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _calibration_from_payload(payload: dict[str, Any]) -> CalibrationSummary:
    target_window = payload.get("calibration_window", payload.get("target_window"))
    target_fpr = _optional_float(
        payload.get("target_fpr", payload.get("fpr_target", payload.get("target_FPR")))
    )
    if target_fpr is None and isinstance(target_window, list) and len(target_window) == 2:
        target_fpr = (float(target_window[0]) + float(target_window[1])) / 2.0

    status = payload.get("status", payload.get("pass_fail"))
    passed = payload.get("passed")
    if status is None and isinstance(passed, bool):
        status = "pass" if passed else "fail"
    if status is None:
        raise ReportValidationError(
            "calibration JSON must include status/pass_fail or a boolean passed field"
        )

    window: dict[str, Any]
    if isinstance(target_window, dict):
        window = dict(target_window)
    elif isinstance(target_window, list) and len(target_window) == 2:
        window = {"lower": target_window[0], "upper": target_window[1]}
    elif target_window is None:
        window = {}
    else:
        raise ReportValidationError("calibration_window/target_window must be an object or pair")

    return CalibrationSummary(
        target_fpr=target_fpr,
        alpha_cap=_optional_float(payload.get("alpha_cap", payload.get("alpha"))),
        realized_fpr=_optional_float(
            payload.get("realized_fpr", payload.get("fpr", payload.get("stack_fpr")))
        ),
        calibration_window=window,
        threshold=_optional_float(payload.get("threshold")),
        status=str(status).lower(),
    )


def _measurement_from_payload(payload: dict[str, Any]) -> MeasurementSummary:
    if "measurement" in payload and isinstance(payload["measurement"], dict):
        payload = dict(payload["measurement"])

    interval = payload.get("interval")
    if isinstance(interval, dict):
        lower = interval.get("lower", interval.get("lo"))
        upper = interval.get("upper", interval.get("hi"))
    else:
        lower = payload.get("interval_lower", payload.get("lo"))
        upper = payload.get("interval_upper", payload.get("hi"))

    point = payload.get("point_estimate", payload.get("estimate"))
    metric_family = payload.get("metric_family")
    interval_method = payload.get("interval_method", payload.get("method"))
    delta = payload.get("delta")
    confidence_level = payload.get("confidence_level")
    sample_sizes = payload.get("sample_sizes")

    if point is None and isinstance(payload.get("point"), dict):
        point = payload["point"].get("cc_hat")
        metric_family = metric_family or "CC"
    if lower is None and isinstance(payload.get("ci"), dict):
        lower = payload["ci"].get("lo")
        upper = payload["ci"].get("hi")
        interval_method = interval_method or "FH-Bernstein"
        delta = delta if delta is not None else payload["ci"].get("delta")
    if sample_sizes is None and {"n1", "n0"}.issubset(payload):
        sample_sizes = {"n1": payload["n1"], "n0": payload["n0"]}

    if point is None or lower is None or upper is None:
        raise ReportValidationError(
            "measurement JSON must include point_estimate and interval lower/upper"
        )
    if sample_sizes is None or not isinstance(sample_sizes, dict):
        raise ReportValidationError("measurement JSON must include sample_sizes object")
    if confidence_level is None and delta is not None:
        confidence_level = 1.0 - float(delta)

    return MeasurementSummary(
        metric_family=str(metric_family or "CC"),
        point_estimate=float(point),
        interval_lower=float(lower),
        interval_upper=float(upper),
        confidence_level=_optional_float(confidence_level),
        delta=_optional_float(delta),
        interval_method=str(interval_method or "unspecified"),
        sample_sizes={str(key): int(value) for key, value in sample_sizes.items()},
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_evidence_roles(entries: list[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for entry in entries:
        if entry.count("=") != 1:
            raise ValueError("--evidence-role entries must be formatted as PATH=ROLE")
        raw_path, raw_role = entry.split("=", 1)
        path = raw_path.strip()
        role = raw_role.strip()
        if not path or not role:
            raise ValueError("--evidence-role entries must include a non-empty PATH and ROLE")
        if path != raw_path or role != raw_role:
            raise ValueError(
                "--evidence-role PATH and ROLE must not include surrounding whitespace"
            )
        if _EVIDENCE_ROLE_RE.fullmatch(role) is None:
            raise ValueError(f"--evidence-role ROLE must match ^[a-z][a-z0-9_]*$ (got {role!r})")
        roles[path] = role
    return roles


def _evidence_artifacts(
    evidence_paths: list[Path],
    *,
    role_overrides: dict[str, str],
    dedicated_artifacts: list[tuple[Path | None, str]],
) -> list[EvidenceArtifact]:
    artifacts: list[EvidenceArtifact] = []
    seen: set[str] = set()

    for path in evidence_paths:
        key = str(path)
        artifacts.append(EvidenceArtifact.from_path(path, role=role_overrides.get(key, "artifact")))
        seen.add(key)

    for path, role in dedicated_artifacts:
        if path is None:
            continue
        key = str(path)
        artifacts.append(EvidenceArtifact.from_path(path, role=role_overrides.get(key, role)))
        seen.add(key)

    for raw_path, role in role_overrides.items():
        if raw_path in seen:
            continue
        artifacts.append(EvidenceArtifact.from_path(Path(raw_path), role=role))
    return artifacts


def _command_string(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
