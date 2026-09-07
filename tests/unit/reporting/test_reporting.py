from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from cc.reporting.canonical import CanonicalJSONError, canonical_json_bytes, sha256_canonical
from cc.reporting.report import (
    CANONICALIZATION_METHOD,
    CLAIM_LEVELS,
    CalibrationSummary,
    CCReport,
    ClaimSummary,
    EnvironmentMetadata,
    EvidenceArtifact,
    GitMetadata,
    MeasurementSummary,
    QuantitativeProposition,
    ReportValidationError,
    RunSummary,
    build_cc_report,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "reporting"


def test_canonical_hash_is_key_order_independent_and_value_sensitive() -> None:
    left = {"b": 2, "a": {"y": [1, 2], "x": "value"}}
    right = {"a": {"x": "value", "y": [1, 2]}, "b": 2}
    changed = {"a": {"x": "changed", "y": [1, 2]}, "b": 2}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_canonical(left) == sha256_canonical(right)
    assert sha256_canonical(left) != sha256_canonical(changed)


def test_canonical_hash_ignores_json_whitespace_and_rejects_noncanonical_types() -> None:
    compact = json.loads('{"a":{"x":"value","y":[1,2]},"b":2}')
    spaced = json.loads(
        """
        {
          "b": 2,
          "a": {
            "y": [1, 2],
            "x": "value"
          }
        }
        """
    )

    assert canonical_json_bytes(compact) == canonical_json_bytes(spaced)
    assert sha256_canonical(compact) == sha256_canonical(spaced)
    with pytest.raises(CanonicalJSONError, match="non-JSON-native"):
        canonical_json_bytes({"bad": {"set-members-are-unordered"}})
    with pytest.raises(CanonicalJSONError, match="non-string key"):
        canonical_json_bytes({1: "numeric keys are not canonical"})


def test_receipt_hash_excludes_canonical_hash_field() -> None:
    report = _report()
    original_hash = report["receipt"]["canonical_hash"]

    report["receipt"]["canonical_hash"] = "0" * 64

    assert sha256_canonical(report) == original_hash


def test_report_builder_happy_path_validates_against_schema() -> None:
    report = _report()

    Draft202012Validator(_schema()).validate(report)

    assert report["schema_version"] == "cc.report.v0.3.1"
    assert report["receipt"]["hash_algorithm"] == "sha256"
    assert report["receipt"]["canonical_hash"] == sha256_canonical(report)
    assert report["evidence"]["artifacts"][0]["sha256"]


def test_structured_quantitative_proposition_is_receipt_bound_and_schema_valid() -> None:
    report = _report(
        claim=ClaimSummary(
            statement="The structured interval proposition is checked separately from prose.",
            allowed_claim_level="bounded_empirical",
            non_claims=["This does not certify production safety."],
            quantitative_proposition=QuantitativeProposition(
                metric_family="CC", relation="upper_bound", threshold=1.2
            ),
        )
    )

    Draft202012Validator(_schema()).validate(report)
    parsed = CCReport.model_validate(report)

    assert parsed.claim.quantitative_proposition is not None
    assert parsed.claim.quantitative_proposition.threshold == 1.2
    assert report["receipt"]["canonical_hash"] == sha256_canonical(report)


def test_report_model_round_trip_preserves_canonical_json_identity() -> None:
    report = _report()

    model = CCReport.model_validate(report)
    payload = model.model_dump(mode="json")
    reparsed = CCReport.model_validate_json(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )

    assert payload == report
    assert isinstance(model.assumptions, tuple)
    assert isinstance(model.claim.non_claims, tuple)
    assert isinstance(model.evidence.artifacts, tuple)
    assert canonical_json_bytes(reparsed.model_dump(mode="json")) == canonical_json_bytes(payload)
    assert reparsed.receipt.canonical_hash == sha256_canonical(payload)


def test_report_model_fails_closed_on_incomplete_or_extra_payloads() -> None:
    missing = _report()
    del missing["claim"]["non_claims"]

    with pytest.raises(ValidationError, match="non_claims"):
        CCReport.model_validate(missing)

    extra = _report()
    extra["claim"]["lifecycle_state"] = "supported"

    with pytest.raises(ValidationError, match="Extra inputs"):
        CCReport.model_validate(extra)


def test_report_model_rejects_reserved_overclaim_vocabulary() -> None:
    report = _report()
    report["claim"]["statement"] = "This model is production_ready for deployment."
    report["receipt"]["canonical_hash"] = sha256_canonical(report)

    with pytest.raises(ValidationError, match="production_ready"):
        CCReport.model_validate(report)


def test_report_builder_rejects_invalid_interval_ordering() -> None:
    with pytest.raises(ReportValidationError, match="interval lower"):
        _report(
            measurement=MeasurementSummary(
                metric_family="CC",
                point_estimate=1.1,
                interval_lower=1.2,
                interval_upper=1.0,
                confidence_level=0.95,
                interval_method="FH-Bernstein",
                sample_sizes={"n1": 10, "n0": 10},
            )
        )


def test_report_builder_rejects_blank_claim_contract_text() -> None:
    with pytest.raises(ReportValidationError, match=r"claim\.statement"):
        _report(
            claim=ClaimSummary(
                statement="  ",
                allowed_claim_level="diagnostic",
            )
        )

    with pytest.raises(ReportValidationError, match=r"claim\.non_claims"):
        _report(
            claim=ClaimSummary(
                statement="Diagnostic receipt for fixture review.",
                allowed_claim_level="diagnostic",
                non_claims=["  "],
            )
        )


def test_missing_evidence_path_fails() -> None:
    with pytest.raises(FileNotFoundError, match="Evidence file not found"):
        EvidenceArtifact.from_path(FIXTURES / "missing.json")


def test_cli_builds_deterministic_report_from_fixture_inputs(tmp_path: Path) -> None:
    out1 = tmp_path / "report1.json"
    out2 = tmp_path / "report2.json"
    args = _cli_args(out1)

    first = _run_cli(args)
    second = _run_cli([*args[:-1], str(out2)])

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "receipt_sha256:" in first.stdout

    report1 = json.loads(out1.read_text(encoding="utf-8"))
    report2 = json.loads(out2.read_text(encoding="utf-8"))

    Draft202012Validator(_schema()).validate(report1)
    Draft202012Validator(_schema()).validate(report2)
    assert report1 == report2
    assert report1["receipt"]["canonical_hash"] == sha256_canonical(report1)


def test_cli_missing_evidence_returns_nonzero(tmp_path: Path) -> None:
    args = _cli_args(tmp_path / "report.json")
    missing_index = args.index("tests/fixtures/reporting/artifact.txt")
    args[missing_index] = "tests/fixtures/reporting/nope.txt"

    result = _run_cli(args)

    assert result.returncode != 0
    assert "Evidence file not found" in result.stderr


def test_new_evidence_roles_are_included_and_change_receipt_hash(tmp_path: Path) -> None:
    decay = tmp_path / "decay.json"
    extremal = tmp_path / "extremal.json"
    decay.write_text('{"schema_version":"cc.claim_decay.v1"}\n', encoding="utf-8")
    extremal.write_text('{"schema_version":"cc.extremal_scenario.v1"}\n', encoding="utf-8")

    base = _report()
    report = _report(
        evidence_artifacts=[
            EvidenceArtifact.from_path(FIXTURES / "artifact.txt"),
            EvidenceArtifact.from_path(decay, role="claim_decay"),
            EvidenceArtifact.from_path(extremal, role="extremal_scenario"),
        ]
    )

    Draft202012Validator(_schema()).validate(report)
    assert [artifact["role"] for artifact in report["evidence"]["artifacts"]] == [
        "artifact",
        "claim_decay",
        "extremal_scenario",
    ]
    assert report["receipt"]["canonical_hash"] != base["receipt"]["canonical_hash"]


def test_cli_accepts_decay_extremal_and_generic_evidence_roles(tmp_path: Path) -> None:
    decay = tmp_path / "decay.json"
    extremal = tmp_path / "extremal.json"
    out = tmp_path / "report.json"
    decay.write_text('{"schema_version":"cc.claim_decay.v1"}\n', encoding="utf-8")
    extremal.write_text('{"schema_version":"cc.extremal_scenario.v1"}\n', encoding="utf-8")
    args = _cli_args(out)
    claim_index = args.index("--claim")
    args[claim_index:claim_index] = [
        "--decay-policy",
        str(decay),
        "--extremal-scenario",
        str(extremal),
        "--evidence-role",
        "tests/fixtures/reporting/artifact.txt=calibration_source",
    ]

    result = _run_cli(args)

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    roles = [artifact["role"] for artifact in report["evidence"]["artifacts"]]
    assert roles == ["calibration_source", "claim_decay", "extremal_scenario"]


@pytest.mark.parametrize(
    "role_spec",
    [
        "decay.json",
        "=claim_decay",
        "decay.json=",
        "decay.json:claim_decay",
        "decay.json=bad role with spaces",
        "decay.json=BadRole",
        "decay.json=bad-role",
        " decay.json=claim_decay",
        "decay.json=claim_decay ",
    ],
)
def test_cli_rejects_invalid_evidence_role_specs(tmp_path: Path, role_spec: str) -> None:
    out = tmp_path / "report.json"
    args = _cli_args(out)
    claim_index = args.index("--claim")
    args[claim_index:claim_index] = ["--evidence-role", role_spec]

    result = _run_cli(args)

    assert result.returncode != 0
    assert "--evidence-role" in result.stderr


def test_cli_plain_evidence_keeps_default_artifact_role(tmp_path: Path) -> None:
    out = tmp_path / "report.json"

    result = _run_cli(_cli_args(out))

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["evidence"]["artifacts"][0]["role"] == "artifact"


def test_checked_in_example_report_is_valid() -> None:
    report = json.loads(
        (ROOT / "examples" / "reporting" / "minimal_cc_report.json").read_text(encoding="utf-8")
    )

    Draft202012Validator(_schema()).validate(report)

    assert report["receipt"]["canonical_hash"] == sha256_canonical(report)


def test_schema_contract_matches_report_builder_constants() -> None:
    schema = _schema()
    schema_claim_levels = tuple(
        option["const"] for option in schema["$defs"]["claim_level"]["oneOf"]
    )

    assert schema_claim_levels == CLAIM_LEVELS
    # The schema accepts both canonicalization profiles so pre-migration
    # receipts stay verifiable; the builder writes the first of them.
    schema_profiles = schema["properties"]["receipt"]["properties"]["canonicalization_method"][
        "enum"
    ]
    assert CANONICALIZATION_METHOD in schema_profiles
    assert schema_profiles[0] == CANONICALIZATION_METHOD
    assert len(schema_profiles) == 2

    report = _report()
    report["claim"]["non_claims"] = []
    errors = list(Draft202012Validator(schema).iter_errors(report))

    assert any(error.validator == "minItems" for error in errors)


def _report(
    *,
    measurement: MeasurementSummary | None = None,
    claim: ClaimSummary | None = None,
    evidence_artifacts: list[EvidenceArtifact] | None = None,
) -> dict[str, Any]:
    return build_cc_report(
        run=RunSummary(
            run_id="smoke-example",
            config_path="experiments/configs/smoke.yaml",
            config_hash="1" * 64,
            seed=123,
            command="fixture build-report",
        ),
        calibration=CalibrationSummary(
            target_fpr=0.05,
            alpha_cap=0.05,
            realized_fpr=0.04,
            calibration_window={"lower": 0.04, "upper": 0.06},
            threshold=0.13392857142857142,
            status="pass",
        ),
        measurement=measurement
        or MeasurementSummary(
            metric_family="CC",
            point_estimate=1.11,
            interval_lower=1.05,
            interval_upper=1.18,
            confidence_level=0.95,
            interval_method="FH-Bernstein",
            sample_sizes={"n1": 200, "n0": 200},
        ),
        claim=claim
        or ClaimSummary(
            statement=(
                "At the pinned operating point, the composed guardrail has a bounded "
                "empirical CC interval under the stated assumptions."
            ),
            allowed_claim_level="bounded_empirical",
            non_claims=[
                "This report does not certify production safety.",
                "This report does not generalize outside the named evaluation distribution.",
            ],
        ),
        evidence_artifacts=evidence_artifacts
        if evidence_artifacts is not None
        else [EvidenceArtifact.from_path(FIXTURES / "artifact.txt")],
        audit_log=EvidenceArtifact.from_path(FIXTURES / "audit.jsonl", role="audit_log"),
        figure_manifest=EvidenceArtifact.from_path(
            FIXTURES / "figure_manifest.json", role="figure_manifest"
        ),
        assumptions=[
            "The fixture data are treated as a fixed evaluation distribution.",
            "The calibration window is the named fixture operating window.",
        ],
        report_id="cc-report-smoke-example",
        created_at="2026-01-01T00:00:00Z",
        framework_version="0.3.1-fixture",
        git=GitMetadata(commit="0" * 40, dirty=False, branch="main"),
        environment=EnvironmentMetadata(
            python_version="3.12.0",
            platform="fixture-platform",
            dependency_hash="2" * 64,
        ),
    )


def _cli_args(out: Path) -> list[str]:
    return [
        "build-report",
        "--run-id",
        "smoke-example",
        "--measurement-json",
        "tests/fixtures/reporting/measurement.json",
        "--calibration-json",
        "tests/fixtures/reporting/calibration_summary.json",
        "--evidence",
        "tests/fixtures/reporting/artifact.txt",
        "--audit-log",
        "tests/fixtures/reporting/audit.jsonl",
        "--figure-manifest",
        "tests/fixtures/reporting/figure_manifest.json",
        "--claim",
        (
            "At the pinned operating point, the composed guardrail has a bounded "
            "empirical CC interval under the stated assumptions."
        ),
        "--claim-level",
        "bounded_empirical",
        "--non-claim",
        "This report does not certify production safety.",
        "--non-claim",
        "This report does not generalize outside the named evaluation distribution.",
        "--assumption",
        "The fixture data are treated as a fixed evaluation distribution.",
        "--assumption",
        "The calibration window is the named fixture operating window.",
        "--config-path",
        "experiments/configs/smoke.yaml",
        "--config-hash",
        "1" * 64,
        "--seed",
        "123",
        "--command",
        "fixture build-report",
        "--report-id",
        "cc-report-smoke-example",
        "--created-at",
        "2026-01-01T00:00:00Z",
        "--framework-version",
        "0.3.1-fixture",
        "--git-commit",
        "0" * 40,
        "--git-dirty",
        "false",
        "--git-branch",
        "main",
        "--python-version",
        "3.12.0",
        "--platform",
        "fixture-platform",
        "--dependency-hash",
        "2" * 64,
        "--out",
        str(out),
    ]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "-m", "cc.reporting.cli", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _schema() -> dict[str, Any]:
    return json.loads((ROOT / "schemas" / "cc_report.schema.json").read_text(encoding="utf-8"))
