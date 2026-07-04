from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cc.evidence.assurance_schema import (
    ClaimCategory,
    Defeater,
    EvidenceRole,
    ReviewStatus,
    SubClaim,
    assurance_case_from_run,
    export_assurance_case_jsonld,
    export_assurance_case_markdown,
    write_assurance_case_exports,
)


def test_claim_nodes_must_make_defeater_position_explicit() -> None:
    with pytest.raises(ValidationError, match="defeaters"):
        SubClaim(
            id="claim-no-defeater-field",
            statement="A claim that silently omits defeaters is invalid.",
            category=ClaimCategory.COMPOSITION_RISK_BOUNDED,
        )

    with pytest.raises(ValidationError, match="no_defeaters_justification"):
        SubClaim(
            id="claim-empty-defeaters",
            statement="A claim with no defeaters needs an explicit justification.",
            category=ClaimCategory.COMPOSITION_RISK_BOUNDED,
            defeaters=[],
        )

    claim = SubClaim(
        id="claim-justified-empty-defeaters",
        statement="A reviewer explicitly justified why no defeaters apply.",
        category=ClaimCategory.COMPOSITION_RISK_BOUNDED,
        defeaters=[],
        no_defeaters_justification="Reviewed against the current hazard log.",
    )

    assert claim.no_defeaters_justification


def test_assurance_case_from_run_groups_actual_outputs_by_claim_category() -> None:
    bundle = {
        "run_id": "run-123",
        "metrics": {
            "composition": "any_block",
            "fh_bounds": {"lower": 0.12, "upper": 0.24, "event": "all_fail"},
            "cliff_certificate": {
                "regime": "critical",
                "lambda_hat": 0.19,
                "critical_value": 0.2,
                "falsifier": "A tighter replication would move the certificate.",
            },
            "ccf_consistency_check": {
                "within_fh_envelope": True,
                "point_estimate": 0.17,
            },
            "coverage_results": [
                {
                    "nominal_coverage": 0.95,
                    "cluster_bootstrap_coverage": 0.94,
                    "within_tolerance": True,
                }
            ],
        },
        "evidence": {
            "artifacts": [
                {
                    "path": "decay.json",
                    "sha256": "a" * 64,
                    "bytes": 123,
                    "role": "claim_decay",
                },
                {
                    "path": "extremal.json",
                    "sha256": "b" * 64,
                    "bytes": 456,
                    "role": "extremal_scenario",
                },
            ]
        },
    }

    case = assurance_case_from_run(bundle)
    subclaims = {claim.category: claim for claim in case.top_claim.subclaims}

    assert case.run_id == "run-123"
    assert case.top_claim.review_status is ReviewStatus.NEEDS_HUMAN_REVIEW
    assert all(
        defeater.review_status is ReviewStatus.NEEDS_HUMAN_REVIEW
        for defeater in case.top_claim.defeaters
    )
    assert subclaims[ClaimCategory.COMPOSITION_RISK_BOUNDED].evidence
    assert subclaims[ClaimCategory.DEPENDENCE_STRUCTURE_CHARACTERIZED].evidence
    assert subclaims[ClaimCategory.UNCERTAINTY_HONESTLY_QUANTIFIED].evidence
    evidence_roles = {
        evidence.role for claim in case.top_claim.subclaims for evidence in claim.evidence
    }
    assert EvidenceRole.CLAIM_DECAY in evidence_roles
    assert EvidenceRole.EXTREMAL_SCENARIO in evidence_roles

    jsonld = export_assurance_case_jsonld(case)
    markdown = export_assurance_case_markdown(case)

    assert jsonld["@context"]
    assert any(node.get("@type") == "Evidence" for node in jsonld["@graph"])
    assert "NEEDS HUMAN REVIEW" in markdown
    assert "composition_risk_bounded" in markdown


def test_write_assurance_case_exports(tmp_path: Path) -> None:
    case = assurance_case_from_run(
        {
            "run_id": "export-test",
            "metrics": {
                "fh_bounds": {"lower": 0.1, "upper": 0.2},
                "coverage_results": {"nominal_coverage": 0.95},
            },
        }
    )

    paths = write_assurance_case_exports(case, tmp_path)

    jsonld = json.loads(Path(paths["jsonld"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert jsonld["@type"] == "AssuranceCase"
    assert "Assurance Case" in markdown


def test_claim_decay_does_not_clear_statistical_coverage_gap_or_review_status() -> None:
    case = assurance_case_from_run(
        {
            "run_id": "decay-only",
            "evidence": {
                "artifacts": [
                    {
                        "path": "decay.json",
                        "sha256": "a" * 64,
                        "bytes": 123,
                        "role": "claim_decay",
                    }
                ]
            },
        }
    )
    uncertainty = {claim.category: claim for claim in case.top_claim.subclaims}[
        ClaimCategory.UNCERTAINTY_HONESTLY_QUANTIFIED
    ]

    assert any(evidence.role is EvidenceRole.CLAIM_DECAY for evidence in uncertainty.evidence)
    assert any(
        "No statistical coverage" in defeater.description for defeater in uncertainty.defeaters
    )
    assert uncertainty.review_status is ReviewStatus.NEEDS_HUMAN_REVIEW
    assert all(
        evidence.review_status is ReviewStatus.AUTO_POPULATED for evidence in uncertainty.evidence
    )
    assert all(evidence.human_review_required is True for evidence in uncertainty.evidence)
    assert "does not prove claim validity" in uncertainty.evidence[0].description


def test_extremal_scenario_does_not_clear_fh_gap_or_review_status() -> None:
    case = assurance_case_from_run(
        {
            "run_id": "extremal-only",
            "evidence": {
                "artifacts": [
                    {
                        "path": "extremal.json",
                        "sha256": "b" * 64,
                        "bytes": 456,
                        "role": "extremal_scenario",
                    }
                ]
            },
        }
    )
    composition = {claim.category: claim for claim in case.top_claim.subclaims}[
        ClaimCategory.COMPOSITION_RISK_BOUNDED
    ]

    assert any(evidence.role is EvidenceRole.EXTREMAL_SCENARIO for evidence in composition.evidence)
    assert any(
        "No FH/Frechet-Hoeffding" in defeater.description for defeater in composition.defeaters
    )
    assert composition.review_status is ReviewStatus.NEEDS_HUMAN_REVIEW
    assert all(
        evidence.review_status is ReviewStatus.AUTO_POPULATED for evidence in composition.evidence
    )
    assert all(evidence.human_review_required is True for evidence in composition.evidence)
    assert "does not prove deployment safety" in composition.evidence[0].description


def test_human_review_defeater_defaults_are_explicit() -> None:
    defeater = Defeater(id="defeater-1", description="A reviewer must resolve this challenge.")

    assert defeater.review_status is ReviewStatus.NEEDS_HUMAN_REVIEW
    assert defeater.human_review_required is True
