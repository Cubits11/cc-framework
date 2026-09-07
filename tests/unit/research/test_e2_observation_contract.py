"""E2-P -- the measurement contract must be enforceable, not merely written.

The preregistration's exit criterion is that another researcher could build a
conforming dataset without asking what any field means.  That is only true if
non-conformance is detected mechanically, so these tests pin the failures the
contract exists to prevent -- above all unpaired items, which entangle observed
dependence with item composition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_e2_observations import (  # noqa: E402
    audit,
    design_violations,
    load_rows,
    schema_violations,
    warnings_from,
)

SCHEMA = json.loads(
    (REPO_ROOT / "schemas" / "cc.e2_observation_row.v1.json").read_text(encoding="utf-8")
)
EXAMPLE = REPO_ROOT / "examples" / "e2" / "observations.example.jsonl"


def _row(**overrides: Any) -> dict[str, Any]:
    digest = "a" * 64
    row: dict[str, Any] = {
        "schema_version": "cc.e2_observation_row.v1",
        "study_id": "S",
        "run_id": "R",
        "item_id": "item-1",
        "item_source": "src",
        "item_source_version": "v1",
        "item_hash": digest,
        "guardrail_id": "gr.a",
        "guardrail_version": "1.0",
        "policy_version": "p1",
        "configuration_hash": digest,
        "shared_dependency_group": None,
        "raw_input_hash": digest,
        "raw_output_hash": digest,
        "raw_outcome": "allow",
        "normalized_outcome": 1,
        "normalizer_version": "h.v1",
        "execution_timestamp": "2026-08-21T10:00:00Z",
        "execution_status": "success",
        "missingness_code": "observed",
        "predeclared_exclusion": False,
        "exclusion_reason": None,
        "replicate_id": 0,
    }
    row.update(overrides)
    return row


# -- the shipped example is the reference a researcher copies -----------------


def test_example_dataset_is_conformant() -> None:
    rows = load_rows(EXAMPLE)
    assert schema_violations(rows, SCHEMA) == []
    assert design_violations(rows) == []


def test_example_dataset_still_reports_its_limitations() -> None:
    """Conformance is not a clean bill of health."""

    notes = warnings_from(audit(load_rows(EXAMPLE)))
    joined = " ".join(notes)
    assert "upstream_moderation" in joined
    assert "shared dependency" in joined
    assert "duplicate item content" in joined
    assert "unknown" in joined


# -- layer B: shared-item pairing --------------------------------------------


def test_unpaired_items_are_a_violation() -> None:
    """A guardrail silently missing an item is the contract's core failure."""

    rows = [
        _row(item_id="item-1", guardrail_id="gr.a"),
        _row(item_id="item-2", guardrail_id="gr.a"),
        _row(item_id="item-1", guardrail_id="gr.b"),
    ]
    problems = design_violations(rows)
    assert any("shared-item violation" in problem for problem in problems), problems
    assert any("gr.b" in problem for problem in problems), problems


def test_recorded_missingness_satisfies_pairing() -> None:
    """Absence must be recorded as a row, not an omission -- and then it passes."""

    rows = [
        _row(item_id="item-1", guardrail_id="gr.a"),
        _row(item_id="item-2", guardrail_id="gr.a"),
        _row(item_id="item-1", guardrail_id="gr.b"),
        _row(
            item_id="item-2",
            guardrail_id="gr.b",
            execution_status="timeout",
            missingness_code="timeout",
            raw_output_hash=None,
            raw_outcome=None,
            normalized_outcome=None,
            normalizer_version=None,
        ),
    ]
    assert design_violations(rows) == []
    assert schema_violations(rows, SCHEMA) == []


def test_one_guardrail_cannot_form_an_e2_dependence_study() -> None:
    problems = design_violations([_row()])
    assert any("at least two guardrails" in problem for problem in problems)


# -- layer C: the reduction h must be frozen ---------------------------------


def test_normalizer_drift_is_a_violation() -> None:
    rows = [_row(), _row(item_id="item-2", normalizer_version="h.v2")]
    assert any("normalizer_version is not frozen" in p for p in design_violations(rows))


def test_duplicate_observations_are_a_violation() -> None:
    rows = [_row(), _row()]
    assert any("duplicate observation" in p for p in design_violations(rows))


# -- schema conditionals ------------------------------------------------------


def test_observed_rows_must_carry_a_decision() -> None:
    """An 'observed' row with no outcome is incoherent."""

    rows = [_row(normalized_outcome=None, raw_outcome=None)]
    assert schema_violations(rows, SCHEMA)


@pytest.mark.parametrize("code", ["timeout", "execution_error", "upstream_moderation"])
def test_unobserved_rows_must_not_fabricate_a_decision(code: str) -> None:
    rows = [_row(missingness_code=code, execution_status="error", normalized_outcome=1)]
    assert schema_violations(rows, SCHEMA)


def test_exclusions_must_state_a_reason() -> None:
    rows = [
        _row(
            missingness_code="excluded_post",
            execution_status="skipped",
            raw_outcome=None,
            raw_output_hash=None,
            normalized_outcome=None,
            normalizer_version=None,
            exclusion_reason=None,
        )
    ]
    assert schema_violations(rows, SCHEMA)


def test_predeclared_exclusion_cannot_relabel_a_post_hoc_drop() -> None:
    """`predeclared_exclusion` is reserved for rules frozen before outcomes."""

    rows = [
        _row(
            predeclared_exclusion=True,
            missingness_code="excluded_post",
            execution_status="skipped",
            raw_outcome=None,
            raw_output_hash=None,
            normalized_outcome=None,
            normalizer_version=None,
            exclusion_reason="dropped after seeing results",
        )
    ]
    assert schema_violations(rows, SCHEMA)


def test_excluded_pre_must_affirm_predeclared_exclusion() -> None:
    rows = [
        _row(
            predeclared_exclusion=False,
            missingness_code="excluded_pre",
            execution_status="skipped",
            raw_outcome=None,
            raw_output_hash=None,
            normalized_outcome=None,
            normalizer_version=None,
            exclusion_reason="frozen eligibility rule",
        )
    ]
    assert schema_violations(rows, SCHEMA)


def test_timeout_cannot_be_recorded_as_success() -> None:
    rows = [
        _row(
            missingness_code="timeout",
            execution_status="success",
            raw_outcome=None,
            raw_output_hash=None,
            normalized_outcome=None,
            normalizer_version=None,
        )
    ]
    assert schema_violations(rows, SCHEMA)


# -- audit surface ------------------------------------------------------------


def test_replicate_disagreement_is_surfaced() -> None:
    rows = [_row(replicate_id=0, normalized_outcome=1), _row(replicate_id=1, normalized_outcome=0)]
    assert audit(rows)["replicate_unstable_pairs"] == [["item-1", "gr.a"]]
    assert any("stochastic" in note for note in warnings_from(audit(rows)))


def test_complete_case_fraction_tracks_missingness() -> None:
    rows = [
        _row(item_id="item-1", guardrail_id="gr.a"),
        _row(item_id="item-1", guardrail_id="gr.b"),
        _row(item_id="item-2", guardrail_id="gr.a"),
        _row(
            item_id="item-2",
            guardrail_id="gr.b",
            execution_status="timeout",
            missingness_code="timeout",
            raw_output_hash=None,
            raw_outcome=None,
            normalized_outcome=None,
            normalizer_version=None,
        ),
    ]
    report = audit(rows)
    assert report["complete_case_items"] == 1
    assert report["items"] == 2
    assert report["complete_case_fraction"] == 0.5
