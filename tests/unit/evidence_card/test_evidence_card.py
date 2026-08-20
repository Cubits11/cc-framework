"""Evidence cards: the object, and the things it refuses to be.

Most of these tests assert an *absence*. That is deliberate. The failure mode
this object exists to prevent is a surface that collapses three orthogonal
labels into one green checkmark, so the tests that matter most are the ones
asserting no such collapse is reachable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from cc.evidence_card import (
    EVIDENCE_STATES,
    LABEL_MEANINGS,
    PUBLICATION_STATES,
    VERDICTS,
    ArtifactRef,
    EvidenceCard,
    EvidenceCardError,
    cards_to_site_manifest,
    render_labels,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "schemas" / "cc.evidence_card.v1.json").read_text())
CARDS_DIR = ROOT / "evidence-cards" / "cards"
SITE_MANIFEST = ROOT / "evidence-cards" / "site-evidence-manifest.v1.json"


def _card(**overrides) -> EvidenceCard:
    base = {
        "card_id": "demo.claim",
        "claim": "The kernel computes classical Frechet-Hoeffding bounds.",
        "maturity": "C3",
        "source_revision": "abc123",
        "command": ("pytest tests/unit/kernel",),
        "falsifier": "A marginal configuration where the LP and the closed form disagree.",
        "assumptions": ("Exact marginals are supplied.",),
        "non_claims": ("Does not prove deployment safety.",),
        "evidence_state": "local-only",
        "verdict": "not-run",
        "publication_state": "draft",
    }
    base.update(overrides)
    return EvidenceCard(**base)


# --- the refusals that make a card evidence ----------------------------------


def test_a_card_without_a_falsifier_is_refused():
    """A claim no observation could refute is an assertion, not evidence."""
    with pytest.raises(EvidenceCardError, match="falsifier"):
        _card(falsifier="   ")


def test_a_card_without_a_command_is_refused():
    with pytest.raises(EvidenceCardError, match="at least one command"):
        _card(command=())


def test_a_card_without_a_non_claim_is_refused():
    with pytest.raises(EvidenceCardError, match="explicit non-claim"):
        _card(non_claims=())


@pytest.mark.parametrize("verdict", ["pass", "fail"])
def test_a_verdict_without_a_record_of_the_run_is_refused(verdict):
    """You cannot assert a result without saying what produced it."""
    with pytest.raises(EvidenceCardError, match="result_detail"):
        _card(verdict=verdict, result_detail=None)


def test_a_retraction_without_a_reason_is_refused():
    """Withdrawing a claim without saying why rewrites the record."""
    with pytest.raises(EvidenceCardError, match="retraction_reason"):
        _card(publication_state="retracted")


def test_a_superseded_card_must_name_its_successor():
    with pytest.raises(EvidenceCardError, match="supersedes"):
        _card(publication_state="superseded")


@pytest.mark.parametrize(
    ("label", "bad"),
    [
        ("evidence_state", "production"),
        ("verdict", "green"),
        ("verdict", "ok"),
        ("publication_state", "live"),
    ],
)
def test_labels_outside_their_declared_sets_are_refused(label, bad):
    with pytest.raises(EvidenceCardError, match=label):
        _card(**{label: bad})


# --- the collapse this object exists to prevent -------------------------------


def test_the_card_exposes_no_composite_status():
    """No `status`, `score`, `is_ok`, or `badge`.

    A single glyph lets a reader stop before asking any of the three questions
    the labels answer. If one is ever added, this test is the tripwire.
    """
    card = _card()
    for forbidden in (
        "status",
        "score",
        "is_ok",
        "ok",
        "badge",
        "health",
        "passing",
        "overall",
        "summary",
        "grade",
    ):
        assert not hasattr(card, forbidden), f"EvidenceCard grew a composite: {forbidden}"


def test_render_labels_returns_all_three_or_nothing():
    """There is no single-label renderer: a surface that can show one will show
    the flattering one."""
    rendered = render_labels(_card())
    assert set(rendered) == {"evidence_state", "verdict", "publication_state"}
    for entry in rendered.values():
        assert entry["value"]
        assert entry["meaning"]


def test_labels_are_independent_every_combination_is_constructible():
    """No label constrains another.

    A retracted card may still carry a passing verdict: the run happened, and
    the claim was withdrawn anyway. A local-only pass says nothing about
    deployment. If a future change couples them, this fails.
    """
    built = 0
    for evidence_state in EVIDENCE_STATES:
        for verdict in VERDICTS:
            for publication_state in PUBLICATION_STATES:
                card = _card(
                    evidence_state=evidence_state,
                    verdict=verdict,
                    publication_state=publication_state,
                    result_detail="ran: 6 passed" if verdict in ("pass", "fail") else None,
                    retraction_reason=(
                        "superseded by a sharper bound"
                        if publication_state == "retracted"
                        else None
                    ),
                    supersedes=("demo.claim.v2" if publication_state == "superseded" else None),
                )
                assert card.evidence_state == evidence_state
                assert card.verdict == verdict
                assert card.publication_state == publication_state
                built += 1
    assert built == len(EVIDENCE_STATES) * len(VERDICTS) * len(PUBLICATION_STATES)


def test_every_label_value_carries_a_meaning():
    """A bare token in a UI is an invitation to guess."""
    for label, values in (
        ("evidence_state", EVIDENCE_STATES),
        ("verdict", VERDICTS),
        ("publication_state", PUBLICATION_STATES),
    ):
        for value in values:
            assert LABEL_MEANINGS[label][value].strip()


def test_not_run_is_the_default_verdict_of_the_generator():
    """A pass must be earned by running something."""
    assert _card().verdict == "not-run"


# --- serialization ------------------------------------------------------------


def test_to_json_keeps_the_three_labels_separate():
    payload = _card().to_json()
    assert set(payload["labels"]) == {"evidence_state", "verdict", "publication_state"}
    # No top-level flattened alias may appear beside them.
    for forbidden in ("status", "score", "badge", "ok", "passing"):
        assert forbidden not in payload


def test_to_json_round_trips():
    payload = _card().to_json()
    assert json.loads(json.dumps(payload)) == payload


def test_missing_artifacts_are_recorded_not_skipped():
    """A claim pointing at a file that is gone is a finding, not an absence."""
    card = _card(artifacts=(ArtifactRef(path="docs/gone.md", missing=True),))
    artifact = card.to_json()["artifacts"][0]
    assert artifact["missing"] is True
    assert artifact["sha256"] is None


# --- the site manifest --------------------------------------------------------


def test_site_manifest_counts_per_label_and_never_aggregates():
    cards = [
        _card(card_id="a", verdict="not-run"),
        _card(card_id="b", verdict="pass", result_detail="6 passed"),
        _card(card_id="c", evidence_state="aws-synth-only", verdict="unverifiable"),
    ]
    site = cards_to_site_manifest(cards, source_revision="abc", generated_note="test")

    assert site["card_count"] == 3
    assert site["counts_by_label"]["verdict"]["pass"] == 1
    assert site["counts_by_label"]["verdict"]["not-run"] == 1
    assert site["counts_by_label"]["evidence_state"]["aws-synth-only"] == 1
    # No aggregate anywhere.
    for forbidden in ("score", "passing_rate", "health", "overall", "summary_status"):
        assert forbidden not in site


def test_site_manifest_states_that_its_counts_are_not_a_score():
    site = cards_to_site_manifest([_card()], source_revision="abc", generated_note="n")
    joined = " ".join(site["non_claims"]).lower()
    assert "orthogonal" in joined
    assert "no aggregate score" in joined
    assert "not-run" in joined


# --- the committed artifacts --------------------------------------------------


def test_schema_is_valid_and_defines_no_aggregate():
    Draft202012Validator.check_schema(SCHEMA)
    properties = set(SCHEMA["properties"])
    for forbidden in ("status", "score", "badge", "health", "overall"):
        assert forbidden not in properties


@pytest.mark.parametrize("path", sorted(CARDS_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_committed_cards_validate_against_the_schema(path):
    errors = list(Draft202012Validator(SCHEMA).iter_errors(json.loads(path.read_text())))
    assert not errors, [e.message for e in errors[:3]]


def test_committed_cards_are_current():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_evidence_cards.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_committed_cards_claim_no_results():
    """The committed scaffold must never carry a verdict from someone's laptop."""
    site = json.loads(SITE_MANIFEST.read_text())
    verdicts = site["counts_by_label"]["verdict"]
    assert verdicts["not-run"] == site["card_count"]
    assert verdicts["pass"] == 0
    assert verdicts["fail"] == 0


def test_every_committed_card_has_a_falsifier_and_a_non_claim():
    for path in sorted(CARDS_DIR.glob("*.json")):
        card = json.loads(path.read_text())
        assert card["falsifier"].strip(), path.name
        assert card["non_claims"], path.name
        assert card["assumptions"], path.name


def test_the_synthetic_aws_lane_is_labelled_as_such():
    """Synthetic evidence is never live evidence, and the label must say so."""
    card = json.loads((CARDS_DIR / "enterprise.reference_v0_1.json").read_text())
    assert card["labels"]["evidence_state"] == "aws-synth-only"
    assert "never live evidence" in " ".join(card["assumptions"]).lower()
