"""Evidence cards: the unit an Atlas displays, and the thing it refuses to be.

An evidence card is a claim together with everything a reader needs in order to
disagree with it: how it was operationalized, which command produces it, what
would falsify it, what it assumes, and what it explicitly does not claim.

The design constraint that shapes this module is negative. A card is **not** a
trust score, and no function here computes one. Three labels travel with every
card and they are deliberately *orthogonal*:

``evidence_state``
    Where the evidence came from -- ``local-only``, ``aws-synth-only``,
    ``aws-live``, ``illustrative``.
``verdict``
    What happened when the command ran -- ``pass``, ``fail``, ``unverifiable``,
    ``not-run``.
``publication_state``
    Where the card is in its own lifecycle -- ``draft``, ``released``,
    ``superseded``, ``retracted``.

They answer different questions and none implies another. A ``pass`` that is
``local-only`` says nothing about deployed behaviour. A ``retracted`` card may
still carry a ``pass``: the run happened, and the claim was withdrawn anyway.
Collapsing them into one green checkmark is exactly the failure this object
exists to prevent, so :class:`EvidenceCard` provides no composite status
property, and :func:`render_labels` returns all three or raises.

Two fields are required that most claim registries treat as optional:

* ``falsifier`` -- what observation would show the claim is wrong. A claim with
  no falsifier is not evidence; it is an assertion, and the emitter refuses it.
* ``non_claims`` -- what the card explicitly does not establish.

The default verdict is ``not-run``. A ``pass`` is only reachable by executing
the command and recording the result, so a card cannot acquire a passing verdict
by being written confidently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

__all__ = [
    "EVIDENCE_STATES",
    "PUBLICATION_STATES",
    "SCHEMA_VERSION",
    "VERDICTS",
    "ArtifactRef",
    "EvidenceCard",
    "EvidenceCardError",
    "EvidenceState",
    "PublicationState",
    "Verdict",
    "render_labels",
]

SCHEMA_VERSION: Final = "cc.evidence_card.v1"

#: Where the evidence came from. This is a statement about provenance, not
#: about quality: ``aws-live`` evidence can record a failure, and
#: ``local-only`` evidence can be entirely sound for a local claim.
EvidenceState: TypeAlias = Literal[
    "local-only",
    "aws-synth-only",
    "aws-live",
    "illustrative",
]

#: What happened when the command ran. ``not-run`` is the default and the only
#: honest value for a card whose command has not been executed. ``unverifiable``
#: is distinct from ``fail``: the check could not be performed at all, which is
#: a different fact about the world than the check performing and failing.
Verdict: TypeAlias = Literal["pass", "fail", "unverifiable", "not-run"]

#: Where the card is in its own lifecycle. Retraction is a first-class state,
#: not a deletion: a withdrawn claim stays visible with its reason.
PublicationState: TypeAlias = Literal["draft", "released", "superseded", "retracted"]

EVIDENCE_STATES: Final[tuple[EvidenceState, ...]] = (
    "local-only",
    "aws-synth-only",
    "aws-live",
    "illustrative",
)
VERDICTS: Final[tuple[Verdict, ...]] = ("pass", "fail", "unverifiable", "not-run")
PUBLICATION_STATES: Final[tuple[PublicationState, ...]] = (
    "draft",
    "released",
    "superseded",
    "retracted",
)

#: Human-readable gloss for each label value. The Atlas shows these next to the
#: value so a reader never has to infer what a bare token means.
LABEL_MEANINGS: Final[Mapping[str, Mapping[str, str]]] = {
    "evidence_state": {
        "local-only": "Produced on a local machine. Says nothing about cloud or "
        "deployed behaviour.",
        "aws-synth-only": "Produced against synthesized or emulated AWS, not a "
        "live account. Synthetic evidence is never live evidence.",
        "aws-live": "Produced against a live AWS account, for the named run only.",
        "illustrative": "A worked example or figure. Not measured evidence, and "
        "must never be cited as a result.",
    },
    "verdict": {
        "pass": "The command ran and its checks succeeded. Scope is whatever the "
        "command covers -- no wider.",
        "fail": "The command ran and its checks did not succeed.",
        "unverifiable": "The check could not be performed. This is not a pass and not a fail.",
        "not-run": "The command has not been executed for this card. No result is claimed.",
    },
    "publication_state": {
        "draft": "Not yet reviewed for publication.",
        "released": "Published. Says nothing about whether the claim is true, only "
        "that it was reviewed for release.",
        "superseded": "A later card replaces this one. Retained so the record is not rewritten.",
        "retracted": "Withdrawn. The reason is on the card; the card is not deleted.",
    },
}


class EvidenceCardError(ValueError):
    """Raised when a card is missing something that makes it evidence."""


@dataclass(frozen=True)
class ArtifactRef:
    """A file the claim rests on, bound by digest.

    ``sha256`` may be ``None`` only when ``missing`` is true -- a path named by a
    claim that does not exist on disk. That is recorded rather than skipped: a
    claim pointing at a file that is gone is a finding, not an absence.
    """

    path: str
    sha256: str | None = None
    bytes_len: int | None = None
    missing: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes_len,
            "missing": self.missing,
        }


@dataclass(frozen=True)
class EvidenceCard:
    """One claim, with everything needed to disagree with it.

    Raises:
        EvidenceCardError: A required field is empty, or a label carries a value
            outside its declared set. Construction is the enforcement point --
            an invalid card cannot exist to be rendered.
    """

    card_id: str
    claim: str
    maturity: str
    source_revision: str
    command: tuple[str, ...]
    falsifier: str
    assumptions: tuple[str, ...]
    non_claims: tuple[str, ...]
    evidence_state: EvidenceState
    verdict: Verdict
    publication_state: PublicationState
    artifacts: tuple[ArtifactRef, ...] = ()
    result_detail: str | None = None
    counterevidence: tuple[str, ...] = ()
    supersedes: str | None = None
    retraction_reason: str | None = None
    schema_version: str = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        for name in ("card_id", "claim", "maturity", "source_revision", "falsifier"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvidenceCardError(
                    f"{self.card_id or '<unnamed>'}: {name} must be a non-empty string"
                )

        if not self.command:
            raise EvidenceCardError(
                f"{self.card_id}: a card needs at least one command. A claim no "
                "command can demonstrate is an assertion, not evidence."
            )
        if not self.non_claims:
            raise EvidenceCardError(
                f"{self.card_id}: a card needs at least one explicit non-claim. "
                "Stating what a result does not establish is the point."
            )

        for label, allowed in (
            ("evidence_state", EVIDENCE_STATES),
            ("verdict", VERDICTS),
            ("publication_state", PUBLICATION_STATES),
        ):
            value = getattr(self, label)
            if value not in allowed:
                raise EvidenceCardError(
                    f"{self.card_id}: {label}={value!r} is not one of {allowed}"
                )

        if self.publication_state == "retracted" and not (self.retraction_reason or "").strip():
            raise EvidenceCardError(
                f"{self.card_id}: a retracted card must carry its retraction_reason. "
                "Withdrawing a claim without saying why rewrites the record."
            )
        if self.publication_state == "superseded" and not (self.supersedes or "").strip():
            raise EvidenceCardError(
                f"{self.card_id}: a superseded card must set 'supersedes' to name "
                "the card that replaces it."
            )
        if self.verdict in ("pass", "fail") and not (self.result_detail or "").strip():
            raise EvidenceCardError(
                f"{self.card_id}: verdict={self.verdict!r} requires result_detail "
                "naming what actually ran. A verdict without a record of the run "
                "is an assertion."
            )

    # NOTE: there is deliberately no `status`, `score`, `is_ok`, or `badge`
    # property on this class. The three labels answer different questions and a
    # single composite would let a reader stop before asking any of them. If a
    # caller wants one glyph, they must choose which question they are asking.

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-native card. Every label is present and separate."""
        return {
            "schema_version": self.schema_version,
            "card_id": self.card_id,
            "claim": self.claim,
            "maturity": self.maturity,
            "source_revision": self.source_revision,
            "command": list(self.command),
            "artifacts": [artifact.to_json() for artifact in self.artifacts],
            "labels": {
                "evidence_state": self.evidence_state,
                "verdict": self.verdict,
                "publication_state": self.publication_state,
            },
            "label_meanings": {
                "evidence_state": LABEL_MEANINGS["evidence_state"][self.evidence_state],
                "verdict": LABEL_MEANINGS["verdict"][self.verdict],
                "publication_state": LABEL_MEANINGS["publication_state"][self.publication_state],
            },
            "result_detail": self.result_detail,
            "falsifier": self.falsifier,
            "counterevidence": list(self.counterevidence),
            "assumptions": list(self.assumptions),
            "non_claims": list(self.non_claims),
            "supersedes": self.supersedes,
            "retraction_reason": self.retraction_reason,
        }


def render_labels(card: EvidenceCard) -> dict[str, dict[str, str]]:
    """Return all three labels with their meanings, for display.

    This is the only rendering helper provided, and it returns **three**
    entries. There is no single-label variant, because a surface that can show
    one label will show the flattering one.
    """
    return {
        label: {
            "value": getattr(card, label),
            "meaning": LABEL_MEANINGS[label][getattr(card, label)],
        }
        for label in ("evidence_state", "verdict", "publication_state")
    }


def cards_to_site_manifest(
    cards: Sequence[EvidenceCard],
    *,
    source_revision: str,
    generated_note: str,
) -> dict[str, Any]:
    """Bundle cards into the manifest a static Atlas consumes.

    The manifest carries counts *per label*, never an aggregate. A reader
    learns how many cards are ``not-run`` and how many are ``local-only``
    separately, because a site that reported "7/8 passing" would be describing
    something nobody measured.
    """
    by_label: dict[str, dict[str, int]] = {
        "evidence_state": dict.fromkeys(EVIDENCE_STATES, 0),
        "verdict": dict.fromkeys(VERDICTS, 0),
        "publication_state": dict.fromkeys(PUBLICATION_STATES, 0),
    }
    for card in cards:
        for label in by_label:
            by_label[label][getattr(card, label)] += 1

    return {
        "schema_version": "cc.site_evidence_manifest.v1",
        "source_revision": source_revision,
        "generated_note": generated_note,
        "card_count": len(cards),
        "counts_by_label": by_label,
        "label_meanings": {k: dict(v) for k, v in LABEL_MEANINGS.items()},
        "cards": [card.to_json() for card in cards],
        "non_claims": [
            "This manifest lists claims and the commands that test them. It does "
            "not establish that any claim is true.",
            "The three labels are orthogonal. A passing verdict on local-only "
            "evidence says nothing about deployed behaviour, and a released "
            "publication state says only that a card was reviewed for release.",
            "Counts are reported per label. There is no aggregate score, and any "
            "surface that computes one from this file is misusing it.",
            "A card with verdict 'not-run' has no result. It must not be "
            "displayed as passing, pending, or healthy.",
        ],
    }
