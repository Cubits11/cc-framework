"""Evidence cards -- the unit an Evidence Atlas displays.

A card is a claim plus everything needed to disagree with it. See
:mod:`cc.evidence_card._card` for the design note, in particular why there is no
composite status property.
"""

from __future__ import annotations

from cc.evidence_card._card import (
    EVIDENCE_STATES,
    LABEL_MEANINGS,
    PUBLICATION_STATES,
    SCHEMA_VERSION,
    VERDICTS,
    ArtifactRef,
    EvidenceCard,
    EvidenceCardError,
    EvidenceState,
    PublicationState,
    Verdict,
    cards_to_site_manifest,
    render_labels,
)

__all__ = [
    "EVIDENCE_STATES",
    "LABEL_MEANINGS",
    "PUBLICATION_STATES",
    "SCHEMA_VERSION",
    "VERDICTS",
    "ArtifactRef",
    "EvidenceCard",
    "EvidenceCardError",
    "EvidenceState",
    "PublicationState",
    "Verdict",
    "cards_to_site_manifest",
    "render_labels",
]
