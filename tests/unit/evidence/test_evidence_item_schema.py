from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path("schemas/evidence/evidence-item.schema.json")
HEX = "a" * 64
URN = f"urn:sha256:{HEX}"


def _schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _actor(actor_id: str = "ev:governance:actors:person:reviewer-a") -> dict[str, str]:
    return {
        "actor_id": actor_id,
        "actor_type": "person",
        "display_name": "Reviewer A",
    }


def _digest(value: str = HEX) -> dict[str, str]:
    return {
        "algorithm": "SHA-256",
        "value": value,
        "urn": f"urn:sha256:{value}",
    }


def _evidence_item() -> dict[str, object]:
    non_claims = [
        "This evidence item is not a deployment approval.",
        "This evidence item does not prove deployment safety.",
        "Receipt integrity is not statistical validity or truth.",
        "Human review does not upgrade or modify confidence.",
    ]
    return {
        "schema_version": "cc.evidence_item.v1",
        "id": URN,
        "namespace": "ev:ai:assurance:evidence:item-a",
        "custody_status": "admitted",
        "evidence_class": "model_eval_artifact",
        "non_claims": non_claims,
        "provenance": {
            "prov_entity_id": "ev:ai:assurance:entity:item-a",
            "source_uri": "https://example.org/evidence/item-a",
            "source_type": "repository",
            "captured_at": "2026-07-05T20:00:00Z",
            "author": {
                "display_name": "Example Lab",
                "actor_type": "organization",
            },
            "collector": _actor("ev:governance:actors:system:ingest-bot"),
            "license_assertion": "internal-review",
        },
        "acquisition": {
            "submitted_at": "2026-07-05T20:01:00Z",
            "submitted_by": _actor("ev:governance:actors:person:submitter-a"),
            "submission_channel": "cli",
            "package_format": "bagit",
            "submission_manifest_hash": _digest(),
        },
        "extraction": {
            "mode": "automated",
            "method": "structured-parser",
            "performed_at": "2026-07-05T20:02:00Z",
            "extractor_id": "parser-v1",
            "pipeline_version": "2026.07.05",
            "quality_score": 0.95,
        },
        "content": {
            "media_type": "application/json",
            "language": "en",
            "title": "Bounded evaluation artifact",
            "abstract": "A scoped evaluation artifact for bounded evidence review.",
            "text": "Observed rates are scoped to the declared evaluation run.",
            "claim_fragments": [
                {
                    "fragment_id": "frag-main",
                    "char_start": 0,
                    "char_end": 64,
                    "text": "The observed interval is bounded by the declared assumptions.",
                }
            ],
            "byte_size": 4096,
            "storage_refs": {
                "raw_object_ref": "storage/tiers/raw/item-a.json",
                "normalized_text_ref": "storage/tiers/normalized/item-a.txt",
            },
        },
        "integrity": {
            "primary_hash": _digest(),
            "canonical_json_hash": _digest("b" * 64),
            "nfc_normalized_content_hash": _digest("c" * 64),
            "canonicalization_profile": (
                "NFC + canonical JSON with sorted keys, compact separators, UTF-8, "
                "and allow_nan=false"
            ),
            "signature_status": "verified",
            "dedupe_status": "unique",
        },
        "epistemic": {
            "confidence_score": 0.916,
            "confidence_formula": "round(0.24P + 0.24I + 0.18R + 0.16C + 0.10T + 0.08E, 3)",
            "component_scores": {
                "provenance_score": 0.95,
                "integrity_score": 1.0,
                "source_reliability_score": 0.9,
                "corroboration_score": 0.75,
                "temporal_validity_score": 0.9,
                "extraction_quality_score": 0.95,
            },
            "evidence_tier": "tier-1",
            "source_reliability": 0.9,
            "corroboration_count": 2,
            "conflict_state": "none",
            "confidence_rationale": "Confidence follows the declared score components.",
            "receipt_integrity_caveat": (
                "Integrity checks bind bytes and provenance continuity, not statistical validity, "
                "truth, or deployment safety."
            ),
        },
        "domain_tags": ["ev:ai:assurance:domain:guardrails"],
        "chain_of_custody": [
            {
                "event_index": 0,
                "event_type": "acquired",
                "performed_at": "2026-07-05T20:01:00Z",
                "actor": _actor("ev:governance:actors:system:ingest-bot"),
                "object_hash": URN,
                "previous_event_hash": None,
                "event_payload_canonical_hash": f"urn:sha256:{'d' * 64}",
                "event_hash": f"urn:sha256:{'e' * 64}",
                "hash_rule": (
                    "event_hash = SHA256(canonical_event_n || previous_event_hash) "
                    "over NFC-normalized canonical JSON"
                ),
                "reason": "Initial acquisition event.",
            }
        ],
        "audit": {
            "schema_validated_at": "2026-07-05T20:03:00Z",
            "validator_id": "jsonschema-draft-2020-12",
            "policy_bundle_hash": f"urn:sha256:{'f' * 64}",
            "receipt_caveats_appended": True,
        },
    }


def test_evidence_item_schema_accepts_representative_item() -> None:
    errors = sorted(_validator().iter_errors(_evidence_item()), key=lambda error: error.path)

    assert errors == []


def test_evidence_item_schema_requires_non_claims() -> None:
    item = _evidence_item()
    item["non_claims"] = []

    errors = list(_validator().iter_errors(item))

    assert any(
        "is too short" in error.message or "does not contain" in error.message for error in errors
    )


def test_evidence_item_schema_rejects_overclaiming_fragment_text() -> None:
    item = deepcopy(_evidence_item())
    fragments = item["content"]["claim_fragments"]  # type: ignore[index]
    fragments[0]["text"] = "This result proves the model is safe for deployment."  # type: ignore[index]

    errors = list(_validator().iter_errors(item))

    assert any("should not be valid" in error.message for error in errors)
