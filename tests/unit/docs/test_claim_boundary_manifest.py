from __future__ import annotations

from scripts.validate_claim_boundary_manifest import CLAIM_REQUIRED, load_manifest, validate_manifest


MAJOR_FORBIDDEN_UPGRADES = {
    ("Governance PASS", "deployment safety"),
    ("Receipt/hash integrity", "statistical validity"),
    ("JSON schema validity", "semantic truth"),
    ("Human review", "stronger empirical evidence"),
    ("Exploratory red-team discovery", "confirmatory proof"),
}


def test_claim_boundary_manifest_parses_and_validates() -> None:
    manifest = load_manifest()

    assert validate_manifest(manifest) == []


def test_claim_boundary_manifest_required_fields_and_unique_ids() -> None:
    manifest = load_manifest()
    seen: set[str] = set()

    for claim in manifest["claims"]:
        assert CLAIM_REQUIRED <= claim.keys()
        assert claim["id"] not in seen
        seen.add(claim["id"])
        assert claim["non_claims"]


def test_claim_boundary_manifest_contains_major_forbidden_upgrades() -> None:
    manifest = load_manifest()
    upgrades = {(item["from"], item["to"]) for item in manifest["forbidden_upgrades"]}

    assert MAJOR_FORBIDDEN_UPGRADES <= upgrades
