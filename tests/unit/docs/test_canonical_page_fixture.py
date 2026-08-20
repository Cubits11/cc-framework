"""The canonical page embeds a real fixture, so it must not drift from the capsule.

`visual_identity/canonical_page/index.html` recomputes SHA-256 in the visitor's
browser and compares it against a digest recorded in the deterministic claim
governance capsule. If the capsule is regenerated and the page is not updated,
the page would show a legitimate artifact failing closed - an honest-looking
screen making a false statement. These tests fail first instead.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "visual_identity" / "canonical_page" / "index.html"
CAPSULE = ROOT / "examples" / "claim_governance_capsule"
FIXTURE = CAPSULE / "expected" / "calibration.json"
MANIFEST = CAPSULE / "manifest.expected.json"


def _embedded(name: str) -> str:
    match = re.search(rf"^const {name} = (.+);$", PAGE.read_text(), re.MULTILINE)
    assert match, f"{PAGE.name} no longer defines {name}"
    return json.loads(match.group(1))


def _recorded_digest() -> str:
    manifest = json.loads(MANIFEST.read_text())
    entry = next(f for f in manifest["files"] if f["filename"] == FIXTURE.name)
    return entry["sha256"]


def test_embedded_document_matches_the_capsule_fixture() -> None:
    assert _embedded("DOC") == FIXTURE.read_text()


def test_embedded_digest_matches_the_capsule_manifest() -> None:
    assert _embedded("EXPECTED") == _recorded_digest()


def test_page_check_passes_on_unmodified_bytes() -> None:
    digest = hashlib.sha256(_embedded("DOC").encode()).hexdigest()

    assert digest == _embedded("EXPECTED")


def test_capsule_manifest_digest_is_a_plain_file_digest() -> None:
    """The browser hashes raw bytes; the manifest must mean the same thing."""
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == _recorded_digest()
