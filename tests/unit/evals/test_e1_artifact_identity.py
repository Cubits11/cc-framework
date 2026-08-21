"""E1.1 -- executable definition of E1 artifact identity.

``manifest_payload_sha256`` must answer exactly one question:

    "Is this the same scientific payload?"

It must NOT partly answer "was this generated at the same filesystem
location?".  These tests pin the whole invariant family rather than the single
regression that motivated the fix:

    H(P, d1) == H(P, d2)     for arbitrary output roots d1, d2
    H(P1, d) != H(P2, d)     whenever an epistemically material field changes

A naive fix could satisfy the first condition by hashing too little, so the
mutation-sensitivity half is not optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cc.evals.dependence_benchmark import (
    E1_MANIFEST_SCHEMA_VERSION,
    verify_e1_artifacts,
    write_e1_artifacts,
)

# Small but structurally complete design: keeps the identity surface intact
# while staying fast enough for a unit test.
FAST_DESIGN: dict[str, Any] = {
    "seed": 7,
    "delta": 0.10,
    "replicates": 2,
    "sample_sizes": (16,),
}


def _emit(output_dir: Path, /, **overrides: Any) -> dict[str, Any]:
    """Write artifacts into ``output_dir`` and return the parsed manifest."""

    design = {**FAST_DESIGN, **overrides}
    write_e1_artifacts(
        output_dir,
        generation_command=f"scripts/reproduce_e1_dependence_evidence.py --output-dir {output_dir}",
        **design,
    )
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _digest(output_dir: Path, /, **overrides: Any) -> str:
    return str(_emit(output_dir, **overrides)["manifest_payload_sha256"])


# --------------------------------------------------------------------------
# MUST NOT change the digest: pure relocation.
# --------------------------------------------------------------------------


def test_digest_is_invariant_under_relocation(tmp_path: Path) -> None:
    """Identical payloads emitted to unrelated roots share one identity."""

    roots = [
        tmp_path / "replay-a",
        tmp_path / "deeply" / "nested" / "replay-b",
        tmp_path / "a-much-longer-directory-name-c",
    ]
    digests = {_digest(root) for root in roots}
    assert len(digests) == 1, f"relocation changed scientific identity: {digests}"


def test_relocation_preserves_provenance_without_hashing_it(tmp_path: Path) -> None:
    """Provenance still records *where*; it just does not define identity."""

    left = _emit(tmp_path / "left")
    right = _emit(tmp_path / "right")

    assert left["manifest_payload_sha256"] == right["manifest_payload_sha256"]
    # Provenance is genuinely preserved, and genuinely different.
    assert left["execution_provenance"]["output_dir"] != right["execution_provenance"]["output_dir"]
    assert (
        left["execution_provenance"]["generation_command"]
        != right["execution_provenance"]["generation_command"]
    )
    # And it is explicitly flagged as outside the digest.
    assert left["execution_provenance"]["hashed_by_manifest_payload_sha256"] is False
    assert "execution_provenance" not in left["identity_payload"]


def test_relocated_replay_verifies(tmp_path: Path) -> None:
    """An external challenger replaying into their own directory passes."""

    target = tmp_path / "someone-elses-checkout" / "artifacts"
    _emit(target)
    assert verify_e1_artifacts(target, regenerate=False) == []


# --------------------------------------------------------------------------
# MUST change the digest: epistemically material mutations.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 8),
        ("delta", 0.05),
        ("replicates", 3),
        ("sample_sizes", (32,)),
    ],
)
def test_digest_changes_under_material_mutation(tmp_path: Path, field: str, value: Any) -> None:
    """Changing any frozen design field must change scientific identity."""

    baseline = _digest(tmp_path / "baseline")
    mutated = _digest(tmp_path / f"mutated-{field}", **{field: value})
    assert baseline != mutated, f"mutating {field!r} left the digest unchanged"


def test_digest_covers_the_result_payload(tmp_path: Path) -> None:
    """Editing a data file must break verification, not merely the byte size."""

    target = tmp_path / "tampered"
    _emit(target)
    study_path = target / "study.json"
    payload = json.loads(study_path.read_text(encoding="utf-8"))
    payload["coverage_rows"][0]["covered"] = int(payload["coverage_rows"][0]["covered"]) - 1
    study_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    errors = verify_e1_artifacts(target, regenerate=False)
    assert errors, "a mutated result payload verified clean"
    assert any("hash mismatch" in error for error in errors), errors


def test_identity_payload_is_self_describing(tmp_path: Path) -> None:
    """Identity carries the design it claims, cross-checked against the study."""

    manifest = _emit(tmp_path / "identity")
    identity = manifest["identity_payload"]

    assert manifest["schema_version"] == E1_MANIFEST_SCHEMA_VERSION
    assert identity["study_id"] == "E1-dependence-evidence"
    assert identity["design"]["seed"] == FAST_DESIGN["seed"]
    assert identity["design"]["replicates"] == FAST_DESIGN["replicates"]
    assert [record["filename"] for record in identity["files"]] == ["study.json", "coverage.csv"]


def test_manifest_design_must_match_the_study(tmp_path: Path) -> None:
    """A manifest whose declared design contradicts study.json is rejected."""

    target = tmp_path / "contradiction"
    manifest = _emit(target)
    manifest["identity_payload"]["design"]["seed"] = 999_999
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    errors = verify_e1_artifacts(target, regenerate=False)
    assert any("design does not match" in error for error in errors), errors
