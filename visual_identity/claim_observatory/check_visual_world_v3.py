#!/usr/bin/env python3
"""QA checks for the Claim Observatory V3 visual world."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    ROOT / "V3_FROM_V1_CRITIQUE.md",
    ROOT / "WORLD_BIBLE_V3.md",
    ROOT / "README_V3_WORLD.md",
    ROOT / "create_claim_observatory_world_v3.py",
    ROOT / "claim_observatory_world_v3.blend",
    ROOT / "visual_world_manifest_v3.json",
]

REQUIRED_RENDERS = [
    ROOT / "renders/world_v3_hero.png",
    ROOT / "renders/world_v3_claim_capsule.png",
    ROOT / "renders/world_v3_nonclaims_wall.png",
    ROOT / "renders/world_v3_decay_clock.png",
    ROOT / "renders/world_v3_support_graph.png",
    ROOT / "renders/world_v3_replay_engine.png",
]

REQUIRED_SCRIPT_STRINGS = [
    "Not safety scores",
    "Evidence-bound claims",
    "PASS under verifier rules",
    "Not a deployment-safety proof",
    "THIS CLAIM DOES NOT SAY",
    "Meaning survives replay",
    "Receipt integrity",
    "claims are mortal",
]

FORBIDDEN_SCRIPT_PHRASES = [
    "certified safe",
    "approved for deployment",
    "guaranteed safety",
    "trusted AI",
    "verified safe",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def check_exists(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        fail(f"missing {label}: {', '.join(str(path.relative_to(ROOT)) for path in missing)}")


def main() -> int:
    check_exists(REQUIRED_FILES, "V3 files")
    check_exists(REQUIRED_RENDERS, "V3 renders")

    script_path = ROOT / "create_claim_observatory_world_v3.py"
    script_text = script_path.read_text(encoding="utf-8")
    lower_script = script_text.lower()

    for needle in REQUIRED_SCRIPT_STRINGS:
        if needle not in script_text:
            fail(f"scene-generation script is missing required phrase: {needle}")

    allowed_caveat = "does not mean the ai system is safe in deployment"
    for phrase in FORBIDDEN_SCRIPT_PHRASES:
        if phrase in lower_script and allowed_caveat not in lower_script:
            fail(f"scene-generation script contains forbidden phrase: {phrase}")

    manifest_path = ROOT / "visual_world_manifest_v3.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"visual manifest is not valid JSON: {exc}")

    if manifest.get("schema") != "cc.visual_claim_observatory_world.v3":
        fail("visual manifest schema mismatch")

    camera_names = {camera.get("name") for camera in manifest.get("cameras", [])}
    required_cameras = {
        "Camera_WorldHero",
        "Camera_Arrival",
        "Camera_ClaimCapsule",
        "Camera_EvidenceVault",
        "Camera_SupportGraphOrrery",
        "Camera_NonClaimsWall",
        "Camera_DecayClockRoom",
        "Camera_ChallengeRange",
        "Camera_ReplayManifestEngine",
        "Camera_HumanReviewTribunal",
        "Camera_LedgerTower",
        "Camera_FrechetAtomGarden",
    }
    missing_cameras = sorted(required_cameras - camera_names)
    if missing_cameras:
        fail(f"visual manifest is missing cameras: {', '.join(missing_cameras)}")

    print("OK: V3 visual world files, renders, strings, cameras, and manifest passed QA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

