from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "claims" / "claim_boundary_manifest.v0.1.json"

TOP_LEVEL_REQUIRED = {
    "schema_version",
    "generated_for",
    "purpose",
    "claim_levels",
    "claims",
    "forbidden_upgrades",
    "global_non_claims",
}

CLAIM_REQUIRED = {
    "id",
    "level",
    "lane",
    "status",
    "claim_text",
    "supporting_files",
    "supporting_tests_or_commands",
    "non_claims",
    "risk_if_overstated",
}


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any], root: Path = ROOT) -> list[str]:
    errors: list[str] = []

    missing_top = sorted(TOP_LEVEL_REQUIRED - manifest.keys())
    if missing_top:
        errors.append(f"missing top-level keys: {', '.join(missing_top)}")

    claim_levels = manifest.get("claim_levels")
    if not isinstance(claim_levels, dict) or not claim_levels:
        errors.append("claim_levels must be a non-empty object")
        claim_levels = {}

    claims = manifest.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("claims must be a non-empty list")
        claims = []

    seen_ids: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"claim at index {index} must be an object")
            continue

        claim_id = str(claim.get("id", f"<index {index}>"))
        missing_claim = sorted(CLAIM_REQUIRED - claim.keys())
        if missing_claim:
            errors.append(f"{claim_id}: missing claim keys: {', '.join(missing_claim)}")

        if claim_id in seen_ids:
            errors.append(f"duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)

        level = claim.get("level")
        if not isinstance(level, str) or not level:
            errors.append(f"{claim_id}: level must be a non-empty string")
        else:
            for part in level.split("/"):
                if part not in claim_levels:
                    errors.append(f"{claim_id}: unknown claim level {part!r}")

        supporting_files = claim.get("supporting_files")
        if not isinstance(supporting_files, list) or not supporting_files:
            errors.append(f"{claim_id}: supporting_files must be a non-empty list")
        else:
            for file_path in supporting_files:
                if not isinstance(file_path, str) or not file_path:
                    errors.append(f"{claim_id}: supporting file entries must be non-empty strings")
                    continue
                if not (root / file_path).exists():
                    errors.append(f"{claim_id}: missing supporting file on disk: {file_path}")

        non_claims = claim.get("non_claims")
        if not isinstance(non_claims, list) or not non_claims:
            errors.append(f"{claim_id}: non_claims must contain at least one entry")

        tests = claim.get("supporting_tests_or_commands")
        if not isinstance(tests, list) or not tests:
            errors.append(f"{claim_id}: supporting_tests_or_commands must be a non-empty list")

    forbidden_upgrades = manifest.get("forbidden_upgrades")
    if not isinstance(forbidden_upgrades, list) or not forbidden_upgrades:
        errors.append("forbidden_upgrades must be a non-empty list")
    else:
        for index, upgrade in enumerate(forbidden_upgrades):
            if not isinstance(upgrade, dict):
                errors.append(f"forbidden upgrade at index {index} must be an object")
                continue
            for key in ("from", "to", "reason"):
                if not upgrade.get(key):
                    errors.append(f"forbidden upgrade at index {index} missing {key!r}")

    global_non_claims = manifest.get("global_non_claims")
    if not isinstance(global_non_claims, list) or not global_non_claims:
        errors.append("global_non_claims must be a non-empty list")

    return errors


def main() -> int:
    manifest = load_manifest()
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        "Claim boundary manifest validation passed: "
        f"{len(manifest['claims'])} claims, "
        f"{len(manifest['forbidden_upgrades'])} forbidden upgrades."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
