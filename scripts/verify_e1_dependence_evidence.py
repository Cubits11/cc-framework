#!/usr/bin/env python
"""Verify E1 Dependence-Evidence Study artifacts and deterministic regeneration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cc.evals.dependence_benchmark import verify_e1_artifacts

DEFAULT_ARTIFACT_DIR = Path("artifacts/empirical/e1")


def main() -> int:
    """Verify hashes, witnesses, and a clean in-memory reproduction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Check artifacts only; skip the stronger deterministic regeneration comparison.",
    )
    args = parser.parse_args()
    errors = verify_e1_artifacts(args.artifact_dir, regenerate=not args.no_regenerate)
    if errors:
        for error in errors:
            print(f"E1 verification error: {error}", file=sys.stderr)
        return 1
    print(f"Verified E1 artifacts in {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
