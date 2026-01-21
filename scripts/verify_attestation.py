#!/usr/bin/env python3
"""Verify an audit run attestation against its manifest + results."""

import argparse
from pathlib import Path

from cc.core.audit_runner import verify_attestation


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify audit run attestation.")
    parser.add_argument("run_dir", help="Run directory containing attestation.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    attestation_path = run_dir / "attestation.json"
    manifest_path = run_dir / "execution_manifest.json"
    results_path = run_dir / "results.jsonl"

    ok, reason = verify_attestation(attestation_path, manifest_path, results_path)
    if ok:
        print(f"attestation verified: {reason}")
    else:
        raise SystemExit(f"attestation verification failed: {reason}")


if __name__ == "__main__":
    main()
