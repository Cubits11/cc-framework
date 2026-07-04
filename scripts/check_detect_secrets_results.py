#!/usr/bin/env python3
"""Fail if a detect-secrets JSON scan contains findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: check_detect_secrets_results.py DETECT_SECRETS_JSON", file=sys.stderr)
        return 2

    findings = json.loads(Path(args[0]).read_text(encoding="utf-8")).get("results", {})
    if not isinstance(findings, dict):
        print("detect-secrets output missing object field: results", file=sys.stderr)
        return 2

    for path, entries in findings.items():
        count = len(entries) if isinstance(entries, list) else 1
        print(f"{path}: {count} candidate secret(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
