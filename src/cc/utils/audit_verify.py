# src/cc/utils/audit_verify.py
"""CLI helper to verify tamper-evident guardrail audit logs."""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

from cc.cartographer.audit import verify_chain


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Verify a guardrail audit JSONL chain.")
    parser.add_argument("path", help="Path to the audit JSONL log.")
    args = parser.parse_args(argv)

    verify_chain(args.path)
    print(f"audit chain OK: {args.path}")


if __name__ == "__main__":
    main()
