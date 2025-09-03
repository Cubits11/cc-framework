"""
Module: audit
Purpose: Tamper-evident JSONL audit chain (stable JSON, SHA-256 links)
Dependencies: hashlib, json, os, typing, datetime
Author: Pranav Bhave
Date: 2025-08-31
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, cast

__all__ = [
    "append_jsonl",
    "verify_chain",
    "tail_sha",
    "make_record",
    "rehash_file",
]

# ---------------- Stable JSON & IO ----------------


def _stable_dumps(obj: Mapping[str, Any]) -> str:
    """
    Deterministically serialize a mapping to JSON:
    - keys sorted
    - compact separators (no extraneous whitespace)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _ensure_parent_dir(path: str) -> None:
    """Ensure that the parent directory of ``path`` exists."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iter_jsonl(path: str):
    """Yield (1-based line number, parsed JSON) for each non-empty line."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def _compute_record_sha(rec_no_sha: Mapping[str, Any]) -> str:
    """
    Compute a SHA-256 hash for a record, ignoring its 'sha256' field.
    """
    payload = _stable_dumps(rec_no_sha).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------- Chain helpers ------------------


def tail_sha(path: str) -> Optional[str]:
    """
    Return the 'sha256' of the last record in a JSONL file or None if absent.
    """
    if not os.path.exists(path):
        return None
    try:
        last_nonempty: Optional[str] = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_nonempty = line
        if not last_nonempty:
            return None
        obj = json.loads(last_nonempty)
        val = obj.get("sha256")
        return val if isinstance(val, str) else None
    except Exception:
        return None


def append_jsonl(path: str, rec: Mapping[str, Any]) -> str:
    """
    Append a record to a JSONL audit log, adding a hash chain.

    Adds:
      - ``prev_sha256``: the previous record's sha256 (or None)
      - ``sha256``: the hash of the current record's content (excluding itself)

    Returns the hex digest of the appended record.
    """
    _ensure_parent_dir(path)

    prev = tail_sha(path)

    # Build the hashable record without its own sha256
    base: Dict[str, Any] = dict(rec)
    base["prev_sha256"] = prev

    sha = _compute_record_sha(base)

    final_rec = dict(base)
    final_rec["sha256"] = sha

    with open(path, "a", encoding="utf-8") as f:
        # write with stable layout to ensure verify_chain recomputes the same SHA
        f.write(_stable_dumps(final_rec) + "\n")

    return sha


def verify_chain(path: str) -> None:
    """
    Verify a JSONL hash chain.

    Checks:
      - Each record's ``sha256`` matches the hash of its content (excluding that field).
      - Each record's ``prev_sha256`` matches the previous record's ``sha256``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    prev: Optional[str] = None
    for i, obj in _iter_jsonl(path):
        exp_sha = obj.get("sha256")
        if exp_sha is None:
            raise RuntimeError(f"Line {i}: missing sha256")

        # Recompute hash on content minus sha256
        content = {k: v for k, v in obj.items() if k != "sha256"}
        got_sha = _compute_record_sha(content)
        if got_sha != exp_sha:
            raise RuntimeError(f"Line {i}: SHA mismatch (expected {exp_sha}, got {got_sha})")

        # Verify chain pointer
        if content.get("prev_sha256") != prev:
            raise RuntimeError(f"Line {i}: chain break (prev_sha256 mismatch)")

        prev = exp_sha


def rehash_file(path: str) -> None:
    """
    Recompute sha256 for each line (preserving prev_sha256 links) and rewrite file.
    Makes a .bak next to the file.
    """
    import shutil

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    tmp = f"{path}.tmp"
    bak = f"{path}.bak"
    shutil.copy2(path, bak)
    prev: Optional[str] = None
    with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["prev_sha256"] = prev
            payload = {k: v for k, v in obj.items() if k != "sha256"}
            sha = _compute_record_sha(payload)
            obj["sha256"] = sha
            fout.write(_stable_dumps(obj) + "\n")
            prev = sha
    os.replace(tmp, path)


# ---------------- Record builder ----------------


def _now_iso_utc() -> str:
    """ISO 8601 timestamp (UTC, seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ci(ci: Optional[Tuple[Optional[float], Optional[float]]]) -> Optional[Tuple[float, float]]:
    """
    Normalize CI:
      - if None -> None
      - if any bound is None -> None
      - else -> (float(lo), float(hi))
    """
    if ci is None:
        return None
    lo, hi = ci
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


def make_record(
    cfg: Mapping[str, Any],
    j_a: float,
    j_a_ci: Optional[Tuple[Optional[float], Optional[float]]],
    j_b: float,
    j_b_ci: Optional[Tuple[Optional[float], Optional[float]]],
    j_comp: float,
    j_comp_ci: Optional[Tuple[Optional[float], Optional[float]]],
    cc_max: float,
    delta_add: float,
    decision: str,
    figures: Iterable[str],
) -> Dict[str, Any]:
    """
    Build a JSON-serializable audit record. CI fields are `null` if not available.
    """
    comp = cast(str, cfg.get("comp", "AND"))
    name_a = cast(str, cfg.get("A", "A"))
    name_b = cast(str, cfg.get("B", "B"))

    rec: Dict[str, Any] = {
        "meta": {
            "schema": "cartographer/audit.v1",
            "ts": _now_iso_utc(),
        },
        "cfg": {
            "epsilon": cfg.get("epsilon"),
            "T": cfg.get("T"),
            "A": name_a,
            "B": name_b,
            "comp": comp,
            "samples": cfg.get("samples"),
            "seed": cfg.get("seed"),
        },
        "metrics": {
            "J_A": float(j_a),
            "J_A_CI": _ci(j_a_ci),
            "J_B": float(j_b),
            "J_B_CI": _ci(j_b_ci),
            "J_comp": float(j_comp),
            "J_comp_CI": _ci(j_comp_ci),
            "CC_max": float(cc_max),
            "Delta_add": float(delta_add),
        },
        "decision": str(decision),
        "figures": list(figures),
        # Chain fields filled by append_jsonl:
        # "prev_sha256": ...,
        # "sha256": ...,
    }
    return rec