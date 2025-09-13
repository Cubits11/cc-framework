# src/cc/cartographer/audit.py
"""
Module: audit
Purpose:
  (A) Tamper-evident JSONL audit chain (stable JSON, SHA-256 links)
  (B) FH-ceiling auditor for composed J against Fréchet–Hoeffding envelopes

Dependencies: hashlib, json, os, typing, datetime, numpy
Author: Pranav Bhave
Date: 2025-08-31 (updated 2025-09-12)

Design notes (A)
----------------
- Each line is a complete JSON object with:
    • sha256: SHA-256 of the line’s content excluding the sha256 field itself
    • prev_sha256: pointer to the previous record’s sha256 (or null for the first)
- Serialization uses a *stable* JSON form: sorted keys + compact separators.
- Appends are flushed and fsync’d for durability (opt-out by setting fsync=False).
- `verify_chain` replays the file and re-computes hashes to detect tampering.

Design notes (B)
----------------
- The FH auditor uses `envelope_over_rocs` to compute the J ceiling grid
  and checks observed composed J values at specified cross-threshold pairs.
- Two entry points:
    • `audit_fh_ceiling_by_index`: indices into ROC arrays + observed J.
    • `audit_fh_ceiling_by_points`: explicit ROC points + observed J (maps to indices).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, cast, Literal

import numpy as np

from .bounds import envelope_over_rocs, ensure_anchors

__all__ = [
    # JSONL chain
    "append_jsonl",
    "verify_chain",
    "tail_sha",
    "make_record",
    "rehash_file",
    "append_record",   # convenience: make_record + append_jsonl
    "AuditError",
    # FH auditor
    "audit_fh_ceiling_by_index",
    "audit_fh_ceiling_by_points",
]


# =============================================================================
# Errors
# =============================================================================


@dataclass
class AuditError(Exception):
    """Base class for audit chain errors."""
    message: str

    def __str__(self) -> str:
        return self.message


# =============================================================================
# Stable JSON & IO
# =============================================================================


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


def _iter_jsonl(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (1-based line number, parsed JSON) for each non-empty line."""
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise AuditError(f"Line {i}: invalid JSON ({e})")
            if not isinstance(obj, dict):
                raise AuditError(f"Line {i}: JSON object expected")
            yield i, cast(Dict[str, Any], obj)


def _compute_record_sha(rec_without_sha: Mapping[str, Any]) -> str:
    """
    Compute a SHA-256 hash for a record, ignoring its 'sha256' field.
    """
    payload = _stable_dumps(rec_without_sha).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# =============================================================================
# Chain helpers
# =============================================================================


def tail_sha(path: str) -> Optional[str]:
    """
    Return the 'sha256' of the last record in a JSONL file or None if absent.
    """
    if not os.path.exists(path):
        return None
    last_nonempty: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_nonempty = line
    if not last_nonempty:
        return None
    try:
        obj = json.loads(last_nonempty)
    except json.JSONDecodeError:
        return None
    val = obj.get("sha256")
    return val if isinstance(val, str) else None


def append_jsonl(path: str, rec: Mapping[str, Any], *, fsync: bool = True) -> str:
    """
    Append a record to a JSONL audit log, adding a hash chain.

    Adds:
      - ``prev_sha256``: the previous record's sha256 (or None)
      - ``sha256``: the hash of the current record's content (excluding itself)

    Returns:
      Hex digest of the appended record.
    """
    _ensure_parent_dir(path)
    prev = tail_sha(path)

    # Build the hashable record without its own sha256
    base: Dict[str, Any] = dict(rec)
    base["prev_sha256"] = prev

    sha = _compute_record_sha(base)

    final_rec = dict(base)
    final_rec["sha256"] = sha
    line = _stable_dumps(final_rec) + "\n"

    # Append + flush + optional fsync for durability
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        if fsync:
            os.fsync(f.fileno())

    return sha


def verify_chain(path: str) -> None:
    """
    Verify a JSONL hash chain.

    Checks:
      - Each record's ``sha256`` matches the hash of its content (excluding that field).
      - Each record's ``prev_sha256`` matches the previous record's ``sha256``.

    Raises:
      AuditError on any tampering or structural issue.
      FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    prev: Optional[str] = None
    for i, obj in _iter_jsonl(path):
        exp_sha = obj.get("sha256")
        if not isinstance(exp_sha, str):
            raise AuditError(f"Line {i}: missing or invalid 'sha256'")

        # Recompute hash on content minus sha256
        content = {k: v for k, v in obj.items() if k != "sha256"}
        got_sha = _compute_record_sha(content)
        if got_sha != exp_sha:
            raise AuditError(
                f"Line {i}: SHA mismatch (expected {exp_sha}, recomputed {got_sha})"
            )

        # Verify chain pointer
        if content.get("prev_sha256") != prev:
            raise AuditError(
                f"Line {i}: chain break (prev_sha256 mismatch; saw {content.get('prev_sha256')}, expected {prev})"
            )

        prev = exp_sha


def rehash_file(path: str) -> None:
    """
    Recompute sha256 for each line (preserving prev_sha256 links) and rewrite file.
    Makes a .bak next to the file.

    This is useful if records were written with a different serializer or fields
    were re-ordered, and you want to canonicalize the stable layout.
    """
    import shutil

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    tmp = f"{path}.tmp"
    bak = f"{path}.bak"
    shutil.copy2(path, bak)

    prev: Optional[str] = None
    with open(path, "r", encoding="utf-8") as fin, open(
        tmp, "w", encoding="utf-8"
    ) as fout:
        for i, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise AuditError(f"Line {i}: invalid JSON during rehash ({e})")
            if not isinstance(obj, dict):
                raise AuditError(f"Line {i}: JSON object expected during rehash")

            # Reconnect chain pointer and recompute hash
            obj["prev_sha256"] = prev
            payload = {k: v for k, v in obj.items() if k != "sha256"}
            sha = _compute_record_sha(payload)
            obj["sha256"] = sha

            fout.write(_stable_dumps(obj) + "\n")
            prev = sha

    os.replace(tmp, path)


# =============================================================================
# Record builder
# =============================================================================


def _now_iso_utc() -> str:
    """ISO 8601 timestamp (UTC, seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ci(
    ci: Optional[Tuple[Optional[float], Optional[float]]]
) -> Optional[Tuple[float, float]]:
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


def _to_str_list(items: Iterable[Any]) -> list[str]:
    """Normalize an iterable into a list[str] for JSON output."""
    out: list[str] = []
    for x in items:
        if x is None:
            continue
        out.append(str(x))
    return out


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
        "figures": _to_str_list(figures),
        # Chain fields filled by append_jsonl:
        # "prev_sha256": ...,
        # "sha256": ...,
    }
    return rec


# =============================================================================
# Convenience
# =============================================================================


def append_record(
    path: str,
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
    *,
    fsync: bool = True,
) -> str:
    """
    Convenience: build a record with `make_record` and append it to the chain.
    Returns the appended record's sha256.
    """
    rec = make_record(
        cfg=cfg,
        j_a=j_a,
        j_a_ci=j_a_ci,
        j_b=j_b,
        j_b_ci=j_b_ci,
        j_comp=j_comp,
        j_comp_ci=j_comp_ci,
        cc_max=cc_max,
        delta_add=delta_add,
        decision=decision,
        figures=figures,
    )
    return append_jsonl(path, rec, fsync=fsync)


# =============================================================================
# Fréchet–Hoeffding (FH) ceiling auditor
# =============================================================================

Violation = Tuple[int, int, float, float]  # (i_a, i_b, J_obs, J_cap)


def audit_fh_ceiling_by_index(
    roc_a: np.ndarray,
    roc_b: np.ndarray,
    pairs: Iterable[Tuple[int, int, float]],
    *,
    comp: Literal["AND", "OR", "and", "or"] = "AND",
    add_anchors: bool = False,
    tol: float = 1e-12,
) -> List[Violation]:
    """
    Given indices into roc_a/roc_b and observed J at those index pairs, flag any FH-cap violations.
    Returns list of (i_a, i_b, J_obs, J_cap) for offending points (empty if none).

    Args:
        roc_a, roc_b: arrays of shape (Na,2) and (Nb,2) with columns [FPR, TPR].
        pairs: iterable of (i_a, i_b, J_obs).
        comp: "AND" or "OR" composition.
        add_anchors: if True, ensure (0,0) and (1,1) present (indices will refer to the *augmented* arrays).
        tol: allowed numerical slack (J_obs > J_cap + tol counts as violation).
    """
    if add_anchors:
        A = ensure_anchors(roc_a)
        B = ensure_anchors(roc_b)
    else:
        A = np.asarray(roc_a, dtype=float)
        B = np.asarray(roc_b, dtype=float)

    # Build FH envelope grid once
    _, Jgrid = envelope_over_rocs(A, B, comp=comp, add_anchors=False)
    H, W = Jgrid.shape

    bad: List[Violation] = []
    for ia, ib, j_obs in pairs:
        if not (0 <= ia < H and 0 <= ib < W):
            # ignore out-of-range indices quietly
            continue
        j_cap = float(Jgrid[ia, ib])
        if j_obs > j_cap + tol:
            bad.append((int(ia), int(ib), float(j_obs), j_cap))
    return bad


def audit_fh_ceiling_by_points(
    roc_a: np.ndarray,
    roc_b: np.ndarray,
    triples: Iterable[Tuple[Tuple[float, float], Tuple[float, float], float]],
    *,
    comp: Literal["AND", "OR", "and", "or"] = "AND",
    add_anchors: bool = False,
    tol: float = 1e-12,
    strict: bool = False,
) -> List[Tuple[Tuple[float, float], Tuple[float, float], float, float]]:
    """
    Like `audit_fh_ceiling_by_index`, but accepts explicit ROC points for each rail.

    Args:
        triples: iterable of ((fpr_a, tpr_a), (fpr_b, tpr_b), J_obs).
        strict: if True, raise if a point is not found exactly in roc_a/roc_b;
                if False, skip points that do not match exactly.

    Returns:
        List of ((fpr_a, tpr_a), (fpr_b, tpr_b), J_obs, J_cap) for violations.
    """
    if add_anchors:
        A = ensure_anchors(roc_a)
        B = ensure_anchors(roc_b)
    else:
        A = np.asarray(roc_a, dtype=float)
        B = np.asarray(roc_b, dtype=float)

    # Map points to indices by exact match
    def idx_of(M: np.ndarray, pt: Tuple[float, float]) -> Optional[int]:
        mask = (np.isclose(M[:, 0], pt[0])) & (np.isclose(M[:, 1], pt[1]))
        where = np.nonzero(mask)[0]
        return int(where[0]) if where.size > 0 else None

    # Envelope grid
    _, Jgrid = envelope_over_rocs(A, B, comp=comp, add_anchors=False)

    violations: List[Tuple[Tuple[float, float], Tuple[float, float], float, float]] = []
    for pa, pb, j_obs in triples:
        ia = idx_of(A, pa)
        ib = idx_of(B, pb)
        if ia is None or ib is None:
            if strict:
                missing = []
                if ia is None:
                    missing.append(f"A{pa}")
                if ib is None:
                    missing.append(f"B{pb}")
                raise AuditError(f"Point(s) not found in ROC arrays: {', '.join(missing)}")
            else:
                # Skip silently when not strict
                continue
        j_cap = float(Jgrid[ia, ib])
        if j_obs > j_cap + tol:
            violations.append((pa, pb, float(j_obs), j_cap))
    return violations
