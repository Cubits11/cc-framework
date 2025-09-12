# src/cc/cartographer/suggest.py
"""
Suggest next (epsilon, T, comp) trials from audit history.

Usage:
  python -m cc.cartographer.suggest --history runs/audit.jsonl --out experiments/grids/next.json

Policy:
- Read the last valid record from the tamper-evident audit chain.
- Classify the regime via CC_max:
    < 0.95  → "constructive"  (A∧B beats best single rail; exploit locally)
    ≤ 1.05  → "independent"   (A∧B ≈ best; probe lightly)
    > 1.05  → "destructive"   (A∧B underperforms; retreat + flip rule)
- Propose a small, deduplicated grid around (epsilon, T, comp) with bounded steps.

Notes:
- This is a pragmatic search suggester for experiments, not a theorem prover.
- All outputs are JSON-serializable dictionaries with keys: epsilon, T, comp, rationale.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .audit import _iter_jsonl  # reuse the stable JSONL reader


# =============================================================================
# Config / bounds
# =============================================================================


@dataclass(frozen=True)
class Bounds:
    eps_min: float = 0.0
    eps_max: float = 1.0
    T_min: float = 0.0
    T_max: float = 10.0  # soft cap; can be raised if your runs use larger horizons


BOUNDS = Bounds()


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _round4(x: float) -> float:
    return float(round(x, 4))


def _region(cc_max: float) -> str:
    """Classify composition regime by CC_max (conservative normalization)."""
    if cc_max < 0.95:
        return "constructive"
    if cc_max <= 1.05:
        return "independent"
    return "destructive"


def _neighbors(x: float, steps: Sequence[float], lo: float, hi: float) -> List[float]:
    """Generate deduped neighbor candidates x + step clamped to [lo, hi]."""
    seen = set()
    out: List[float] = []
    for s in steps:
        v = _clamp(float(x) + float(s), lo, hi)
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _last_valid(path: str) -> Optional[Dict[str, Any]]:
    """
    Return the last record that has both 'cfg' and 'metrics' dicts.
    If none found, return None.
    """
    last: Optional[Dict[str, Any]] = None
    for _, obj in _iter_jsonl(path):
        cfg = obj.get("cfg")
        mets = obj.get("metrics")
        if isinstance(cfg, dict) and isinstance(mets, dict):
            last = obj
    return last


def _cold_start(k: int) -> List[Dict[str, Any]]:
    """Canonical seeds when there is no history."""
    eps_grid = (0.00, 0.10, 0.20, 0.40)
    T_grid = (1.0, 2.0, 5.0)
    comps = ("AND", "OR")
    out: List[Dict[str, Any]] = []
    for e in eps_grid:
        for t in T_grid:
            for c in comps:
                out.append(
                    {
                        "epsilon": _round4(_clamp(e, BOUNDS.eps_min, BOUNDS.eps_max)),
                        "T": _round4(_clamp(t, BOUNDS.T_min, BOUNDS.T_max)),
                        "comp": c,
                        "rationale": "cold start seed",
                    }
                )
    return out[:k]


# =============================================================================
# Suggestion logic
# =============================================================================


def suggest(history: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    Propose up to k next trials based on the last audited configuration and outcome.

    Heuristics by zone:
      - constructive: local exploitation around (epsilon, T), try both comp rules.
      - independent: light epsilon probe, single comp; add one comp toggle probe.
      - destructive: retreat with larger steps and flip comp.
    """
    rec = _last_valid(history)
    if not rec:
        return _cold_start(k)

    cfg = rec.get("cfg", {})
    mets = rec.get("metrics", {})

    # Pull last parameters with robust defaults
    e0_raw = cfg.get("epsilon")
    t0_raw = cfg.get("T")
    comp0 = cfg.get("comp", "AND")
    cc_raw = mets.get("CC_max", 1.0)

    try:
        e0 = float(e0_raw)
    except Exception:
        e0 = 0.10
    try:
        t0 = float(t0_raw)
    except Exception:
        t0 = 2.0
    try:
        cc = float(cc_raw)
    except Exception:
        cc = 1.0

    # Clamp into supported bounds
    e0 = _clamp(e0, BOUNDS.eps_min, BOUNDS.eps_max)
    t0 = _clamp(t0, BOUNDS.T_min, BOUNDS.T_max)
    comp0 = "AND" if str(comp0).upper() == "AND" else "OR"

    zone = _region(cc)

    proposals: List[Dict[str, Any]] = []

    if zone == "constructive":
        # Exploit the valley; small isotropic moves; test both comp rules
        for e in _neighbors(e0, (-0.05, 0.0, +0.05), BOUNDS.eps_min, BOUNDS.eps_max):
            for t in _neighbors(t0, (-0.5, 0.0, +0.5), BOUNDS.T_min, BOUNDS.T_max):
                for c in (comp0, ("OR" if comp0 == "AND" else "AND")):
                    proposals.append(
                        {
                            "epsilon": _round4(e),
                            "T": _round4(t),
                            "comp": c,
                            "rationale": "exploit constructive valley",
                        }
                    )

    elif zone == "independent":
        # Gentle probe primarily over epsilon; keep T fixed; one comp toggle
        for e in _neighbors(e0, (-0.10, 0.0, +0.10, +0.20), BOUNDS.eps_min, BOUNDS.eps_max):
            proposals.append(
                {
                    "epsilon": _round4(e),
                    "T": _round4(t0),
                    "comp": comp0,
                    "rationale": "independent plateau: epsilon sweep",
                }
            )
        proposals.append(
            {
                "epsilon": _round4(e0),
                "T": _round4(t0),
                "comp": "OR" if comp0 == "AND" else "AND",
                "rationale": "independent plateau: comp toggle probe",
            }
        )

    else:  # destructive
        # Move away aggressively and flip composition rule
        flip = "OR" if comp0 == "AND" else "AND"
        for e in _neighbors(e0, (-0.20, -0.10, +0.10, +0.20), BOUNDS.eps_min, BOUNDS.eps_max):
            for t in _neighbors(t0, (-1.0, +1.0, +2.0), BOUNDS.T_min, BOUNDS.T_max):
                proposals.append(
                    {
                        "epsilon": _round4(e),
                        "T": _round4(t),
                        "comp": flip,
                        "rationale": "destructive wedge: retreat + flip comp",
                    }
                )

    # Deduplicate and keep ordering stable; cap to k
    seen: set[Tuple[float, float, str]] = set()
    uniq: List[Dict[str, Any]] = []
    for s in proposals:
        key = (float(s["epsilon"]), float(s["T"]), str(s["comp"]))
        if key not in seen:
            uniq.append(s)
            seen.add(key)

    return uniq[: max(0, int(k))]


# =============================================================================
# CLI
# =============================================================================


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Suggest next (epsilon, T, comp) trials from audit history.")
    p.add_argument("--history", required=True, help="Path to runs/audit.jsonl")
    p.add_argument("--out", required=True, help="Path to write JSON suggestions (e.g., experiments/grids/next.json)")
    p.add_argument("--k", type=int, default=6, help="Number of suggestions to emit")
    args = p.parse_args(argv)

    recs = suggest(args.history, k=args.k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2)
    print(f"wrote {len(recs)} suggestions to {args.out}")


if __name__ == "__main__":
    main()