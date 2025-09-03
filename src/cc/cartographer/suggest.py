"""
Suggest next (epsilon, T, comp) trials from audit history.

Usage:
  python -m cc.cartographer.suggest --history runs/audit.jsonl --out experiments/grids/next.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

from .audit import _iter_jsonl  # reuse internal reader


def _region(cc_max: float) -> str:
    if cc_max < 0.95:
        return "constructive"
    if cc_max <= 1.05:
        return "independent"
    return "destructive"


def _neighbors(x: float, steps: Tuple[float, ...]) -> List[float]:
    out = [max(0.0, x + s) for s in steps]
    # dedupe while preserving order
    seen = set()
    keep: List[float] = []
    for v in out:
        if v not in seen:
            keep.append(v); seen.add(v)
    return keep


def _last(path: str) -> Optional[Dict[str, Any]]:
    last = None
    for _, obj in _iter_jsonl(path):
        last = obj
    return last


def suggest(history: str, k: int = 6) -> List[Dict[str, Any]]:
    rec = _last(history)
    out: List[Dict[str, Any]] = []
    if not rec:
        # cold start: canonical seeds
        seeds = [
            {"epsilon": e, "T": t, "comp": c, "rationale": "cold start seed"}
            for e in (0.0, 0.1, 0.2)
            for t in (1.0, 2.0, 5.0)
            for c in ("AND", "OR")
        ]
        return seeds[:k]

    cfg = rec.get("cfg", {})
    mets = rec.get("metrics", {})
    e0 = cfg.get("epsilon"); t0 = cfg.get("T"); comp0 = cfg.get("comp", "AND")
    cc = float(mets.get("CC_max", 1.0))
    zone = _region(cc)

    # reasonable defaults if missing
    e0 = float(e0) if isinstance(e0, (int, float)) else 0.1
    t0 = float(t0) if isinstance(t0, (int, float)) else 2.0

    if zone == "constructive":
        # exploit nearby; small moves; try both rules
        for e in _neighbors(e0, (-0.05, 0.0, +0.05)):
            for t in _neighbors(t0, (-0.5, 0.0, +0.5)):
                for c in (comp0, "OR" if comp0 == "AND" else "AND"):
                    out.append({"epsilon": round(e, 4), "T": round(t, 4), "comp": c,
                                "rationale": "exploit constructive valley"})
    elif zone == "independent":
        # shrink search—probe epsilon only; toggle rule once
        for e in _neighbors(e0, (-0.1, 0.0, +0.1, +0.2)):
            out.append({"epsilon": round(e, 4), "T": round(t0, 4), "comp": comp0,
                        "rationale": "independent plateau: favor single, light epsilon sweep"})
        out.append({"epsilon": round(e0, 4), "T": round(t0, 4),
                    "comp": "OR" if comp0 == "AND" else "AND",
                    "rationale": "independent plateau: rule toggle probe"})
    else:
        # destructive: move away; bigger steps + flip rule
        for e in _neighbors(e0, (-0.2, -0.1, +0.1, +0.2)):
            for t in _neighbors(t0, (-1.0, +1.0, +2.0)):
                out.append({"epsilon": round(e, 4), "T": round(max(0.0, t), 4),
                            "comp": "OR" if comp0 == "AND" else "AND",
                            "rationale": "destructive wedge: retreat and flip rule"})

    # light dedupe and trim
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for s in out:
        key = (s["epsilon"], s["T"], s["comp"])
        if key not in seen:
            uniq.append(s); seen.add(key)
    return uniq[:k]


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=6)
    args = p.parse_args(argv)

    recs = suggest(args.history, k=args.k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2)
    print(f"wrote {len(recs)} suggestions to {args.out}")


if __name__ == "__main__":
    main()