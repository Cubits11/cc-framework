#!/usr/bin/env python3
"""
scripts/rails_compare.py

Compute J (Youden's J) for two rails under an FPR window, plus:
- composition (any / both)
- independence baselines
- lightweight dependence diagnostics (mutual information, overlap)

Integration test expects:
- CLI: --csv <path> --out <path>
- Output CSV contains at least: A_J, B_J, Combo_any_J, Indep_any_J
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Core metric helpers
# -----------------------------


def binarize(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores >= thr).astype(int)


def confusion(y: np.ndarray, yhat: np.ndarray) -> Dict[str, int]:
    tp = int(((y == 1) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def rates(m: Dict[str, int]) -> Tuple[float, float, float]:
    tp, fn, tn, fp = m["tp"], m["fn"], m["tn"], m["fp"]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = 1.0 - fpr
    return float(tpr), float(fpr), float(tnr)


def youden_j(tpr: float, tnr: float) -> float:
    return float(tpr + tnr - 1.0)


def compose_any(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a == 1) | (b == 1)).astype(int)


def compose_both(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a == 1) & (b == 1)).astype(int)


def independence_j(tpr_a: float, fpr_a: float, tpr_b: float, fpr_b: float, mode: str) -> float:
    """
    Independence baseline: assume A,B decisions are independent Bernoulli
    with given TPR/FPR; compute composed TPR/FPR then Youden's J.
    """
    mode = mode.lower().strip()
    if mode == "any":
        tpr = 1.0 - (1.0 - tpr_a) * (1.0 - tpr_b)
        fpr = 1.0 - (1.0 - fpr_a) * (1.0 - fpr_b)
    elif mode == "both":
        tpr = tpr_a * tpr_b
        fpr = fpr_a * fpr_b
    else:
        raise ValueError("mode must be 'any' or 'both'")
    return youden_j(tpr, 1.0 - fpr)


# -----------------------------
# Dependence diagnostics
# -----------------------------


def mutual_information(a_block: np.ndarray, b_block: np.ndarray) -> float:
    """
    2x2 mutual information in bits, computed from empirical frequencies.
    """
    import math

    a = a_block == 1
    b = b_block == 1

    p11 = float(np.mean(a & b))
    p10 = float(np.mean(a & (~b)))
    p01 = float(np.mean((~a) & b))
    p00 = float(np.mean((~a) & (~b)))

    px1 = p11 + p10
    px0 = p01 + p00
    py1 = p11 + p01
    py0 = p10 + p00

    def term(p: float, q: float) -> float:
        return 0.0 if p <= 0.0 or q <= 0.0 else p * math.log(p / q, 2)

    return float(
        term(p11, px1 * py1) + term(p10, px1 * py0) + term(p01, px0 * py1) + term(p00, px0 * py0)
    )


def overlap_ratio(a_block: np.ndarray, b_block: np.ndarray) -> float:
    """
    Jaccard-style overlap of positive decisions: |A∩B| / |A∪B|.
    Safe when union is 0.
    """
    inter = int(((a_block == 1) & (b_block == 1)).sum())
    union = int(((a_block == 1) | (b_block == 1)).sum())
    return float(inter / union) if union else 0.0


# -----------------------------
# Threshold sweep
# -----------------------------


@dataclass(frozen=True)
class SweepBest:
    thr: float
    j: float
    tpr: float
    fpr: float
    tnr: float
    hit_window: bool


def sweep_thr_for_fpr(
    y: np.ndarray,
    scores: np.ndarray,
    fpr_min: float,
    fpr_max: float,
    grid_n: int = 2001,
) -> SweepBest:
    """
    Find threshold maximizing Youden's J subject to fpr_min <= FPR <= fpr_max.

    If *no* threshold hits the window (possible for tiny or degenerate data),
    fall back to the overall best-J threshold and mark hit_window=False.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    # Build a numeric grid that respects score scale (not assuming [0,1]).
    lo = float(np.nanmin(scores))
    hi = float(np.nanmax(scores))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Scores contain no finite values.")

    if lo == hi:
        # Degenerate scores: only one effective classifier; try 2 points.
        grid = np.array([lo, lo + 1e-12], dtype=float)
    else:
        grid = np.linspace(lo, hi, int(grid_n), dtype=float)

    best_overall: Optional[SweepBest] = None
    best_window: Optional[SweepBest] = None

    for thr in grid:
        yhat = binarize(scores, float(thr))
        tpr, fpr, tnr = rates(confusion(y, yhat))
        jj = youden_j(tpr, tnr)

        cand = SweepBest(thr=float(thr), j=float(jj), tpr=tpr, fpr=fpr, tnr=tnr, hit_window=False)

        if (best_overall is None) or (cand.j > best_overall.j):
            best_overall = cand

        if fpr_min <= fpr <= fpr_max:
            cand_w = SweepBest(
                thr=float(thr), j=float(jj), tpr=tpr, fpr=fpr, tnr=tnr, hit_window=True
            )
            if (best_window is None) or (cand_w.j > best_window.j):
                best_window = cand_w

    assert best_overall is not None  # for type-checkers
    return best_window if best_window is not None else best_overall


# -----------------------------
# CLI / IO
# -----------------------------


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")


def run(csv_path: Path, out_path: Path, fpr_min: float, fpr_max: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _require_cols(df, ("label", "rail_a_score", "rail_b_score"))

    y = df["label"].to_numpy(dtype=int)
    a_scores = df["rail_a_score"].to_numpy(dtype=float)
    b_scores = df["rail_b_score"].to_numpy(dtype=float)

    best_a = sweep_thr_for_fpr(y, a_scores, fpr_min=fpr_min, fpr_max=fpr_max)
    best_b = sweep_thr_for_fpr(y, b_scores, fpr_min=fpr_min, fpr_max=fpr_max)

    a_hat = binarize(a_scores, best_a.thr)
    b_hat = binarize(b_scores, best_b.thr)

    any_hat = compose_any(a_hat, b_hat)
    both_hat = compose_both(a_hat, b_hat)

    any_tpr, any_fpr, any_tnr = rates(confusion(y, any_hat))
    both_tpr, both_fpr, both_tnr = rates(confusion(y, both_hat))

    indep_any = independence_j(best_a.tpr, best_a.fpr, best_b.tpr, best_b.fpr, mode="any")
    indep_both = independence_j(best_a.tpr, best_a.fpr, best_b.tpr, best_b.fpr, mode="both")

    # Diagnostics about dependence in the chosen thresholded decisions
    mi_bits = mutual_information(a_hat, b_hat)
    overlap = overlap_ratio(a_hat, b_hat)

    row = {
        # what the test expects:
        "A_J": best_a.j,
        "B_J": best_b.j,
        "Combo_any_J": youden_j(any_tpr, any_tnr),
        "Indep_any_J": indep_any,
        # extra (useful for research/debugging)
        "A_thr": best_a.thr,
        "A_tpr": best_a.tpr,
        "A_fpr": best_a.fpr,
        "A_hit_window": bool(best_a.hit_window),
        "B_thr": best_b.thr,
        "B_tpr": best_b.tpr,
        "B_fpr": best_b.fpr,
        "B_hit_window": bool(best_b.hit_window),
        "Combo_any_tpr": any_tpr,
        "Combo_any_fpr": any_fpr,
        "Combo_both_J": youden_j(both_tpr, both_tnr),
        "Combo_both_tpr": both_tpr,
        "Combo_both_fpr": both_fpr,
        "Indep_both_J": indep_both,
        "MI_bits_A_vs_B": mi_bits,
        "Overlap_pos_Jaccard": overlap,
        "fpr_window_min": float(fpr_min),
        "fpr_window_max": float(fpr_max),
        "n": int(len(df)),
    }

    out_df = pd.DataFrame([row])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare two rails under an FPR window and write summary CSV."
    )
    p.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Input CSV with columns: label, rail_a_score, rail_b_score",
    )
    p.add_argument("--out", required=True, type=Path, help="Output CSV path (summary row).")
    p.add_argument(
        "--fpr-min", type=float, default=0.0, help="Lower bound of FPR window (inclusive)."
    )
    p.add_argument(
        "--fpr-max", type=float, default=0.1, help="Upper bound of FPR window (inclusive)."
    )
    args = p.parse_args()

    if not (
        0.0 <= args.fpr_min <= 1.0 and 0.0 <= args.fpr_max <= 1.0 and args.fpr_min <= args.fpr_max
    ):
        raise SystemExit("Invalid FPR window. Need 0<=fpr-min<=fpr-max<=1.")

    run(args.csv, args.out, fpr_min=args.fpr_min, fpr_max=args.fpr_max)


if __name__ == "__main__":
    main()
