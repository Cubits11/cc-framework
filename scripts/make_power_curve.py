#!/usr/bin/env python3
# scripts/make_power_curve.py
# FH–Bernstein CI half-width for CC as a function of (n1, n0)
# - Robust inversion of the Bernstein tail for each class
# - Sorted/filtered contour levels (no matplotlib errors)
# - Optional target half-width with recommended (n1*, n0*) marker
# - Memo-friendly text summary to STDOUT

import argparse
import math
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- math utils ----------------------------- #

def var_envelope(lo: float, hi: float) -> float:
    """
    Worst-case Bernoulli variance over an interval [lo, hi].
    """
    if lo <= 0.5 <= hi:
        return 0.25
    return max(lo * (1 - lo), hi * (1 - hi))


def bernstein_tail(eps: float, n: int, vbar: float) -> float:
    """
    Two-sided Bernstein tail bound (dependence-agnostic) for a Bernoulli mean.
    P(|p̂ - p| >= eps) <= 2 * exp(- n * eps^2 / (2 vbar + 2 eps / 3))
    """
    if eps <= 0 or n <= 0 or vbar < 0:
        return 1.0
    denom = 2.0 * vbar + (2.0 / 3.0) * eps
    return 2.0 * math.exp(-n * (eps ** 2) / denom)


def invert_bernstein_for_eps(n: int, vbar: float, delta: float,
                             eps_hi: float = 1.0) -> float:
    """
    Solve for eps >= 0 such that bernstein_tail(eps, n, vbar) == delta
    using monotone bisection. Returns the *smallest* eps satisfying tail <= delta.
    """
    if n <= 0:
        return float("nan")
    # Tail is decreasing in eps; ensure the right bracket actually achieves <= delta.
    lo, hi = 0.0, max(1e-12, eps_hi)
    if bernstein_tail(hi, n, vbar) > delta:  # insufficient hi; expand until it works
        for _ in range(40):
            hi *= 2.0
            if hi > 1e3:  # safety
                break
            if bernstein_tail(hi, n, vbar) <= delta:
                break
    # Bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bernstein_tail(mid, n, vbar) > delta:
            lo = mid  # need larger eps
        else:
            hi = mid
    return hi


def cc_halfwidth(n1: int, n0: int, v1: float, v0: float, D: float, delta: float) -> float:
    """
    Half-width t for CC = (1 - (p1 - p0)) / D, with per-class delta/2 allocation.
    """
    if D <= 0:
        return float("nan")
    e1 = invert_bernstein_for_eps(n1, v1, delta / 2.0)
    e0 = invert_bernstein_for_eps(n0, v0, delta / 2.0)
    return (e1 + e0) / D


# ------------------------------ plotting ------------------------------ #

def make_levels(minv: float, maxv: float) -> np.ndarray:
    """
    Choose a reasonable set of increasing contour levels within [minv, maxv].
    """
    # Candidate grid; we’ll keep only those that fall inside the range.
    candidates = np.array([0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40])
    levels = candidates[(candidates > minv) & (candidates < maxv)]
    # Guarantee strictly increasing and at least a couple of levels:
    if levels.size < 2:
        # fallback: make 5 evenly spaced levels
        levels = np.linspace(minv, maxv, num=5, endpoint=False)[1:]
    return np.unique(np.sort(levels))


def find_recommendation(H: np.ndarray, n1_vals: np.ndarray, n0_vals: np.ndarray,
                        target_t: Optional[float]) -> Optional[Tuple[int, int, float]]:
    """
    If target_t is provided, find the (n1, n0) with minimal n1 + n0
    such that half-width <= target_t. Returns (n1*, n0*, t*), or None.
    """
    if target_t is None:
        return None
    mask = (H <= target_t)
    if not mask.any():
        return None
    # Among feasible grid cells, minimize total samples (n1 + n0).
    best_idx = None
    best_cost = None
    best_t = None
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if not mask[i, j]:
                continue
            cost = n1_vals[i] + n0_vals[j]
            if (best_cost is None) or (cost < best_cost) or (cost == best_cost and H[i, j] < best_t):
                best_cost = cost
                best_idx = (i, j)
                best_t = H[i, j]
    i, j = best_idx
    return int(n1_vals[i]), int(n0_vals[j]), float(best_t)


# ------------------------------ main CLI ------------------------------ #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Power curve heatmap: FH–Bernstein CC half-width vs (n1, n0)."
    )
    ap.add_argument("--I1", required=True, help="lo,hi for Y=1 (e.g., 0.37,0.65)")
    ap.add_argument("--I0", required=True, help="lo,hi for Y=0 (e.g., 0.05,0.05)")
    ap.add_argument("--D", type=float, required=True, help="Normalization constant D > 0")
    ap.add_argument("--delta", type=float, default=0.05, help="Two-sided failure prob (default 0.05)")
    ap.add_argument("--target-t", type=float, default=None,
                    help="Optional target half-width to highlight/annotate (e.g., 0.10)")
    ap.add_argument("--out", default="paper/figures/fig_week3_power_curve.png")
    ap.add_argument("--n1min", type=int, default=100)
    ap.add_argument("--n1max", type=int, default=1500)
    ap.add_argument("--n0min", type=int, default=50)
    ap.add_argument("--n0max", type=int, default=500)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--cmap", default="cividis")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not (0 < args.delta < 1):
        raise ValueError("--delta must be in (0,1)")

    I1 = tuple(float(x) for x in args.I1.split(","))
    I0 = tuple(float(x) for x in args.I0.split(","))
    if len(I1) != 2 or len(I0) != 2:
        raise ValueError("Provide --I1 and --I0 as 'lo,hi'")

    v1 = var_envelope(*I1)
    v0 = var_envelope(*I0)

    n1_vals = np.arange(args.n1min, args.n1max + 1, args.step, dtype=int)
    n0_vals = np.arange(args.n0min, args.n0max + 1, args.step, dtype=int)

    # Compute half-width grid
    H = np.zeros((len(n1_vals), len(n0_vals)), dtype=float)
    for i, n1 in enumerate(n1_vals):
        for j, n0 in enumerate(n0_vals):
            H[i, j] = cc_halfwidth(n1, n0, v1, v0, args.D, args.delta)

    # Plot
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(
        H,
        origin="lower",
        aspect="auto",
        cmap=args.cmap,
        extent=[n0_vals[0], n0_vals[-1], n1_vals[0], n1_vals[-1]],
    )

    # Contours (sorted, filtered)
    levels = make_levels(float(H.min()), float(H.max()))
    cs = ax.contour(n0_vals, n1_vals, H, levels=levels, colors="k", linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{v:.2f}", inline=True, fontsize=8)

    # Optional target
    rec = find_recommendation(H, n1_vals, n0_vals, args.target_t)
    if rec is not None:
        n1_star, n0_star, t_star = rec
        ax.scatter([n0_star], [n1_star], s=55, marker="*", color="white",
                   edgecolor="black", linewidths=0.7, zorder=5)
        ax.text(n0_star + 6, n1_star + 6, f"(n1*={n1_star}, n0*={n0_star})\n t≈{t_star:.3f}",
                fontsize=8, color="black", bbox=dict(facecolor="white", alpha=0.6, lw=0.0))
        # Highlight the target contour if it exists
        if args.target_t is not None:
            try:
                ax.contour(n0_vals, n1_vals, H, levels=[args.target_t],
                           colors="magenta", linewidths=1.6, linestyles="--")
            except Exception:
                pass

    ax.set_xlabel("n0  (Y=0 samples)")
    ax.set_ylabel("n1  (Y=1 samples)")
    ax.set_title("FH–Bernstein half-width for CC  (smaller is better)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CI half-width  t")

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}")
    print(f"I1={I1} ⇒ v̄1={v1:.4f}   I0={I0} ⇒ v̄0={v0:.4f}   D={args.D:.3f}   δ={args.delta:.3f}")
    if rec is None and args.target_t is not None:
        print(f"Target t={args.target_t:.3f}: no feasible (n1,n0) in the scanned grid.")
    elif rec is not None:
        n1_star, n0_star, t_star = rec
        print(f"Recommended (min n1+n0) for t ≤ {args.target_t:.3f}: "
              f"n1*={n1_star}, n0*={n0_star}, achieved t≈{t_star:.3f}")


if __name__ == "__main__":
    main()
