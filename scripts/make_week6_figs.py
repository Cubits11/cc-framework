#!/usr/bin/env python
"""
Week-6 figures (upgraded):
- delta_bar.png : Δ (world0 - world1) bars per rail with 95% CI
- roc_grid.png  : ROC points per rail with α-cap line (auto-read) and annotations
- table.csv     : rail, TPR, FPR, Δ(empirical), CI_lo, CI_hi, CC_max

Usage:
  python scripts/make_week6_figs.py \
    --inputs results/week6/keyword/analysis.json [more ...] \
    --out-dir figures/week6/keyword

Nice-to-haves:
  --alpha-cap 0.05       # override α line if not present in inputs
  --sort delta           # sort bars by: name|delta|tpr|fpr (default: name)
  --annotate             # annotate ROC points with (TPR,FPR)
  --svg                  # also write .svg versions of the figures
  --roc-grid             # force ROC grid even for single rail
  --delta-bars           # force Δ bars even for single rail
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import math
import matplotlib.pyplot as plt


# ---------------------------
# IO + parsing
# ---------------------------

def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    try:
        for k in path:
            cur = cur[k]
        return cur
    except (KeyError, TypeError):
        return default


def load_analysis(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rail_name_from_analysis(data: Dict[str, Any], fallback: str) -> str:
    # prefer explicit memo tag if set in experiment config
    memo_tag = _safe_get(data, ["metadata", "configuration", "experiment", "memo_tag"])
    if memo_tag:
        return str(memo_tag)

    # try guardrail name
    gr0_name = _safe_get(data, ["metadata", "configuration", "guardrails", 0, "name"])
    if gr0_name:
        return str(gr0_name)

    # fallback to file stem or "rail"
    return fallback or "rail"


def extract_metrics(data: Dict[str, Any]) -> Tuple[float, float, float, Tuple[float, float], float, Optional[float]]:
    """
    Returns: tpr, fpr, delta_empirical, (ci_lo, ci_hi), cc_max, alpha_cap
    """
    res = _safe_get(data, ["results"], {})
    op = _safe_get(res, ["operating_points"], {})
    j = _safe_get(res, ["j_statistic"], {})

    tpr = float(_safe_get(op, ["world_1", "tpr"], 0.0))
    fpr = float(_safe_get(op, ["world_1", "fpr"], 0.0))

    delta = float(_safe_get(j, ["empirical"], tpr - fpr))
    ci_lo = float(_safe_get(j, ["confidence_interval", "lower"], min(delta, 0.0)))
    ci_hi = float(_safe_get(j, ["confidence_interval", "upper"], max(delta, 0.0)))
    if ci_lo > ci_hi:
        ci_lo, ci_hi = ci_hi, ci_lo  # sanitize if inverted

    cc_max = float(_safe_get(res, ["composability_metrics", "cc_max"], float("nan")))

    alpha_cap = _safe_get(
        data, ["metadata", "configuration", "guardrails", 0, "params", "alpha_cap"], None
    )
    alpha_cap = float(alpha_cap) if alpha_cap is not None else None

    return tpr, fpr, delta, (ci_lo, ci_hi), cc_max, alpha_cap


# ---------------------------
# Plotting
# ---------------------------

def make_delta_bar(
    rows: List[Tuple[str, float, Tuple[float, float]]],
    out_png: Path,
    out_svg: Optional[Path] = None,
    title: str = "Guardrail Δ with 95% CI",
) -> None:
    if not rows:
        return

    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    ci_lo = [r[2][0] for r in rows]
    ci_hi = [r[2][1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels)), 4))
    xs = range(len(labels))
    bars = ax.bar(xs, deltas)

    for i, rect in enumerate(bars):
        # asymmetric error bars
        lower = max(0.0, deltas[i] - ci_lo[i]) if not math.isnan(ci_lo[i]) else 0.0
        upper = max(0.0, ci_hi[i] - deltas[i]) if not math.isnan(ci_hi[i]) else 0.0
        if lower == upper == 0.0:
            continue
        ax.errorbar(
            rect.get_x() + rect.get_width() / 2.0,
            deltas[i],
            yerr=[[lower], [upper]],
            fmt="none",
            capsize=4,
        )

    ax.set_xticks(list(xs), labels, rotation=15, ha="right")
    ax.set_ylabel("Δ = success(World0) - success(World1)")
    ax.axhline(0.0, linewidth=1, linestyle=":", alpha=0.5)
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)


def make_roc_grid(
    rows: List[Tuple[str, float, float]],
    out_png: Path,
    out_svg: Optional[Path] = None,
    alpha_cap: Optional[float] = 0.05,
    annotate: bool = False,
) -> None:
    if not rows:
        return

    # dynamic x-limit with margin, capped at 0.5
    max_fpr = max((f for _n, _t, f in rows), default=0.12)
    xlim = min(0.5, max(0.12, max_fpr * 1.15))

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    if alpha_cap is not None:
        ax.axvline(alpha_cap, linestyle="--", label=f"α cap = {alpha_cap:g}")

    for name, tpr, fpr in rows:
        ax.scatter([fpr], [tpr], label=name)
        if annotate:
            ax.annotate(f"({tpr:.2f}, {fpr:.3f})", (fpr, tpr), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, alpha=0.8)

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 1.0)
    # place legend outside if many rails
    if len(rows) > 6:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(right=0.78)
    else:
        ax.legend()
    ax.set_title("ROC points at calibrated α-cap" if alpha_cap is not None else "ROC points")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Week-6 figure generator (upgraded)")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more analysis.json files")
    ap.add_argument("--out-dir", required=True, help="Output directory for figures & table")
    ap.add_argument("--alpha-cap", type=float, default=None,
                    help="Optional α-cap to draw if not found in inputs")
    ap.add_argument("--sort", choices=["name", "delta", "tpr", "fpr"], default="name",
                    help="Sort order for Δ bars (default: name)")
    ap.add_argument("--annotate", action="store_true",
                    help="Annotate ROC points with (TPR,FPR)")
    ap.add_argument("--svg", action="store_true",
                    help="Also write SVG versions of figures")
    ap.add_argument("--roc-grid", action="store_true",
                    help="Force ROC grid even for single rail")
    ap.add_argument("--delta-bars", action="store_true",
                    help="Force Δ bars even for single rail")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows
    summary_rows: List[Tuple[str, float, float, float, Tuple[float, float], float, Optional[float]]] = []
    for p in args.inputs:
        pth = Path(p)
        data = load_analysis(pth)
        name = rail_name_from_analysis(data, pth.stem)
        tpr, fpr, delta, ci, cc, alpha_cap = extract_metrics(data)
        summary_rows.append((name, tpr, fpr, delta, ci, cc, alpha_cap))

    if not summary_rows:
        print("No inputs provided / parsed; nothing to do.")
        return

    # Determine alpha cap to draw: prefer first non-None from inputs, else CLI, else None
    inferred_alpha = next((a for *_x, a in summary_rows if a is not None), None)
    alpha_to_draw = inferred_alpha if inferred_alpha is not None else args.alpha_cap

    # Sorting for bar chart
    key_funcs = {
        "name": lambda r: r[0].lower(),
        "delta": lambda r: r[3],
        "tpr": lambda r: r[1],
        "fpr": lambda r: r[2],
    }
    sorted_rows = sorted(summary_rows, key=key_funcs[args.sort])

    # ---------------- Write table.csv ----------------
    table_path = out_dir / "table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rail", "TPR", "FPR", "Δ_empirical", "CI_lo", "CI_hi", "CC_max", "alpha_cap"])
        for (name, tpr, fpr, delta, ci, cc, a) in sorted_rows:
            writer.writerow([name, tpr, fpr, delta, ci[0], ci[1], cc, a])

    # Decide which figures to render by default
    if len(summary_rows) == 1 and not (args.delta_bars or args.roc_grid):
        # Single-rail default: make both
        args.delta_bars = True
        args.roc_grid = True

    # ---------------- Δ bars ----------------
    if args.delta_bars:
        rows_for_bars = [(name, delta, ci) for (name, _tpr, _fpr, delta, ci, _cc, _a) in sorted_rows]
        make_delta_bar(
            rows_for_bars,
            out_png=out_dir / "delta_bar.png",
            out_svg=(out_dir / "delta_bar.svg") if args.svg else None,
        )

    # ---------------- ROC grid ----------------
    if args.roc_grid:
        rows_for_roc = [(name, tpr, fpr) for (name, tpr, fpr, _delta, _ci, _cc, _a) in summary_rows]
        make_roc_grid(
            rows_for_roc,
            out_png=out_dir / "roc_grid.png",
            out_svg=(out_dir / "roc_grid.svg") if args.svg else None,
            alpha_cap=alpha_to_draw,
            annotate=args.annotate,
        )


if __name__ == "__main__":
    main()
