#!/usr/bin/env python
"""
Week-6 figures:
- delta_bar.png : Δ (world0 - world1) bars per rail
- roc_grid.png  : ROC points per rail with α=0.05 line
- table.csv     : rail, TPR, FPR, Δ, CI_lo, CI_hi, CC_max

Usage:
  python scripts/make_week6_figs.py \
    --inputs results/week6/keyword/analysis.json [more ...] \
    --out-dir figures/week6/keyword
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt


def load_analysis(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rail_name_from_analysis(data: Dict[str, Any]) -> str:
    try:
        memo_tag = data["metadata"]["configuration"]["experiment"].get("memo_tag")
        if memo_tag:
            return memo_tag
    except Exception:
        pass
    # fallback: derive from file path later if necessary
    return "rail"


def extract_metrics(data: Dict[str, Any]) -> Tuple[float, float, float, Tuple[float, float], float]:
    res = data["results"]
    op = res["operating_points"]
    j = res["j_statistic"]
    tpr = float(op["world_1"]["tpr"])
    fpr = float(op["world_1"]["fpr"])
    delta = float(j["empirical"])
    ci = (float(j["confidence_interval"]["lower"]), float(j["confidence_interval"]["upper"]))
    cc_max = float(res["composability_metrics"]["cc_max"])
    return tpr, fpr, delta, ci, cc_max


def make_delta_bar(rows: List[Tuple[str, float, Tuple[float, float]]], out: Path) -> None:
    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    ci_lo = [r[2][0] for r in rows]
    ci_hi = [r[2][1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = range(len(labels))
    bars = ax.bar(xs, deltas)
    for i, rect in enumerate(bars):
        ax.errorbar(rect.get_x() + rect.get_width() / 2.0, deltas[i],
                    yerr=[[deltas[i] - ci_lo[i]], [ci_hi[i] - deltas[i]]],
                    fmt="none", capsize=4)
    ax.set_xticks(list(xs), labels, rotation=15, ha="right")
    ax.set_ylabel("Δ = success(World0) - success(World1)")
    ax.set_title("Guardrail Δ with 95% CI")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def make_roc_grid(rows: List[Tuple[str, float, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axvline(0.05, linestyle="--", label="α cap = 0.05")
    for name, tpr, fpr in rows:
        ax.scatter([fpr], [tpr], label=name)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title("ROC points at calibrated α-cap")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Week-6 figure generator")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more analysis.json files")
    ap.add_argument("--out-dir", required=True, help="Output directory for figures & table")
    ap.add_argument("--scan", help="Optional scan.csv for CI width panels (not required)")
    ap.add_argument("--roc-grid", action="store_true", help="Force ROC grid even for single rail")
    ap.add_argument("--delta-bars", action="store_true", help="Force delta bars even for single rail")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Tuple[str, float, float, Tuple[float, float], float]] = []
    for p in args.inputs:
        data = load_analysis(Path(p))
        name = rail_name_from_analysis(data)
        tpr, fpr, delta, ci, cc = extract_metrics(data)
        summary_rows.append((name, tpr, fpr, ci, cc))

    # Write table.csv
    table_path = out_dir / "table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rail", "TPR", "FPR", "Δ", "CI_lo", "CI_hi", "CC_max"])
        for (name, tpr, fpr, ci, cc) in summary_rows:
            delta_est = None  # Δ printed separately from analysis if needed
            # We don't store delta_est here; instead we compute from CI center:
            delta_est = (ci[0] + ci[1]) / 2.0  # just a display proxy
            writer.writerow([name, tpr, fpr, delta_est, ci[0], ci[1], cc])

    # Figures
    if len(summary_rows) == 1 and not (args.delta_bars or args.roc_grid):
        # Single-rail default: make both
        args.delta_bars = True
        args.roc_grid = True

    if args.delta_bars:
        # For delta bars, we prefer using analysis Δ and CI — reload accurately
        rows_for_bars = []
        for p in args.inputs:
            data = load_analysis(Path(p))
            name = rail_name_from_analysis(data)
            j = data["results"]["j_statistic"]
            delta = float(j["empirical"])
            ci = (float(j["confidence_interval"]["lower"]), float(j["confidence_interval"]["upper"]))
            rows_for_bars.append((name, delta, ci))
        make_delta_bar(rows_for_bars, out_dir / "delta_bar.png")

    if args.roc_grid:
        rows_for_roc = [(name, tpr, fpr) for (name, tpr, fpr, _ci, _cc) in summary_rows]
        make_roc_grid(rows_for_roc, out_dir / "roc_grid.png")


if __name__ == "__main__":
    main()