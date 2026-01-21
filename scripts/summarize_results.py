#!/usr/bin/env python
"""
Utility summarizer:
--utility mode reads a final_results.json and produces:
  - CSV of per-world mean±sd for utility_score
  - Histogram PNG overlaying world 0 vs world 1

Expected input structure (generic):
[
  {"world_bit": 0, "utility_score": <float>, ...},
  {"world_bit": 1, "utility_score": <float>, ...},
  ...
]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_final_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported final_results.json structure")


def compute_stats(records: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    world_bins: dict[int, list[float]] = {0: [], 1: []}
    for r in records:
        wb = int(r.get("world_bit", -1))
        if wb not in (0, 1):
            continue
        if "utility_score" in r and r["utility_score"] is not None:
            world_bins[wb].append(float(r["utility_score"]))
    stats: dict[int, dict[str, float]] = {}
    for wb, xs in world_bins.items():
        if xs:
            arr = np.array(xs, dtype=float)
            stats[wb] = {
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            }
        else:
            stats[wb] = {"n": 0, "mean": float("nan"), "sd": float("nan")}
    return stats


def write_csv(stats: dict[int, dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["world", "n", "mean", "sd"])
        for wb in (0, 1):
            s = stats.get(wb, {"n": 0, "mean": float("nan"), "sd": float("nan")})
            w.writerow([wb, s["n"], s["mean"], s["sd"]])


def make_histogram(
    stats: dict[int, dict[str, float]], records: list[dict[str, Any]], out_png: Path
) -> None:
    x0 = [
        float(r["utility_score"])
        for r in records
        if int(r.get("world_bit", -1)) == 0 and "utility_score" in r
    ]
    x1 = [
        float(r["utility_score"])
        for r in records
        if int(r.get("world_bit", -1)) == 1 and "utility_score" in r
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 20
    if x0:
        ax.hist(x0, bins=bins, alpha=0.5, label="World 0")
    if x1:
        ax.hist(x1, bins=bins, alpha=0.5, label="World 1")
    ax.set_title("Utility Score Distribution by World")
    ax.set_xlabel("utility_score")
    ax.set_ylabel("count")
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize results artifacts")
    ap.add_argument("--utility", action="store_true", help="Compute per-world utility summary")
    ap.add_argument("--final-results", help="Path to final_results.json")
    ap.add_argument("--out-csv", help="Output CSV path")
    ap.add_argument("--out-fig", help="Output histogram PNG")
    args = ap.parse_args()

    if args.utility:
        if not args.final_results or not args.out_csv or not args.out_fig:
            raise SystemExit("ERROR: --utility requires --final-results, --out-csv, and --out-fig.")
        recs = load_final_results(Path(args.final_results))
        stats = compute_stats(recs)
        write_csv(stats, Path(args.out_csv))
        make_histogram(stats, recs, Path(args.out_fig))
        print(f"Wrote {args.out_csv} and {args.out_fig}")
        return

    raise SystemExit("No mode selected. Use --utility.")


if __name__ == "__main__":
    main()
