"""Generate FH CC-interval plot from a CSV input."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def generate_fh_bounds_plot(df: pd.DataFrame, output_path: Path) -> None:
    required = {"pair", "cc_lower", "cc_point", "cc_upper"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pairs = list(df["pair"])
    fig, ax = plt.subplots(figsize=(12, 8))

    n_pairs = len(pairs)
    for i, row in enumerate(df.itertuples(index=False), start=0):
        y = n_pairs - i - 1
        cc_lower = float(row.cc_lower)
        cc_point = float(row.cc_point)
        cc_upper = float(row.cc_upper)

        overlaps_ind = not (cc_upper < 0.95 or cc_lower > 1.05)
        color = "orange" if overlaps_ind else "steelblue"

        ax.plot([cc_lower, cc_upper], [y, y], "o-", linewidth=3, markersize=8, color=color, alpha=0.7)
        ax.plot([cc_point], [y], "X", markersize=12, color="darkred", zorder=10)
        ax.text(-0.05, y, row.pair, ha="right", va="center", fontsize=11, fontweight="bold")

    ax.axvspan(0, 0.95, alpha=0.1, color="green", label="Constructive")
    ax.axvspan(0.95, 1.05, alpha=0.1, color="gray", label="Independent")
    ax.axvspan(1.05, 1.5, alpha=0.1, color="red", label="Destructive")
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)

    ax.set_xlabel(r"$\mathrm{CC}_{\max}$", fontsize=14, fontweight="bold")
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels([])
    ax.set_xlim(0.7, 1.3)
    ax.set_ylim(-0.5, n_pairs - 0.5)

    ax.set_title(
        "Regime Sensitivity via Fr\'echet--Hoeffding Bounds\n"
        "Intervals show range under unknown dependence",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    legend_elements = [
        Patch(facecolor="green", alpha=0.1, label="Constructive"),
        Patch(facecolor="gray", alpha=0.1, label="Independent"),
        Patch(facecolor="red", alpha=0.1, label="Destructive"),
        Line2D([0], [0], marker="o", color="steelblue", linestyle="-", markersize=8, linewidth=3, label="FH Envelope (certain)"),
        Line2D([0], [0], marker="o", color="orange", linestyle="-", markersize=8, linewidth=3, label="FH Envelope (uncertain)"),
        Line2D([0], [0], marker="X", color="darkred", linestyle="", markersize=12, label="Point Estimate"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.grid(axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FH CC-interval plot from CSV.")
    parser.add_argument("--input", required=True, help="CSV with pair, cc_lower, cc_point, cc_upper")
    parser.add_argument(
        "--output",
        default="paper/figures/figure2_fh_bounds.pdf",
        help="Output path for the figure",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    generate_fh_bounds_plot(df, Path(args.output))


if __name__ == "__main__":
    main()
