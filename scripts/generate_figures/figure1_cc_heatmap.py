"""Generate a CC_max heatmap from pairwise composition results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def _build_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    required = {"guardrail_a", "guardrail_b", "cc_max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    guardrails = sorted(set(df["guardrail_a"]).union(df["guardrail_b"]))
    index: Dict[str, int] = {name: i for i, name in enumerate(guardrails)}
    matrix = np.full((len(guardrails), len(guardrails)), np.nan)

    for _, row in df.iterrows():
        i = index[row["guardrail_a"]]
        j = index[row["guardrail_b"]]
        matrix[i, j] = float(row["cc_max"])
        matrix[j, i] = float(row["cc_max"])

    np.fill_diagonal(matrix, 1.0)
    return guardrails, matrix


def generate_cc_heatmap(guardrails: List[str], cc_matrix: np.ndarray, output_path: Path) -> None:
    n = len(guardrails)
    regime_matrix = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            cc = cc_matrix[i, j]
            if np.isnan(cc):
                regime_matrix[i, j] = "—"
            elif cc < 0.95:
                regime_matrix[i, j] = "C"
            elif cc <= 1.05:
                regime_matrix[i, j] = "I"
            else:
                regime_matrix[i, j] = "D"

    colors = ["#2E7D32", "#81C784", "#E0E0E0", "#EF9A9A", "#C62828"]
    cmap = LinearSegmentedColormap.from_list("cc_cmap", colors, N=100)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cc_matrix,
        annot=regime_matrix,
        fmt="s",
        cmap=cmap,
        center=1.0,
        vmin=0.75,
        vmax=1.25,
        xticklabels=guardrails,
        yticklabels=guardrails,
        cbar_kws={"label": r"$\mathrm{CC}_{\max}$", "shrink": 0.8},
        linewidths=0.5,
        linecolor="black",
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("Guardrail 2", fontsize=14, fontweight="bold")
    ax.set_ylabel("Guardrail 1", fontsize=14, fontweight="bold")
    ax.set_title(
        r"Compositional Coupling Matrix ($\mathrm{CC}_{\max}$)\n"
        "C = Constructive, I = Independent, D = Destructive",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CC_max heatmap from CSV.")
    parser.add_argument("--input", required=True, help="CSV with guardrail_a, guardrail_b, cc_max")
    parser.add_argument(
        "--output",
        default="paper/figures/figure1_cc_heatmap.pdf",
        help="Output path for the figure",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    guardrails, matrix = _build_matrix(df)
    generate_cc_heatmap(guardrails, matrix, Path(args.output))


if __name__ == "__main__":
    main()
