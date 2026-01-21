"""Render Week 5 figures from scan.csv metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

NUMERIC_FIELDS = {
    "tpr_a",
    "tpr_b",
    "fpr_a",
    "fpr_b",
    "I1_lo",
    "I1_hi",
    "I0_lo",
    "I0_hi",
    "vbar1",
    "vbar0",
    "cc_hat",
    "ci_lo",
    "ci_hi",
    "ci_width",
}


def load_scan(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in NUMERIC_FIELDS:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = None
                else:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def find_last_complete_row(rows: list[dict]) -> dict | None:
    for row in reversed(rows):
        if row.get("ci_lo") is not None and row.get("ci_hi") is not None:
            return row
    return rows[-1] if rows else None


def figure_ci_widths(row: dict, out_path: Path) -> None:
    widths = [
        (row.get("I0_hi") or 0) - (row.get("I0_lo") or 0),
        (row.get("I1_hi") or 0) - (row.get("I1_lo") or 0),
        row.get("ci_width") or 0,
    ]
    labels = ["World 0 (Wilson)", "World 1 (Wilson)", "Δ (Bootstrap)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, widths, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Interval width")
    ax.set_title("Wilson vs Bootstrap CI width")
    for rect, width in zip(bars, widths, strict=False):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{width:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_roc_slice(row: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    fpr_a = row.get("fpr_a")
    tpr_a = row.get("tpr_a")
    fpr_b = row.get("fpr_b")
    tpr_b = row.get("tpr_b")

    ax.axvline(0.05, color="red", linestyle="--", label="α cap = 0.05")
    if fpr_a is not None and tpr_a is not None:
        ax.scatter([fpr_a], [tpr_a], color="#4C72B0", label="Guardrail (A)")
    if fpr_b is not None and tpr_b is not None:
        ax.scatter([fpr_b], [tpr_b], color="#55A868", label="Baseline (B)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC slice at calibrated threshold")
    ax.set_xlim(0, max(0.1, (fpr_a or 0.05) * 1.5))
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Week 5 diagnostic figures")
    parser.add_argument(
        "--scan", default="results/week5_scan/scan.csv", help="CSV produced by run_two_world"
    )
    parser.add_argument(
        "--out-ci", default="figures/week5_ci_comparison.png", help="Path to CI comparison figure"
    )
    parser.add_argument(
        "--out-roc", default="figures/week5_roc_slice.png", help="Path to ROC slice figure"
    )
    args = parser.parse_args()

    scan_path = Path(args.scan)
    rows = load_scan(scan_path)
    if not rows:
        raise SystemExit(f"No rows found in {scan_path}")
    last = find_last_complete_row(rows)
    if last is None:
        raise SystemExit("No complete metrics row found")

    figure_ci_widths(last, Path(args.out_ci))
    figure_roc_slice(last, Path(args.out_roc))


if __name__ == "__main__":
    main()
