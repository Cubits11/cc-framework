#!/usr/bin/env python
"""Generate the Week 7 diagnostic figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from cc.analysis.week7_utils import PointRecord, aggregate_by_group, summarise_group


FIG_NAMES = [
    "roc_grid_or.png",
    "roc_grid_and.png",
    "j_bands_or.png",
    "j_bands_and.png",
    "ci_diagnostics.png",
    "regime_map_or.png",
    "regime_map_and.png",
]


def load_point(path: Path) -> PointRecord:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    fh_env = raw.get("fh_envelope", {})
    independence = raw.get("independence", {})
    classification = raw.get("classification", {})
    wilson = raw.get("wilson", {})
    bca = raw.get("bca", {})

    return PointRecord(
        topology=raw["topology"],
        rails=tuple(raw["rails"]),
        thresholds=dict(raw["thresholds"]),
        seed=int(raw["seed"]),
        episodes=int(raw["episodes"]),
        empirical_tpr=float(raw["empirical"]["tpr"]),
        empirical_fpr=float(raw["empirical"]["fpr"]),
        empirical_j=float(raw["empirical"]["j"]),
        delta_j=float(raw["empirical"]["j"] - max(m["j"] for m in raw["per_rail"].values())),
        independence_tpr=float(independence.get("tpr", 0.0)),
        independence_fpr=float(independence.get("fpr", 0.0)),
        independence_j=float(independence.get("j", 0.0)),
        fh_tpr_lower=float(fh_env.get("tpr_lower", 0.0)),
        fh_tpr_upper=float(fh_env.get("tpr_upper", 0.0)),
        fh_fpr_lower=float(fh_env.get("fpr_lower", 0.0)),
        fh_fpr_upper=float(fh_env.get("fpr_upper", 0.0)),
        fh_j_lower=float(fh_env.get("j_lower", 0.0)),
        fh_j_upper=float(fh_env.get("j_upper", 0.0)),
        classification=str(classification.get("label", "independent")),
        cc_l=float(classification["cc_l"]) if classification.get("cc_l") is not None else float("nan"),
        d_lamp=bool(classification.get("d_lamp", False)),
        wilson_world0_width=float(wilson.get("world0", {}).get("width", 0.0)),
        wilson_world1_width=float(wilson.get("world1", {}).get("width", 0.0)),
        bca_delta_width=float(bca.get("delta", {}).get("width", 0.0)),
        bca_j_width=float(bca.get("j", {}).get("width", 0.0)),
    )


def load_points(directory: Path) -> List[PointRecord]:
    return [load_point(path) for path in sorted(directory.glob("point_*.json"))]


def ensure_output(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _style_for_classification(label: str) -> Dict[str, object]:
    colors = {
        "constructive": (0.0, 0.45, 0.74),
        "independent": (0.2, 0.6, 0.2),
        "destructive": (0.75, 0.15, 0.1),
    }
    return {"color": colors.get(label, (0.5, 0.5, 0.5))}


def plot_roc_grid(points: List[PointRecord], topology: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    grouped = aggregate_by_group([p for p in points if p.topology == topology])

    for key, entry in grouped.items():
        summary = summarise_group(entry)
        fpr_mean = summary["empirical_fpr"]["mean"]
        tpr_mean = summary["empirical_tpr"]["mean"]
        indep_fpr = summary["independence_fpr"]["mean"]
        indep_tpr = summary["independence_tpr"]["mean"]
        fh_l = summary["fh_fpr_lower"]["mean"]
        fh_u = summary["fh_fpr_upper"]["mean"]
        fh_t_l = summary["fh_tpr_lower"]["mean"]
        fh_t_u = summary["fh_tpr_upper"]["mean"]
        regime = max(summary["classification"], key=summary["classification"].get)
        style = _style_for_classification(regime)

        ax.scatter([fpr_mean], [tpr_mean], marker="o", label=key, **style)
        ax.scatter([indep_fpr], [indep_tpr], marker="D", facecolors="none", edgecolors=style["color"], s=60)
        ax.add_patch(
            plt.Rectangle((fh_l, fh_t_l), fh_u - fh_l, fh_t_u - fh_t_l, fill=False, linestyle=":", edgecolor=style["color"], alpha=0.8)
        )

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC grid ({'OR' if topology == 'serial_or' else 'AND'})")
    ax.set_xlim(0.0, 0.12)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle=":", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_j_bands(points: List[PointRecord], topology: str, out_path: Path) -> None:
    grouped = aggregate_by_group([p for p in points if p.topology == topology])
    labels = []
    bands = []
    indep = []
    empirical = []
    for key, entry in grouped.items():
        summary = summarise_group(entry)
        labels.append(key)
        bands.append((summary["fh_j_lower"]["mean"], summary["fh_j_upper"]["mean"]))
        indep.append(summary["independence_j"]["mean"])
        empirical.append(summary["empirical_j"]["mean"])

    fig, ax = plt.subplots(figsize=(0.5 + 0.45 * len(labels), 4.2))
    xs = np.arange(len(labels))
    for idx, (lo, hi) in enumerate(bands):
        ax.plot([idx, idx], [lo, hi], color="0.3", linewidth=3)
        ax.scatter([idx], [indep[idx]], marker="D", color="tab:orange", label="Independence" if idx == 0 else "")
        ax.scatter([idx], [empirical[idx]], marker="o", color="tab:blue", label="Empirical" if idx == 0 else "")
    ax.axhline(0.0, linestyle=":", color="0.5")
    ax.set_xticks(xs, labels, rotation=30, ha="right")
    ax.set_ylabel("J statistic")
    ax.set_title(f"J bands ({'OR' if topology == 'serial_or' else 'AND'})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ci_diagnostics(points: List[PointRecord], out_path: Path) -> None:
    grouped = aggregate_by_group(points)
    items = list(grouped.items())
    summaries = [summarise_group(entry) for _, entry in items]
    widths_world0 = [summary["wilson_world0_width"]["mean"] for summary in summaries]
    widths_world1 = [summary["wilson_world1_width"]["mean"] for summary in summaries]
    widths_delta = [summary["bca_delta_width"]["mean"] for summary in summaries]
    widths_j = [summary["bca_j_width"]["mean"] for summary in summaries]

    labels = [key for key, _ in items]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(0.6 + 0.5 * len(labels), 4.5))
    ax.bar(x - 0.3, widths_world0, width=0.2, label="Wilson (World0)")
    ax.bar(x - 0.1, widths_world1, width=0.2, label="Wilson (World1)")
    ax.bar(x + 0.1, widths_delta, width=0.2, label="BCa Δ")
    ax.bar(x + 0.3, widths_j, width=0.2, label="BCa J")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("CI width")
    ax.set_title("CI diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_regime_map(points: List[PointRecord], topology: str, out_path: Path) -> None:
    relevant = [p for p in points if p.topology == topology]
    groups = aggregate_by_group(relevant)
    summaries = {key: summarise_group(entry) for key, entry in groups.items()}

    keys = list(summaries.keys())
    delta_means = np.array([summaries[key]["delta_j"]["mean"] for key in keys])

    fig, ax = plt.subplots(figsize=(0.6 + 0.4 * len(keys), 4.5))
    cmap = plt.get_cmap("coolwarm")
    colors = cmap((delta_means - delta_means.min()) / (np.ptp(delta_means) + 1e-9))
    positions = np.arange(len(keys))
    bars = ax.bar(positions, delta_means, color=colors)
    for idx, key in enumerate(keys):
        d_rate = summaries[key]["d_lamp_rate"]
        if d_rate > 0:
            bars[idx].set_hatch("//")
    ax.axhline(0.0, color="0.4", linestyle=":")
    ax.set_ylabel("ΔJ (mean)")
    ax.set_title(f"Regime map ({'OR' if topology == 'serial_or' else 'AND'})")
    ax.set_xticks(positions, keys, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Week 7 figures from point summaries")
    parser.add_argument("--in", dest="input_dir", required=True, help="Directory with point_*.json files")
    parser.add_argument("--out", dest="output_dir", required=True, help="Directory for figure PNGs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_output(output_dir)

    points = load_points(input_dir)
    if not points:
        raise SystemExit("No point summaries found")

    plot_roc_grid(points, "serial_or", output_dir / "roc_grid_or.png")
    plot_roc_grid(points, "parallel_and", output_dir / "roc_grid_and.png")
    plot_j_bands(points, "serial_or", output_dir / "j_bands_or.png")
    plot_j_bands(points, "parallel_and", output_dir / "j_bands_and.png")
    plot_ci_diagnostics(points, output_dir / "ci_diagnostics.png")
    plot_regime_map(points, "serial_or", output_dir / "regime_map_or.png")
    plot_regime_map(points, "parallel_and", output_dir / "regime_map_and.png")

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
