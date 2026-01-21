from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory.fh_bounds import (
    ComposedJBounds,
    compute_cc_bounds,
    default_cc_regime_thresholds,
    parallel_and_composition_bounds,
    serial_or_composition_bounds,
)


def _save_plot(fig: plt.Figure, path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_fh_envelope(
    output_dir: Path,
    scenario_id: str,
    thetas: List[float],
    observed_j: List[float],
    observed_ci: List[Tuple[float, float]],
    bounds: ComposedJBounds,
    j_independence: float,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        thetas,
        [bounds.j_lower] * len(thetas),
        [bounds.j_upper] * len(thetas),
        color="tab:blue",
        alpha=0.2,
        label="FH envelope",
    )
    ax.plot(
        thetas, [j_independence] * len(thetas), color="black", linestyle="--", label="Independence"
    )
    ax.plot(thetas, observed_j, color="tab:orange", marker="o", label="Observed J")
    lower = [ci[0] for ci in observed_ci]
    upper = [ci[1] for ci in observed_ci]
    ax.vlines(thetas, lower, upper, color="tab:orange", alpha=0.5)
    ax.set_xlabel("Dependence parameter (theta/rho)")
    ax.set_ylabel("J statistic")
    ax.set_title(title)
    ax.legend(loc="best")
    output_path = output_dir / f"fh_envelope_{scenario_id}.png"
    _save_plot(
        fig,
        output_path,
        {
            "scenario_id": scenario_id,
            "plot": "fh_envelope",
            "j_lower": bounds.j_lower,
            "j_upper": bounds.j_upper,
            "j_independence": j_independence,
        },
    )
    plt.close(fig)
    return output_path


def plot_cc_regime_heatmap(
    output_dir: Path,
    scenario_id: str,
    miss_grid: List[float],
    fpr_grid: List[float],
    composition_type: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    cc_midpoints = []
    thresholds = default_cc_regime_thresholds()
    for miss in miss_grid:
        row = []
        for fpr in fpr_grid:
            miss_rates = [miss, miss]
            fpr_rates = [fpr, fpr]
            if composition_type == "serial_or":
                bounds = serial_or_composition_bounds(miss_rates, fpr_rates)
            else:
                bounds = parallel_and_composition_bounds(miss_rates, fpr_rates)
            max_individual = max(bounds.individual_j_stats)
            cc_lower, cc_upper = compute_cc_bounds(bounds.j_lower, bounds.j_upper, max_individual)
            row.append(0.5 * (cc_lower + cc_upper))
        cc_midpoints.append(row)

    im = ax.imshow(cc_midpoints, origin="lower", cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(fpr_grid)))
    ax.set_xticklabels([f"{v:.2f}" for v in fpr_grid])
    ax.set_yticks(range(len(miss_grid)))
    ax.set_yticklabels([f"{v:.2f}" for v in miss_grid])
    ax.set_xlabel("FPR per rail")
    ax.set_ylabel("Miss rate per rail")
    ax.set_title(f"CC midpoint heatmap ({composition_type})")
    fig.colorbar(im, ax=ax, label="CC midpoint")
    output_path = output_dir / f"cc_regime_heatmap_{scenario_id}.png"
    _save_plot(
        fig,
        output_path,
        {
            "scenario_id": scenario_id,
            "plot": "cc_regime_heatmap",
            "thresholds": {
                "constructive": thresholds.constructive,
                "destructive": thresholds.destructive,
            },
        },
    )
    plt.close(fig)
    return output_path


def plot_identifiability_map(
    output_dir: Path,
    scenario_id: str,
    miss_grid: List[float],
    fpr_grid: List[float],
    composition_type: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    widths = []
    for miss in miss_grid:
        row = []
        for fpr in fpr_grid:
            miss_rates = [miss, miss]
            fpr_rates = [fpr, fpr]
            if composition_type == "serial_or":
                bounds = serial_or_composition_bounds(miss_rates, fpr_rates)
            else:
                bounds = parallel_and_composition_bounds(miss_rates, fpr_rates)
            row.append(bounds.width)
        widths.append(row)

    im = ax.imshow(widths, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(fpr_grid)))
    ax.set_xticklabels([f"{v:.2f}" for v in fpr_grid])
    ax.set_yticks(range(len(miss_grid)))
    ax.set_yticklabels([f"{v:.2f}" for v in miss_grid])
    ax.set_xlabel("FPR per rail")
    ax.set_ylabel("Miss rate per rail")
    ax.set_title(f"FH width map ({composition_type})")
    fig.colorbar(im, ax=ax, label="J interval width")
    output_path = output_dir / f"identifiability_map_{scenario_id}.png"
    _save_plot(
        fig,
        output_path,
        {"scenario_id": scenario_id, "plot": "identifiability_map"},
    )
    plt.close(fig)
    return output_path


def plot_more_rails_scaling(
    output_dir: Path,
    scenario_id: str,
    k_values: Iterable[int],
    miss_rate: float,
    fpr_rate: float,
    composition_type: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lower_bounds = []
    upper_bounds = []
    k_values_list = list(k_values)
    for k in k_values_list:
        miss_rates = [miss_rate] * k
        fpr_rates = [fpr_rate] * k
        if composition_type == "serial_or":
            bounds = serial_or_composition_bounds(miss_rates, fpr_rates)
        else:
            bounds = parallel_and_composition_bounds(miss_rates, fpr_rates)
        lower_bounds.append(bounds.j_lower)
        upper_bounds.append(bounds.j_upper)

    ax.plot(k_values_list, lower_bounds, marker="o", label="J lower bound")
    ax.plot(k_values_list, upper_bounds, marker="o", label="J upper bound")
    ax.set_xlabel("Number of rails (k)")
    ax.set_ylabel("J bounds")
    ax.set_title(f"Scaling with rails ({composition_type})")
    ax.legend(loc="best")
    output_path = output_dir / f"rail_scaling_{scenario_id}.png"
    _save_plot(
        fig,
        output_path,
        {
            "scenario_id": scenario_id,
            "plot": "rail_scaling",
            "miss_rate": miss_rate,
            "fpr_rate": fpr_rate,
        },
    )
    plt.close(fig)
    return output_path


def plot_cii_distribution(
    output_dir: Path,
    scenario_id: str,
    thetas: List[float],
    cii_values: List[float],
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thetas, cii_values, marker="o", color="tab:purple")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Dependence parameter (theta/rho)")
    ax.set_ylabel("CII (kappa)")
    ax.set_title("CII vs dependence")
    output_path = output_dir / f"cii_distribution_{scenario_id}.png"
    _save_plot(
        fig,
        output_path,
        {"scenario_id": scenario_id, "plot": "cii_distribution"},
    )
    plt.close(fig)
    return output_path
