#!/usr/bin/env python3
"""Benchmark and validate n-way Frechet-class bounds.

Outputs:
  - frechet_tightness_convergence.csv
  - frechet_tightness_convergence.png
  - frechet_width_grid.csv
  - frechet_width_grid.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cc.kernel.frechet_classes import (  # noqa: E402
    PairwiseDependence,
    distribution_moments,
    frechet_bounds,
    joint_probability_to_dependence,
    sample_binary_vectors,
)

EventName = Literal["and", "or"]


def _event_rate(samples: np.ndarray, event: EventName) -> float:
    if event == "and":
        return float(np.mean(np.all(samples == 1, axis=1)))
    return float(np.mean(np.any(samples == 1, axis=1)))


def _constraints_from_distribution(
    distribution: np.ndarray,
    n_events: int,
) -> tuple[np.ndarray, list[PairwiseDependence]]:
    marginals, pairwise_joint = distribution_moments(distribution, n_events)
    constraints: list[PairwiseDependence] = []
    pairs = list(combinations(range(n_events), 2))
    for k, (i, j) in enumerate(pairs[: max(1, min(n_events, len(pairs)))]):
        kind = "spearman_rho" if k % 2 == 0 else "kendall_tau"
        value = joint_probability_to_dependence(
            float(marginals[i]),
            float(marginals[j]),
            float(pairwise_joint[i, j]),
            kind=kind,
        )
        constraints.append(PairwiseDependence(i=i, j=j, kind=kind, value=value))
    return marginals, constraints


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_tightness_convergence(
    output_dir: Path,
    *,
    seed: int,
    replicates: int,
    sample_sizes: list[int],
) -> Path:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for n_events in (2, 3, 4, 5):
        base_distribution = rng.dirichlet(np.full(1 << n_events, 1.2))
        marginals, constraints = _constraints_from_distribution(base_distribution, n_events)
        for event in ("and", "or"):
            result = frechet_bounds(
                marginals,
                pairwise=constraints,
                event=event,
                return_distributions=True,
            )
            extremals = [
                ("lower", result.lower, result.lower_distribution),
                ("upper", result.upper, result.upper_distribution),
            ]
            for endpoint, analytic, distribution in extremals:
                if distribution is None:
                    raise RuntimeError("expected LP extremal distributions")
                for size in sample_sizes:
                    estimates = []
                    for _ in range(replicates):
                        samples = sample_binary_vectors(
                            distribution,
                            size,
                            rng=rng,
                            n_events=n_events,
                        )
                        estimates.append(_event_rate(samples, event))
                    estimate_array = np.asarray(estimates, dtype=float)
                    rows.append(
                        {
                            "n_events": n_events,
                            "event": event,
                            "endpoint": endpoint,
                            "sample_size": size,
                            "analytic": analytic,
                            "estimate_mean": float(np.mean(estimate_array)),
                            "estimate_sd": float(np.std(estimate_array, ddof=0)),
                            "abs_error": float(abs(np.mean(estimate_array) - analytic)),
                        }
                    )

    csv_path = output_dir / "frechet_tightness_convergence.csv"
    _write_csv(csv_path, rows)
    _plot_tightness(output_dir / "frechet_tightness_convergence.png", rows)
    return csv_path


def run_width_grid(output_dir: Path, *, grid_size: int) -> Path:
    rows: list[dict[str, object]] = []
    grid = np.linspace(0.05, 0.95, grid_size)
    fixed_p3 = 0.55
    for p1 in grid:
        for p2 in grid:
            marginals = np.asarray([p1, p2, fixed_p3], dtype=float)
            constraints = [
                PairwiseDependence(0, 1, "spearman_rho", 0.0),
                PairwiseDependence(0, 2, "kendall_tau", 0.0),
                PairwiseDependence(1, 2, "spearman_rho", 0.0),
            ]
            for event in ("and", "or"):
                classical = frechet_bounds(marginals, event=event)
                improved = frechet_bounds(marginals, pairwise=constraints, event=event)
                rows.append(
                    {
                        "p1": float(p1),
                        "p2": float(p2),
                        "p3": fixed_p3,
                        "event": event,
                        "classical_width": classical.width,
                        "improved_width": improved.width,
                        "width_reduction": classical.width - improved.width,
                    }
                )

    csv_path = output_dir / "frechet_width_grid.csv"
    _write_csv(csv_path, rows)
    _plot_width_grid(output_dir / "frechet_width_grid.png", rows, grid_size)
    return csv_path


def _plot_tightness(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for row_idx, n_events in enumerate((2, 3, 4, 5)):
        for col_idx, event in enumerate(("and", "or")):
            ax = axes[row_idx, col_idx]
            for endpoint, color in (("lower", "#2f6f9f"), ("upper", "#b24a3b")):
                selected = [
                    r
                    for r in rows
                    if r["n_events"] == n_events
                    and r["event"] == event
                    and r["endpoint"] == endpoint
                ]
                selected.sort(key=lambda r: int(r["sample_size"]))
                x = np.asarray([r["sample_size"] for r in selected], dtype=float)
                y = np.asarray([r["estimate_mean"] for r in selected], dtype=float)
                sd = np.asarray([r["estimate_sd"] for r in selected], dtype=float)
                analytic = float(selected[0]["analytic"])
                ax.plot(x, y, marker="o", color=color, label=f"{endpoint} MC")
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12, linewidth=0)
                ax.axhline(analytic, color=color, linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"n={n_events}, {event.upper()}")
            if col_idx == 0:
                ax.set_ylabel("event probability")
            if row_idx == 3:
                ax.set_xlabel("sample size")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_width_grid(path: Path, rows: list[dict[str, object]], grid_size: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row_idx, event in enumerate(("and", "or")):
        selected = [r for r in rows if r["event"] == event]
        classical = np.asarray([r["classical_width"] for r in selected], dtype=float).reshape(
            grid_size, grid_size
        )
        improved = np.asarray([r["improved_width"] for r in selected], dtype=float).reshape(
            grid_size, grid_size
        )
        reduction = np.asarray([r["width_reduction"] for r in selected], dtype=float).reshape(
            grid_size, grid_size
        )
        for col_idx, (title, values) in enumerate(
            (
                ("classical width", classical),
                ("improved width", improved),
                ("width reduction", reduction),
            )
        ):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                values,
                origin="lower",
                extent=(0.05, 0.95, 0.05, 0.95),
                aspect="auto",
                cmap="viridis",
            )
            ax.set_title(f"{event.upper()} {title}")
            ax.set_xlabel("p1")
            if col_idx == 0:
                ax.set_ylabel("p2")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "theory" / "figures",
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[100, 300, 1000, 3000, 10000],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tightness_csv = run_tightness_convergence(
        args.output_dir,
        seed=args.seed,
        replicates=args.replicates,
        sample_sizes=args.sample_sizes,
    )
    width_csv = run_width_grid(args.output_dir, grid_size=args.grid_size)
    print(f"Wrote {tightness_csv}")
    print(f"Wrote {tightness_csv.with_suffix('.png')}")
    print(f"Wrote {width_csv}")
    print(f"Wrote {width_csv.with_suffix('.png')}")


if __name__ == "__main__":
    main()
