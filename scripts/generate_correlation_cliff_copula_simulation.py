#!/usr/bin/env python3
"""Generate a controlled copula-tail correlation-cliff simulation figure.

This script intentionally writes artifacts from code, not from a notebook.
Default outputs:

  docs/theory/figures/correlation_cliff_copula_sweep.csv
  docs/theory/figures/correlation_cliff_copula_sweep.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from cc.kernel.cliff import sample_copula, theoretical_tail_dependence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000, help="Samples per grid point.")
    parser.add_argument("--points", type=int, default=41, help="Grid points per family.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Rare miss marginal.")
    parser.add_argument("--seed", type=int, default=20260914, help="Base RNG seed.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/theory/figures/correlation_cliff_copula_sweep.csv"),
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=Path("docs/theory/figures/correlation_cliff_copula_sweep.png"),
    )
    parser.add_argument("--no-figure", action="store_true", help="Only write the CSV.")
    args = parser.parse_args()

    if args.n < 100:
        raise ValueError("--n must be at least 100.")
    if args.points < 5:
        raise ValueError("--points must be at least 5.")
    if not 0.0 < args.epsilon < 0.2:
        raise ValueError("--epsilon must be in (0, 0.2).")

    rows = _simulate_rows(
        n=args.n,
        points=args.points,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    rows = _with_derivatives(rows)
    _write_csv(args.out_csv, rows)

    if not args.no_figure:
        _write_figure(args.out_figure, rows)

    print(f"wrote {args.out_csv}")
    if not args.no_figure:
        print(f"wrote {args.out_figure}")


def _simulate_rows(*, n: int, points: int, epsilon: float, seed: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "family": "gaussian",
            "parameter_name": "rho",
            "grid": np.linspace(0.0, 0.98, points),
            "tail_side": "upper",
            "params": lambda x: {"rho": float(x)},
        },
        {
            "family": "clayton",
            "parameter_name": "theta",
            "grid": np.linspace(0.0, 8.0, points),
            "tail_side": "lower",
            "params": lambda x: {"theta": float(x)},
        },
        {
            "family": "gumbel",
            "parameter_name": "theta",
            "grid": np.linspace(1.0, 8.0, points),
            "tail_side": "upper",
            "params": lambda x: {"theta": float(x)},
        },
        {
            "family": "student_t",
            "parameter_name": "rho",
            "grid": np.linspace(0.0, 0.95, points),
            "tail_side": "upper",
            "params": lambda x: {"rho": float(x), "df": 4.0},
        },
    ]

    rows: list[dict[str, Any]] = []
    for family_idx, spec in enumerate(specs):
        family = str(spec["family"])
        param_name = str(spec["parameter_name"])
        tail_side = str(spec["tail_side"])
        grid = np.asarray(spec["grid"], dtype=np.float64)
        param_builder = spec["params"]

        for param_idx, value in enumerate(grid):
            child = np.random.SeedSequence([seed, family_idx, param_idx])
            rng = np.random.default_rng(child)
            params = param_builder(float(value))
            samples = sample_copula(family, params, n, random_state=rng)  # type: ignore[arg-type]
            if tail_side == "lower":
                joint = (samples[:, 0] <= epsilon) & (samples[:, 1] <= epsilon)
            else:
                joint = (samples[:, 0] >= 1.0 - epsilon) & (samples[:, 1] >= 1.0 - epsilon)

            tail = theoretical_tail_dependence(family, params)  # type: ignore[arg-type]
            lambda_relevant = tail.lambda_lower if tail_side == "lower" else tail.lambda_upper
            miss_probability = float(np.mean(joint))
            rows.append(
                {
                    "family": family,
                    "parameter_name": param_name,
                    "parameter_value": float(value),
                    "tail_side": tail_side,
                    "epsilon": float(epsilon),
                    "n": int(n),
                    "seed": int(seed),
                    "miss_probability": miss_probability,
                    "amplification": miss_probability / epsilon,
                    "lambda_lower": float(tail.lambda_lower),
                    "lambda_upper": float(tail.lambda_upper),
                    "lambda_relevant": float(lambda_relevant),
                }
            )
    return rows


def _with_derivatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    families = sorted({str(row["family"]) for row in out})
    for family in families:
        idx = [i for i, row in enumerate(out) if row["family"] == family]
        idx.sort(key=lambda i: float(out[i]["parameter_value"]))
        x = np.array([float(out[i]["parameter_value"]) for i in idx], dtype=np.float64)
        y = np.array([float(out[i]["miss_probability"]) for i in idx], dtype=np.float64)
        amp = np.array([float(out[i]["amplification"]) for i in idx], dtype=np.float64)
        dy = np.gradient(y, x)
        damp = np.gradient(amp, x)
        ddy = np.gradient(dy, x)
        max_derivative_index = int(np.argmax(np.abs(ddy)))
        cliff_parameter = float(x[max_derivative_index])

        for local, row_idx in enumerate(idx):
            out[row_idx]["d_miss_d_parameter"] = float(dy[local])
            out[row_idx]["d_amplification_d_parameter"] = float(damp[local])
            out[row_idx]["d2_miss_d_parameter2"] = float(ddy[local])
            out[row_idx]["max_derivative_change_parameter"] = cliff_parameter
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "parameter_name",
        "parameter_value",
        "tail_side",
        "epsilon",
        "n",
        "seed",
        "miss_probability",
        "amplification",
        "lambda_lower",
        "lambda_upper",
        "lambda_relevant",
        "d_miss_d_parameter",
        "d_amplification_d_parameter",
        "d2_miss_d_parameter2",
        "max_derivative_change_parameter",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    families = ["gaussian", "clayton", "gumbel", "student_t"]
    titles = {
        "gaussian": "Gaussian: high correlation, zero tail coefficient",
        "clayton": "Clayton: lower-tail co-failure",
        "gumbel": "Gumbel: upper-tail co-failure",
        "student_t": "Student-t: symmetric heavy-tail co-failure",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, family in zip(axes.ravel(), families, strict=True):
        family_rows = [row for row in rows if row["family"] == family]
        family_rows.sort(key=lambda row: float(row["parameter_value"]))
        x = np.array([float(row["parameter_value"]) for row in family_rows])
        amp = np.array([float(row["amplification"]) for row in family_rows])
        derivative = np.array([float(row["d_amplification_d_parameter"]) for row in family_rows])
        cliff_x = float(family_rows[0]["max_derivative_change_parameter"])
        parameter_name = str(family_rows[0]["parameter_name"])

        ax.plot(x, amp, color="#1f77b4", linewidth=2.0, label="P(joint miss) / epsilon")
        ax.set_title(titles[family])
        ax.set_xlabel(parameter_name)
        ax.set_ylabel("co-failure amplification")
        ax.axvline(
            cliff_x,
            color="#444444",
            linestyle=":",
            linewidth=1.5,
            label="max derivative change",
        )
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(x, derivative, color="#d62728", linewidth=1.5, alpha=0.85, label="derivative")
        ax2.set_ylabel("finite-difference derivative")

        lines = [
            line
            for line in ax.get_lines() + ax2.get_lines()
            if not line.get_label().startswith("_")
        ]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=8)

    fig.suptitle("Copula Tail-Dependence Cliffs in Composed Guardrail Miss Probability")
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
