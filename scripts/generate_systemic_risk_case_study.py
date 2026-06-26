#!/usr/bin/env python3
"""Generate the systemic-risk stress-test case study artifacts.

Default outputs:

  docs/theory/figures/systemic_risk_case_study.csv
  docs/theory/figures/systemic_risk_case_study.md
  docs/theory/figures/systemic_risk_case_study.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from cc.kernel.ccf_models import beta_factor
from cc.kernel.stress import BaselineDependence, StressBudget, stress_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260626, help="Shuffle seed.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/theory/figures/systemic_risk_case_study.csv"),
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("docs/theory/figures/systemic_risk_case_study.md"),
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=Path("docs/theory/figures/systemic_risk_case_study.png"),
    )
    parser.add_argument("--no-figure", action="store_true", help="Only write CSV and Markdown.")
    args = parser.parse_args()

    samples = _synthetic_two_guardrail_dataset(seed=args.seed)
    baseline = BaselineDependence(samples=samples, name="synthetic_two_guardrail")
    budgets = [0.02, 0.06, 0.10]
    rows = _comparison_rows(baseline, budgets=budgets)

    _write_csv(args.out_csv, rows)
    _write_markdown_report(args.out_report, rows, sample_count=int(samples.shape[0]))
    if not args.no_figure:
        _write_figure(args.out_figure, rows)

    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_report}")
    if not args.no_figure:
        print(f"wrote {args.out_figure}")


def _synthetic_two_guardrail_dataset(*, seed: int) -> np.ndarray:
    """Return exact-count binary failure indicators for two guardrails."""

    rows = (
        [[0, 0]] * 6800
        + [[1, 0]] * 1200
        + [[0, 1]] * 1200
        + [[1, 1]] * 800
    )
    samples = np.asarray(rows, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples, axis=0)
    return samples


def _comparison_rows(
    baseline: BaselineDependence,
    *,
    budgets: list[float],
) -> list[dict[str, Any]]:
    zero = stress_test(baseline, StressBudget(0.0, metric="wasserstein"))
    fh_limit = stress_test(baseline, StressBudget(float("inf"), metric="wasserstein"))
    baseline_rate = float(np.mean(zero.marginals))
    beta_hat = _fit_beta_factor(
        failure_rate=baseline_rate,
        observed_joint=zero.baseline_risk,
    )
    ccf_estimate = beta_factor([baseline_rate, baseline_rate], beta_hat)

    rows: list[dict[str, Any]] = [
        _row(
            method="Empirical copula baseline",
            metric="observed",
            stress_budget=0.0,
            risk=zero.baseline_risk,
            baseline_risk=zero.baseline_risk,
            fh_upper=zero.frechet_upper,
            realized_distance=0.0,
            note="Observed synthetic joint failure rate.",
        ),
        _row(
            method="CCF beta-factor point estimate",
            metric="module4_beta_factor",
            stress_budget=None,
            risk=ccf_estimate,
            baseline_risk=zero.baseline_risk,
            fh_upper=zero.frechet_upper,
            realized_distance=None,
            note=f"beta_hat={beta_hat:.6f}, fitted to the same empirical co-failure rate.",
        ),
    ]

    for amount in budgets:
        result = stress_test(baseline, StressBudget(amount, metric="wasserstein"))
        rows.append(
            _row(
                method=f"Budget-constrained stress eps={amount:.2f}",
                metric="wasserstein",
                stress_budget=amount,
                risk=result.stressed_risk,
                baseline_risk=result.baseline_risk,
                fh_upper=result.frechet_upper,
                realized_distance=result.realized_distance,
                note="Worst fixed-marginal dependence shift inside normalized-Hamming W1 ball.",
            )
        )

    rows.append(
        _row(
            method="Unconstrained FH upper limit",
            metric="frechet_hoeffding",
            stress_budget=float("inf"),
            risk=fh_limit.stressed_risk,
            baseline_risk=fh_limit.baseline_risk,
            fh_upper=fh_limit.frechet_upper,
            realized_distance=fh_limit.realized_distance,
            note="Budget-to-infinity comparison point, not a finite local stress estimate.",
        )
    )
    return rows


def _row(
    *,
    method: str,
    metric: str,
    stress_budget: float | None,
    risk: float,
    baseline_risk: float,
    fh_upper: float,
    realized_distance: float | None,
    note: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "metric": metric,
        "stress_budget": stress_budget,
        "realized_distance": realized_distance,
        "composed_failure_risk": risk,
        "effective_protection": 1.0 - risk,
        "risk_increase_vs_baseline": risk - baseline_risk,
        "gap_to_fh_upper": max(0.0, fh_upper - risk),
        "note": note,
    }


def _fit_beta_factor(*, failure_rate: float, observed_joint: float) -> float:
    """Fit beta in the existing two-component beta-factor formula."""

    q = float(failure_rate)
    target = float(observed_joint)
    if target <= q * q:
        return 0.0

    coeffs = [q * q, q - 2.0 * q * q, q * q - target]
    roots = np.roots(coeffs)
    candidates = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1.0e-10 and 0.0 <= float(root.real) <= 1.0
    ]
    if not candidates:
        return 1.0
    return min(
        candidates,
        key=lambda beta: abs(beta_factor([q, q], beta) - target),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "metric",
        "stress_budget",
        "realized_distance",
        "composed_failure_risk",
        "effective_protection",
        "risk_increase_vs_baseline",
        "gap_to_fh_upper",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, rows: list[dict[str, Any]], *, sample_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _markdown_table(rows)
    text = (
        "# Systemic Risk Stress-Test Case Study\n\n"
        f"Synthetic dataset: {sample_count} two-guardrail demands with exact empirical "
        "marginals `P(F_1)=P(F_2)=0.20` and co-failure `P(F_1 and F_2)=0.08`.\n\n"
        "Finite stress rows solve the fixed-marginal Wasserstein stress problem. "
        "The FH row is the budget-to-infinity limit and is included only as a "
        "comparison endpoint.\n\n"
        f"{table}\n"
    )
    path.write_text(text, encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Method",
        "Budget",
        "Risk",
        "Protection",
        "Increase",
        "Gap to FH",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        budget = row["stress_budget"]
        if budget is None:
            budget_text = "-"
        elif budget == float("inf"):
            budget_text = "infinity"
        else:
            budget_text = f"{float(budget):.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    budget_text,
                    f"{float(row['composed_failure_risk']):.4f}",
                    f"{float(row['effective_protection']):.4f}",
                    f"{float(row['risk_increase_vs_baseline']):.4f}",
                    f"{float(row['gap_to_fh_upper']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["method"]) for row in rows]
    risks = [float(row["composed_failure_risk"]) for row in rows]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2"]

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, risks, color=colors[: len(labels)])
    ax.set_ylabel("Composed failure risk")
    ax.set_title("Finite Dependence Stress vs. CCF Point Estimate and FH Limit")
    ax.set_ylim(0.0, max(risks) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)

    for bar, risk in zip(bars, risks, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.004,
            f"{risk:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
