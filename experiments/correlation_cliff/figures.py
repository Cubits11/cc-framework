# experiments/correlation_cliff/figures.py
from __future__ import annotations

"""
Correlation Cliff — Figures Module
==================================

This module turns *theory/population curves* and *finite-sample simulation outputs*
into publication-grade figures.

Design principles
-----------------
1) Paper-first: every plot is suitable for a PDF export (vector), with consistent axes,
   labeled units, and optional overlays (population vs empirical, confidence bands, etc.).
2) Robustness: this module is tolerant to missing columns. If BCa bootstrap CIs are not
   available, it falls back to replicate-quantile CIs from simulation summaries.
3) Separation of concerns:
   - run_all.py decides what to compute and where to save.
   - figures.py only draws from provided DataFrames.

Expected inputs
---------------
- df_pop (population / analytic curve): one row per lambda, includes:
    lambda, CC_pop, JC_pop, phi_pop_avg, tau_pop_avg
  (exact closed form for FH-linear; “population” for other paths)
- df_sum (simulation summary): one row per lambda, includes:
    lambda, CC_hat_mean, JC_hat_mean, and quantile columns like:
      CC_hat_q0025, CC_hat_q0975, JC_hat_q0025, JC_hat_q0975
  Optionally BCa columns from analyze_bootstrap (if you have it):
      CC_bca_lo, CC_bca_hi (and similarly JC_bca_lo/hi)
- thresholds: dict with keys like:
    lambda_star_emp, lambda_star_pop, phi_star_emp, phi_star_pop, ...

Outputs
-------
- cc_vs_dependence.(pdf|png)
- jc_fh_envelope.(pdf|png)
- theory_vs_empirical.(pdf|png)
- dependence_mapping.(pdf|png)  (optional but strongly recommended)

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------
# Small utilities
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _maybe_col(df: pd.DataFrame, name: str) -> bool:
    return (df is not None) and (name in df.columns)


def _pick_ci_cols(
    df: pd.DataFrame,
    *,
    base: str,
    prefer_bca: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Choose CI columns for a metric (e.g., base="CC" or base="JC").

    Priority:
      1) BCa: f"{base}_bca_lo", f"{base}_bca_hi"
      2) Simulation quantiles: f"{base}_hat_q0025", f"{base}_hat_q0975"

    Returns (lo_col, hi_col), each possibly None.
    """
    if prefer_bca:
        lo = f"{base}_bca_lo"
        hi = f"{base}_bca_hi"
        if _maybe_col(df, lo) and _maybe_col(df, hi):
            return lo, hi

    loq = f"{base}_hat_q0025"
    hiq = f"{base}_hat_q0975"
    if _maybe_col(df, loq) and _maybe_col(df, hiq):
        return loq, hiq

    return None, None


def _savefig(fig: plt.Figure, out_path: Path, dpi: int = 250) -> None:
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _interp_root(x: np.ndarray, y: np.ndarray, target: float) -> Optional[float]:
    """
    Linear interpolation root finder on a grid (x must be increasing).
    Returns first crossing root if bracket exists; else None.
    """
    if len(x) < 2:
        return None
    for i in range(len(x) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if y0 == target:
            return float(x[i])
        if (y0 - target) * (y1 - target) <= 0:
            if y1 == y0:
                return float(x[i])
            t = (target - y0) / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


# ----------------------------
# Plot configuration
# ----------------------------
@dataclass(frozen=True)
class FigureStyle:
    title_prefix: str = ""
    show_grid: bool = True
    neutrality_eta: float = 0.05
    ci_alpha: float = 0.20  # fill transparency for CI band
    envelope_alpha: float = 0.12


# ----------------------------
# Core plots
# ----------------------------
def plot_cc_vs_dependence(
    df_pop: Optional[pd.DataFrame],
    df_sum: pd.DataFrame,
    *,
    out_path_pdf: Path,
    out_path_png: Optional[Path] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    dependence_x: str = "lambda",
    style: FigureStyle = FigureStyle(),
) -> None:
    """
    CC vs dependence (lambda by default), with:
      - empirical mean CC_hat_mean
      - CI band (BCa if present else quantile band)
      - population curve overlay (CC_pop if present)
      - neutrality band around 1 (1±eta)
      - vertical line at estimated threshold(s) if provided
    """
    if "lambda" not in df_sum.columns:
        raise ValueError("df_sum must include 'lambda' column")

    x = (
        df_sum[dependence_x].to_numpy(dtype=float)
        if dependence_x in df_sum.columns
        else df_sum["lambda"].to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    title = f"{style.title_prefix}CC vs dependence".strip()
    ax.set_title(title)

    # Neutrality band around 1
    eta = float(style.neutrality_eta)
    ax.axhline(1.0, linewidth=1.0)
    ax.axhspan(1.0 - eta, 1.0 + eta, alpha=0.08)

    # Empirical mean
    if "CC_hat_mean" not in df_sum.columns:
        raise ValueError("df_sum must include 'CC_hat_mean' column (from summarize_simulation)")
    ax.plot(x, df_sum["CC_hat_mean"].to_numpy(dtype=float), label="Empirical mean")

    # CI band
    lo_col, hi_col = _pick_ci_cols(df_sum, base="CC", prefer_bca=True)
    if lo_col and hi_col:
        lo = df_sum[lo_col].to_numpy(dtype=float)
        hi = df_sum[hi_col].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, alpha=float(style.ci_alpha), label="Empirical CI")

    # Population overlay
    if df_pop is not None and ("CC_pop" in df_pop.columns):
        xp = (
            df_pop[dependence_x].to_numpy(dtype=float)
            if dependence_x in df_pop.columns
            else df_pop["lambda"].to_numpy(dtype=float)
        )
        ax.plot(xp, df_pop["CC_pop"].to_numpy(dtype=float), linestyle="--", label="Population")

    # Threshold markers
    if thresholds:
        # Prefer dependence-space threshold if available, else lambda thresholds
        # e.g., for dependence_x="phi", provide "phi_star_emp"
        dep_key_emp = f"{dependence_x}_star_emp"
        dep_key_pop = f"{dependence_x}_star_pop"
        lam_emp = thresholds.get("lambda_star_emp", None)
        lam_pop = thresholds.get("lambda_star_pop", None)

        if (
            dependence_x != "lambda"
            and dep_key_emp in thresholds
            and thresholds[dep_key_emp] is not None
        ):
            ax.axvline(
                float(thresholds[dep_key_emp]),
                linestyle=":",
                linewidth=1.2,
                label="Empirical threshold",
            )
        elif dependence_x == "lambda" and lam_emp is not None:
            ax.axvline(float(lam_emp), linestyle=":", linewidth=1.2, label="Empirical threshold")

        if (
            dependence_x != "lambda"
            and dep_key_pop in thresholds
            and thresholds[dep_key_pop] is not None
        ):
            ax.axvline(
                float(thresholds[dep_key_pop]),
                linestyle="--",
                linewidth=1.2,
                label="Population threshold",
            )
        elif dependence_x == "lambda" and lam_pop is not None:
            ax.axvline(float(lam_pop), linestyle="--", linewidth=1.2, label="Population threshold")

    ax.set_xlabel(dependence_x)
    ax.set_ylabel("CC")
    if style.show_grid:
        ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="best")

    _savefig(fig, out_path_pdf)
    if out_path_png is not None:
        _savefig(fig, out_path_png)


def plot_jc_fh_envelope(
    df_pop: Optional[pd.DataFrame],
    df_sum: pd.DataFrame,
    *,
    out_path_pdf: Path,
    out_path_png: Optional[Path] = None,
    dependence_x: str = "lambda",
    style: FigureStyle = FigureStyle(),
) -> None:
    """
    J_C vs dependence with FH envelope and population overlay.

    Requires df_sum to contain:
      - JC_hat_mean
      - JC_env_min, JC_env_max

    If df_pop has JC_pop, overlays it.
    """
    if not {"JC_hat_mean", "JC_env_min", "JC_env_max"}.issubset(df_sum.columns):
        raise ValueError("df_sum must contain JC_hat_mean, JC_env_min, JC_env_max")

    x = (
        df_sum[dependence_x].to_numpy(dtype=float)
        if dependence_x in df_sum.columns
        else df_sum["lambda"].to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    title = f"{style.title_prefix}J_C with FH envelope".strip()
    ax.set_title(title)

    env_lo = df_sum["JC_env_min"].to_numpy(dtype=float)
    env_hi = df_sum["JC_env_max"].to_numpy(dtype=float)
    ax.fill_between(x, env_lo, env_hi, alpha=float(style.envelope_alpha), label="FH envelope")

    # Empirical mean
    ax.plot(x, df_sum["JC_hat_mean"].to_numpy(dtype=float), label="Empirical mean")

    # CI band for JC
    lo_col, hi_col = _pick_ci_cols(df_sum, base="JC", prefer_bca=True)
    if lo_col and hi_col:
        lo = df_sum[lo_col].to_numpy(dtype=float)
        hi = df_sum[hi_col].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, alpha=float(style.ci_alpha), label="Empirical CI")

    # Population overlay
    if df_pop is not None and ("JC_pop" in df_pop.columns):
        xp = (
            df_pop[dependence_x].to_numpy(dtype=float)
            if dependence_x in df_pop.columns
            else df_pop["lambda"].to_numpy(dtype=float)
        )
        ax.plot(xp, df_pop["JC_pop"].to_numpy(dtype=float), linestyle="--", label="Population")

    ax.set_xlabel(dependence_x)
    ax.set_ylabel("J_C")
    if style.show_grid:
        ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="best")

    _savefig(fig, out_path_pdf)
    if out_path_png is not None:
        _savefig(fig, out_path_png)


def plot_theory_vs_empirical(
    df_pop: Optional[pd.DataFrame],
    df_sum: pd.DataFrame,
    *,
    out_path_pdf: Path,
    out_path_png: Optional[Path] = None,
    dependence_x: str = "lambda",
    style: FigureStyle = FigureStyle(),
) -> None:
    """
    Error plot: empirical mean minus population curve for CC (and optionally JC).

    If df_pop is missing, it still plots empirical CC with a baseline at 0 error disabled.
    """
    if "CC_hat_mean" not in df_sum.columns:
        raise ValueError("df_sum must contain CC_hat_mean")

    x = (
        df_sum[dependence_x].to_numpy(dtype=float)
        if dependence_x in df_sum.columns
        else df_sum["lambda"].to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    title = f"{style.title_prefix}Empirical minus population (CC)".strip()
    ax.set_title(title)

    if df_pop is None or "CC_pop" not in df_pop.columns:
        # fallback: just show CC_hat_mean; no error
        ax.plot(x, df_sum["CC_hat_mean"].to_numpy(dtype=float), label="Empirical mean CC")
        ax.axhline(1.0, linewidth=1.0, label="CC=1 baseline")
        ax.set_ylabel("CC")
    else:
        # align by lambda (strict inner join)
        dfx = df_sum[["lambda", "CC_hat_mean"]].copy()
        dfp = df_pop[["lambda", "CC_pop"]].copy()
        d = dfx.merge(dfp, on="lambda", how="inner").sort_values("lambda")
        xe = d["lambda"].to_numpy(dtype=float) if dependence_x == "lambda" else x[: len(d)]
        err = (d["CC_hat_mean"] - d["CC_pop"]).to_numpy(dtype=float)
        ax.axhline(0.0, linewidth=1.0)
        ax.plot(xe, err, label="CC_hat_mean - CC_pop")

        # Optional error band using empirical CI if present
        lo_col, hi_col = _pick_ci_cols(df_sum, base="CC", prefer_bca=True)
        if lo_col and hi_col and dependence_x == "lambda":
            # approximate error band by subtracting population curve from CI endpoints
            # (conservative if population curve is treated fixed)
            dfci = (
                df_sum[["lambda", lo_col, hi_col]]
                .merge(dfp, on="lambda", how="inner")
                .sort_values("lambda")
            )
            lo_err = (dfci[lo_col] - dfci["CC_pop"]).to_numpy(dtype=float)
            hi_err = (dfci[hi_col] - dfci["CC_pop"]).to_numpy(dtype=float)
            ax.fill_between(
                dfci["lambda"].to_numpy(dtype=float),
                lo_err,
                hi_err,
                alpha=float(style.ci_alpha),
                label="CI band (shifted)",
            )

        ax.set_ylabel("Error in CC")

    ax.set_xlabel(dependence_x)
    if style.show_grid:
        ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="best")

    _savefig(fig, out_path_pdf)
    if out_path_png is not None:
        _savefig(fig, out_path_png)


def plot_dependence_mapping(
    df_pop: Optional[pd.DataFrame],
    df_sum: pd.DataFrame,
    *,
    out_path_pdf: Path,
    out_path_png: Optional[Path] = None,
    style: FigureStyle = FigureStyle(),
) -> None:
    """
    Dependence mapping plots:
      - phi_avg and tau_avg vs lambda for population and empirical
    This is crucial for translating lambda* into interpretable dependence.
    """
    if "lambda" not in df_sum.columns:
        raise ValueError("df_sum must include 'lambda' column")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    title = f"{style.title_prefix}Dependence mapping (phi, tau)".strip()
    ax.set_title(title)

    x = df_sum["lambda"].to_numpy(dtype=float)

    # Empirical dependence averages (means)
    if "phi_hat_avg_mean" in df_sum.columns:
        ax.plot(
            x, df_sum["phi_hat_avg_mean"].to_numpy(dtype=float), label="Empirical phi_avg (mean)"
        )
    if "tau_hat_avg_mean" in df_sum.columns:
        ax.plot(
            x, df_sum["tau_hat_avg_mean"].to_numpy(dtype=float), label="Empirical tau_avg (mean)"
        )

    # Population
    if df_pop is not None:
        if "phi_pop_avg" in df_pop.columns:
            ax.plot(
                df_pop["lambda"].to_numpy(dtype=float),
                df_pop["phi_pop_avg"].to_numpy(dtype=float),
                linestyle="--",
                label="Population phi_avg",
            )
        if "tau_pop_avg" in df_pop.columns:
            ax.plot(
                df_pop["lambda"].to_numpy(dtype=float),
                df_pop["tau_pop_avg"].to_numpy(dtype=float),
                linestyle="--",
                label="Population tau_avg",
            )

    ax.set_xlabel("lambda")
    ax.set_ylabel("Dependence summary")
    if style.show_grid:
        ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="best")

    _savefig(fig, out_path_pdf)
    if out_path_png is not None:
        _savefig(fig, out_path_png)


# ----------------------------
# Master entry point
# ----------------------------
def make_all_figures(
    *,
    out_dir: Path,
    df_pop: Optional[pd.DataFrame],
    df_sum: pd.DataFrame,
    thresholds: Optional[Dict[str, Any]] = None,
    dependence_x: str = "lambda",
    style: FigureStyle = FigureStyle(),
    also_png: bool = True,
) -> Dict[str, str]:
    """
    Generate the full figure set into out_dir/figures.

    Returns a dict mapping figure-name to saved PDF path.
    """
    out_dir = _as_path(out_dir)
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)

    saved: Dict[str, str] = {}

    def p(name: str) -> Path:
        return fig_dir / name

    # 1) CC plot
    pdf = p("cc_vs_dependence.pdf")
    png = p("cc_vs_dependence.png") if also_png else None
    plot_cc_vs_dependence(
        df_pop,
        df_sum,
        out_path_pdf=pdf,
        out_path_png=png,
        thresholds=thresholds,
        dependence_x=dependence_x,
        style=style,
    )
    saved["cc_vs_dependence"] = str(pdf)

    # 2) JC with envelope
    pdf = p("jc_fh_envelope.pdf")
    png = p("jc_fh_envelope.png") if also_png else None
    plot_jc_fh_envelope(
        df_pop, df_sum, out_path_pdf=pdf, out_path_png=png, dependence_x=dependence_x, style=style
    )
    saved["jc_fh_envelope"] = str(pdf)

    # 3) Error / comparison
    pdf = p("theory_vs_empirical.pdf")
    png = p("theory_vs_empirical.png") if also_png else None
    plot_theory_vs_empirical(
        df_pop, df_sum, out_path_pdf=pdf, out_path_png=png, dependence_x=dependence_x, style=style
    )
    saved["theory_vs_empirical"] = str(pdf)

    # 4) Dependence mapping
    pdf = p("dependence_mapping.pdf")
    png = p("dependence_mapping.png") if also_png else None
    plot_dependence_mapping(df_pop, df_sum, out_path_pdf=pdf, out_path_png=png, style=style)
    saved["dependence_mapping"] = str(pdf)

    return saved
