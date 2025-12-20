# experiments/correlation_cliff/run_all.py
from __future__ import annotations

"""
Correlation Cliff — Run-All Orchestrator
=======================================

This file is the *one-command* runner for experiments/correlation_cliff/.

What it does (end-to-end)
-------------------------
1) Load YAML config (config_s1.yaml by default).
2) Compute a *population curve* over the dependence grid:
     - If path == fh_linear, you can interpret this as "analytic theory"
       (because p11(λ) is defined via FH-linear and all metrics are closed-form).
     - For other paths (fh_power, fh_scurve, gaussian_tau), it computes the exact
       *population quantities under that path definition* (still deterministic).
3) Run Monte Carlo simulation (Multinomial sampling), producing:
     - sim_long.csv (replicate-level)
     - sim_summary.csv (per-lambda means/std/quantiles)
4) Estimate thresholds (empirical mean crossing and population crossing):
     - lambda_star_emp, lambda_star_pop
     - also maps to phi_star_* and tau_star_* if dependence mappings are available
5) Render figures into out_dir/figures:
     - cc_vs_dependence.pdf
     - jc_fh_envelope.pdf
     - theory_vs_empirical.pdf
     - dependence_mapping.pdf
6) Save a manifest JSON capturing determinism and reproducibility metadata.

Expected repo structure
-----------------------
experiments/correlation_cliff/
  config_s1.yaml
  simulate.py
  theory.py
  analyze_bootstrap.py      (optional; not required here)
  figures.py
  run_all.py
  README.md

If analyze_bootstrap.py exists and you want to integrate it later, do it in run_all.py
after simulate_grid: read sim_long, compute BCa per lambda, merge into sim_summary,
and figures.py will automatically prefer BCa columns if present.

"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import json
import os
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ----------------------------
# Local imports
# ----------------------------
try:
    from .simulate import SimConfig, _cfg_from_dict, _load_yaml, simulate_grid, summarize_simulation, p11_from_path
    from .figures import FigureStyle, make_all_figures
    from .theory import (
        TwoWorldMarginals,
        WorldMarginals,
        compute_fh_jc_envelope,
        joint_cells_from_marginals,
        pC_from_joint,
        phi_from_joint,
        kendall_tau_a_from_joint,
    )
except Exception:
    from simulate import SimConfig, _cfg_from_dict, _load_yaml, simulate_grid, summarize_simulation, p11_from_path
    from figures import FigureStyle, make_all_figures
    from theory import (
        TwoWorldMarginals,
        WorldMarginals,
        compute_fh_jc_envelope,
        joint_cells_from_marginals,
        pC_from_joint,
        phi_from_joint,
        kendall_tau_a_from_joint,
    )


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_stamp() -> str:
    # Deterministic enough for file naming; not used for RNG.
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _interp_root(x: np.ndarray, y: np.ndarray, target: float) -> Optional[float]:
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
# Population curve computation (path-aware)
# ----------------------------
def population_curve_from_path(cfg: SimConfig) -> pd.DataFrame:
    """
    Compute deterministic population quantities per lambda under cfg.path.

    This is "theory" for fh_linear, and "path-defined population" otherwise.

    Returns columns:
      lambda, CC_pop, JC_pop, dC_pop, phi_pop_avg, tau_pop_avg,
      pC0_pop, pC1_pop, p110_pop, p111_pop, ...
      plus FH envelope bounds (JC_env_min/max).
    """
    rows: list[Dict[str, Any]] = []

    # singleton leakages depend only on marginals
    JA = abs(cfg.marginals.w1.pA - cfg.marginals.w0.pA)
    JB = abs(cfg.marginals.w1.pB - cfg.marginals.w0.pB)
    Jbest = max(JA, JB)

    jmin_env, jmax_env = compute_fh_jc_envelope(cfg.marginals, cfg.rule)

    for lam in cfg.lambdas:
        lam = float(lam)
        row: Dict[str, Any] = {"lambda": lam, "rule": cfg.rule, "path": cfg.path}
        row["JA"] = float(JA)
        row["JB"] = float(JB)
        row["Jbest"] = float(Jbest)
        row["JC_env_min"] = float(jmin_env)
        row["JC_env_max"] = float(jmax_env)

        # world 0
        p11_0, meta0 = p11_from_path(
            cfg.marginals.w0.pA, cfg.marginals.w0.pB, lam,
            path=cfg.path, path_params=cfg.path_params,
        )
        c0 = joint_cells_from_marginals(cfg.marginals.w0.pA, cfg.marginals.w0.pB, p11_0)
        pC0 = pC_from_joint(cfg.rule, c0, pA=cfg.marginals.w0.pA, pB=cfg.marginals.w0.pB)
        phi0 = phi_from_joint(cfg.marginals.w0.pA, cfg.marginals.w0.pB, c0["p11"])
        tau0 = kendall_tau_a_from_joint(c0)

        # world 1
        p11_1, meta1 = p11_from_path(
            cfg.marginals.w1.pA, cfg.marginals.w1.pB, lam,
            path=cfg.path, path_params=cfg.path_params,
        )
        c1 = joint_cells_from_marginals(cfg.marginals.w1.pA, cfg.marginals.w1.pB, p11_1)
        pC1 = pC_from_joint(cfg.rule, c1, pA=cfg.marginals.w1.pA, pB=cfg.marginals.w1.pB)
        phi1 = phi_from_joint(cfg.marginals.w1.pA, cfg.marginals.w1.pB, c1["p11"])
        tau1 = kendall_tau_a_from_joint(c1)

        dC = float(pC1 - pC0)
        JC = float(abs(dC))
        CC = float(JC / Jbest) if Jbest > 0 else float("nan")

        row.update(
            {
                "pC0_pop": float(pC0),
                "pC1_pop": float(pC1),
                "dC_pop": float(dC),
                "JC_pop": float(JC),
                "CC_pop": float(CC),
                "p11_0_pop": float(c0["p11"]),
                "p11_1_pop": float(c1["p11"]),
                "phi0_pop": float(phi0),
                "phi1_pop": float(phi1),
                "phi_pop_avg": float(0.5 * (phi0 + phi1)),
                "tau0_pop": float(tau0),
                "tau1_pop": float(tau1),
                "tau_pop_avg": float(0.5 * (tau0 + tau1)),
            }
        )

        # store meta fields if present
        for k, v in meta0.items():
            row[f"meta0_{k}"] = float(v)
        for k, v in meta1.items():
            row[f"meta1_{k}"] = float(v)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)


def estimate_thresholds(
    df_pop: pd.DataFrame,
    df_sum: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Estimate crossing points where CC == 1 using linear interpolation.
    Returns both lambda-space and (if available) phi/tau-space thresholds.
    """
    out: Dict[str, Any] = {}

    # Population threshold
    if "lambda" in df_pop.columns and "CC_pop" in df_pop.columns:
        x = df_pop["lambda"].to_numpy(dtype=float)
        y = df_pop["CC_pop"].to_numpy(dtype=float)
        out["lambda_star_pop"] = _interp_root(x, y, 1.0)
    else:
        out["lambda_star_pop"] = None

    # Empirical threshold (mean)
    if "lambda" in df_sum.columns and "CC_hat_mean" in df_sum.columns:
        x = df_sum["lambda"].to_numpy(dtype=float)
        y = df_sum["CC_hat_mean"].to_numpy(dtype=float)
        out["lambda_star_emp"] = _interp_root(x, y, 1.0)
    else:
        out["lambda_star_emp"] = None

    # Map lambda* to dependence summaries if possible
    def _map_at_lambda(df: pd.DataFrame, lam_star: Optional[float], col: str) -> Optional[float]:
        if lam_star is None or "lambda" not in df.columns or col not in df.columns:
            return None
        d = df[["lambda", col]].sort_values("lambda")
        x = d["lambda"].to_numpy(dtype=float)
        y = d[col].to_numpy(dtype=float)
        # interpolate y at lam_star
        if lam_star <= x[0]:
            return float(y[0])
        if lam_star >= x[-1]:
            return float(y[-1])
        # find segment
        for i in range(len(x) - 1):
            if x[i] <= lam_star <= x[i + 1]:
                if x[i + 1] == x[i]:
                    return float(y[i])
                t = (lam_star - x[i]) / (x[i + 1] - x[i])
                return float(y[i] + t * (y[i + 1] - y[i]))
        return None

    out["phi_star_pop"] = _map_at_lambda(df_pop, out["lambda_star_pop"], "phi_pop_avg")
    out["tau_star_pop"] = _map_at_lambda(df_pop, out["lambda_star_pop"], "tau_pop_avg")

    # empirical dependence mapping from summary means if present
    out["phi_star_emp"] = _map_at_lambda(df_sum, out["lambda_star_emp"], "phi_hat_avg_mean")
    out["tau_star_emp"] = _map_at_lambda(df_sum, out["lambda_star_emp"], "tau_hat_avg_mean")

    return out


def write_manifest(
    *,
    out_dir: Path,
    cfg_dict: Dict[str, Any],
    thresholds: Dict[str, Any],
    figure_paths: Dict[str, str],
) -> None:
    """
    Write a reproducibility manifest capturing config + environment + outputs.
    """
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "config": cfg_dict,
        "thresholds": thresholds,
        "figures": figure_paths,
    }
    _write_json(out_dir / "manifest.json", manifest)


# ----------------------------
# Main runner
# ----------------------------
def run(config_path: Path, out_dir: Optional[Path] = None) -> Path:
    """
    Execute the full pipeline.

    Returns the output directory used.
    """
    cfg_dict = _load_yaml(str(config_path))
    cfg = _cfg_from_dict(cfg_dict)

    # Output directory
    if out_dir is None:
        base = Path(cfg_dict.get("output_dir", "")) if "output_dir" in cfg_dict else Path("")
        if str(base).strip():
            out_dir = base
        else:
            # Default to experiments/correlation_cliff/artifacts/<stamp>
            out_dir = Path(__file__).resolve().parent / "artifacts" / _now_stamp()

    _ensure_dir(out_dir)

    # Save resolved config snapshot
    _write_json(out_dir / "config_resolved.json", cfg_dict)

    # 1) Population curve
    df_pop = population_curve_from_path(cfg)
    df_pop.to_csv(out_dir / "population_curve.csv", index=False)

    # 2) Simulation
    df_long = simulate_grid(cfg)
    df_long.to_csv(out_dir / "sim_long.csv", index=False)

    df_sum = summarize_simulation(df_long)
    df_sum.to_csv(out_dir / "sim_summary.csv", index=False)

    # 3) Thresholds
    thresholds = estimate_thresholds(df_pop, df_sum)
    _write_json(out_dir / "thresholds.json", thresholds)

    # 4) Figures
    dependence_x = str(cfg_dict.get("dependence_x", "lambda"))
    style = FigureStyle(
        title_prefix=str(cfg_dict.get("title_prefix", "")),
        neutrality_eta=float(cfg_dict.get("neutrality_eta", 0.05)),
    )
    figure_paths = make_all_figures(
        out_dir=out_dir,
        df_pop=df_pop,
        df_sum=df_sum,
        thresholds=thresholds,
        dependence_x=dependence_x,
        style=style,
        also_png=bool(cfg_dict.get("also_png", True)),
    )

    # 5) Manifest
    write_manifest(out_dir=out_dir, cfg_dict=cfg_dict, thresholds=thresholds, figure_paths=figure_paths)

    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the full correlation_cliff experiment pipeline.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config (default: config_s1.yaml next to this file).")
    ap.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    config_path = Path(args.config) if args.config else (Path(__file__).resolve().parent / "config_s1.yaml")
    out_dir = Path(args.out_dir) if args.out_dir else None

    used_out = run(config_path=config_path, out_dir=out_dir)
    print(f"[correlation_cliff] done. outputs written to: {used_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
