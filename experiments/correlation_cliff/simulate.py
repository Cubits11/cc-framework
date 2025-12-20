# experiments/correlation_cliff/simulate.py
from __future__ import annotations

"""
Correlation Cliff — Simulation Module
=====================================

This module generates *finite-sample* empirical estimates for the correlation-cliff
experiment under a *fixed-marginals, variable-dependence* model.

Core idea
---------
For each world w ∈ {0,1}, rails (A,B) are binary with fixed marginals (pA^w, pB^w).
Dependence is controlled by selecting the feasible overlap probability:

    p11^w = P(A=1,B=1)

within Fréchet–Hoeffding bounds, optionally via different "paths" through the feasible
set. Given p11^w, the full 2×2 joint is:

    p10^w = pA^w - p11^w
    p01^w = pB^w - p11^w
    p00^w = 1 - pA^w - pB^w + p11^w

We then draw counts from a multinomial distribution:

    (N00, N01, N10, N11) ~ Multinomial(n; p00, p01, p10, p11)

and compute empirical estimates:
- pA_hat^w, pB_hat^w, p11_hat^w
- pC_hat^w under rule OR/AND
- J_A_hat, J_B_hat, J_C_hat and CC_hat = J_C_hat / max(J_A_hat, J_B_hat)
- dependence summaries: phi_hat^w, tau_hat^w

Outputs
-------
- replicate-level long DataFrame (one row per replicate per lambda)
- optional aggregated summary DataFrame (mean/std/quantiles per lambda)

This file is designed to be imported by run_all.py, but can also be run as a script.

Notes on rigor / sanity checks
------------------------------
- We compute FH envelopes for J_C (theory module) and optionally flag empirical values
  outside the envelope beyond a tolerance (finite-sample variability can cause small
  overshoots if you compare to population envelopes too strictly; by default we "flag"
  rather than raising).
- We keep worlds independent by default. If you want common random numbers as a variance
  reduction trick, implement paired sampling explicitly (not enabled here).

Dependencies
------------
numpy, pandas. Optional: pyyaml for CLI config loading.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd

Rule = Literal["OR", "AND"]
Path = Literal["fh_linear", "fh_power", "fh_scurve", "gaussian_tau"]


# ----------------------------
# Imports from theory.py
# ----------------------------
try:
    # If run as a package module
    from .theory import (  # type: ignore
        FHBounds,
        TwoWorldMarginals,
        WorldMarginals,
        compute_fh_jc_envelope,
        compute_metrics_for_lambda,
        fh_bounds,
        joint_cells_from_marginals,
        kendall_tau_a_from_joint,
        phi_from_joint,
        p11_fh_linear,
        pC_from_joint,
    )
except Exception:
    # If run as a script from this folder
    from theory import (  # type: ignore
        FHBounds,
        TwoWorldMarginals,
        WorldMarginals,
        compute_fh_jc_envelope,
        compute_metrics_for_lambda,
        fh_bounds,
        joint_cells_from_marginals,
        kendall_tau_a_from_joint,
        phi_from_joint,
        p11_fh_linear,
        pC_from_joint,
    )


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class SimConfig:
    """
    Simulation configuration.

    Parameters
    ----------
    marginals:
        TwoWorldMarginals with fixed pA/pB per world.
    rule:
        "OR" or "AND".
    lambdas:
        Dependence grid, typically in [0,1].
    n:
        Sample size per world per lambda per replicate.
    n_reps:
        Number of Monte Carlo replicates per lambda.
    seed:
        RNG seed for reproducibility.
    path:
        Dependence path name; defaults to FH-linear.
    path_params:
        Path-specific parameters.
    envelope_tol:
        Tolerance for comparing empirical J_C_hat to population envelope [jmin,jmax].
        This is a *flagging* threshold, not a hard feasibility constraint.
    hard_fail_on_invalid:
        If True, raise on any invalid probability construction; otherwise mark flags.
    """

    marginals: TwoWorldMarginals
    rule: Rule
    lambdas: Sequence[float]
    n: int
    n_reps: int = 1
    seed: int = 0

    path: Path = "fh_linear"
    path_params: Dict[str, Any] = field(default_factory=dict)

    envelope_tol: float = 5e-3
    hard_fail_on_invalid: bool = True


# ----------------------------
# Helpers: lambda transforms / paths
# ----------------------------
def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _lam_power(lam: float, gamma: float) -> float:
    if gamma <= 0:
        raise ValueError(f"gamma must be >0, got {gamma}")
    return float(lam**gamma)


def _lam_scurve(lam: float, k: float) -> float:
    """
    Smooth S-curve in [0,1] that preserves endpoints:
      s(lam) = sigmoid(k(lam-0.5)) rescaled to hit exactly 0 at lam=0 and 1 at lam=1.

    This gives a nonlinear traversal of FH interval without leaving feasibility.
    """
    if k <= 0:
        raise ValueError(f"k must be >0, got {k}")

    def sig(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    a = sig(-0.5 * k)
    b = sig(+0.5 * k)
    s = sig(k * (lam - 0.5))
    # Rescale s from [a,b] to [0,1]
    if b == a:
        return float(lam)
    return float((s - a) / (b - a))


def _bvn_cdf_scipy(x: float, y: float, rho: float) -> float:
    """
    Bivariate normal CDF P(Z1<=x, Z2<=y) with corr rho, using SciPy if available.

    Raises ImportError if SciPy is not installed.
    """
    try:
        from scipy.stats import multivariate_normal  # type: ignore
    except Exception as e:
        raise ImportError("SciPy not available for gaussian_tau path") from e

    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return float(multivariate_normal(mean=mean, cov=cov).cdf([x, y]))


def _p11_gaussian_tau(pA: float, pB: float, tau: float) -> float:
    """
    Gaussian copula overlap p11 = C(u,v) with u=pA, v=pB where concordance is set by Kendall tau.

    For Gaussian copula: tau = 2/pi * arcsin(rho)  => rho = sin(pi*tau/2)

    Then p11 = Phi2(Phi^{-1}(u), Phi^{-1}(v); rho).

    This provides a smooth alternative to FH-linear.
    """
    if not (-1.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [-1,1], got {tau}")

    # Convert to rho
    rho = math.sin(math.pi * tau / 2.0)

    try:
        from scipy.stats import norm  # type: ignore
    except Exception as e:
        raise ImportError("SciPy not available for gaussian_tau path") from e

    x = float(norm.ppf(pA))
    y = float(norm.ppf(pB))

    # p11 is C(u,v) = Phi2(x,y; rho)
    p11 = _bvn_cdf_scipy(x, y, rho)
    return p11


def p11_from_path(
    pA: float,
    pB: float,
    lam: float,
    *,
    path: Path,
    path_params: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute p11 given marginals and a dependence parameter lam ∈ [0,1].

    Returns
    -------
    p11, meta
      meta may include:
        - lam_eff: transformed lambda
        - tau: if gaussian_tau path
        - rho: if gaussian_tau path
        - L,U: FH bounds
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lambda must be in [0,1], got {lam}")

    b = fh_bounds(pA, pB)
    meta: Dict[str, float] = {"L": float(b.L), "U": float(b.U)}

    if path == "fh_linear":
        p11 = p11_fh_linear(pA, pB, lam)
        meta["lam_eff"] = float(lam)
        return float(p11), meta

    if path == "fh_power":
        gamma = float(path_params.get("gamma", 1.0))
        lam_eff = _lam_power(lam, gamma)
        p11 = b.L + lam_eff * (b.U - b.L)
        meta["lam_eff"] = float(lam_eff)
        meta["gamma"] = float(gamma)
        return float(p11), meta

    if path == "fh_scurve":
        k = float(path_params.get("k", 8.0))
        lam_eff = _lam_scurve(lam, k)
        p11 = b.L + lam_eff * (b.U - b.L)
        meta["lam_eff"] = float(lam_eff)
        meta["k"] = float(k)
        return float(p11), meta

    if path == "gaussian_tau":
        # Map lam∈[0,1] -> tau∈[-1,1]
        tau = 2.0 * lam - 1.0
        try:
            p11 = _p11_gaussian_tau(pA, pB, tau)
        except ImportError as e:
            raise ImportError(
                "gaussian_tau path requested but SciPy is unavailable. "
                "Install scipy or choose fh_linear/fh_power/fh_scurve."
            ) from e

        # Numerical guard: ensure within FH bounds (should be, but floating may drift)
        p11_clipped = min(max(p11, b.L), b.U)
        meta["lam_eff"] = float(lam)
        meta["tau"] = float(tau)
        meta["rho"] = float(math.sin(math.pi * tau / 2.0))
        meta["clip_amt"] = float(p11 - p11_clipped)
        return float(p11_clipped), meta

    raise ValueError(f"Unknown path: {path}")


# ----------------------------
# Core simulation pieces
# ----------------------------
def _draw_joint_counts(
    rng: np.random.Generator,
    *,
    n: int,
    p00: float,
    p01: float,
    p10: float,
    p11: float,
) -> Tuple[int, int, int, int]:
    """
    Draw multinomial joint counts (N00, N01, N10, N11).

    Order follows theory.joint_cells_from_marginals dict keys:
        p00, p01, p10, p11
    """
    p = np.array([p00, p01, p10, p11], dtype=float)
    # Normalize defensively
    s = float(p.sum())
    if s <= 0:
        raise ValueError("Invalid probability sum in multinomial draw.")
    p = p / s
    counts = rng.multinomial(int(n), pvals=p, size=1)[0]
    return int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])


def _empirical_from_counts(
    *,
    n: int,
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    rule: Rule,
) -> Dict[str, float]:
    """
    Compute empirical probabilities from joint counts, plus phi and tau summaries.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if (n00 + n01 + n10 + n11) != n:
        raise ValueError("Counts do not sum to n.")

    p00 = n00 / n
    p01 = n01 / n
    p10 = n10 / n
    p11 = n11 / n

    pA = (n10 + n11) / n
    pB = (n01 + n11) / n

    cells = {"p00": p00, "p01": p01, "p10": p10, "p11": p11}
    pC = pC_from_joint(rule, cells, pA=pA, pB=pB)
    phi = phi_from_joint(pA, pB, p11)
    tau = kendall_tau_a_from_joint(cells)

    return {
        "p00_hat": float(p00),
        "p01_hat": float(p01),
        "p10_hat": float(p10),
        "p11_hat": float(p11),
        "pA_hat": float(pA),
        "pB_hat": float(pB),
        "pC_hat": float(pC),
        "phi_hat": float(phi),
        "tau_hat": float(tau),
    }


def simulate_replicate_at_lambda(
    cfg: SimConfig,
    *,
    lam: float,
    rep: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Simulate one replicate at a given lambda.

    Returns a single row dict (wide format) containing:
      - lambda, rep
      - true cell probs per world
      - sampled counts per world
      - empirical hats per world
      - derived metrics (J_A_hat, J_B_hat, J_C_hat, CC_hat)
      - theory values for reference (CC_theory, etc.)
      - envelope flags
    """
    # -------- Theory at lam (population quantities for overlay) --------
    theory = compute_metrics_for_lambda(cfg.marginals, cfg.rule, lam)

    # -------- Build per-world joint probs (true) and sample --------
    out: Dict[str, Any] = {"lambda": float(lam), "rep": int(rep), "rule": cfg.rule, "path": cfg.path}

    # FH envelope for J_C based on *true* marginals (path-independent; feasibility only)
    jmin, jmax = compute_fh_jc_envelope(cfg.marginals, cfg.rule)
    out["JC_env_min"] = float(jmin)
    out["JC_env_max"] = float(jmax)

    # Per world sampling
    for w, wm in [(0, cfg.marginals.w0), (1, cfg.marginals.w1)]:
        p11, meta = p11_from_path(
            wm.pA,
            wm.pB,
            lam,
            path=cfg.path,
            path_params=cfg.path_params,
        )

        # Construct joint cells (true probabilities)
        try:
            cells = joint_cells_from_marginals(wm.pA, wm.pB, p11)
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            # Mark invalid and fill NaNs
            out[f"invalid_joint_w{w}"] = True
            out[f"invalid_joint_msg_w{w}"] = str(e)
            for k in ["p00", "p01", "p10", "p11"]:
                out[f"{k}_true_w{w}"] = float("nan")
            continue

        # Store meta (e.g., clip_amt for gaussian_tau)
        for mk, mv in meta.items():
            out[f"{mk}_w{w}"] = float(mv)

        out[f"pA_true_w{w}"] = float(wm.pA)
        out[f"pB_true_w{w}"] = float(wm.pB)
        out[f"p00_true_w{w}"] = float(cells["p00"])
        out[f"p01_true_w{w}"] = float(cells["p01"])
        out[f"p10_true_w{w}"] = float(cells["p10"])
        out[f"p11_true_w{w}"] = float(cells["p11"])

        # Sample multinomial joint counts
        n00, n01, n10, n11 = _draw_joint_counts(
            rng,
            n=cfg.n,
            p00=cells["p00"],
            p01=cells["p01"],
            p10=cells["p10"],
            p11=cells["p11"],
        )
        out[f"n00_w{w}"] = int(n00)
        out[f"n01_w{w}"] = int(n01)
        out[f"n10_w{w}"] = int(n10)
        out[f"n11_w{w}"] = int(n11)

        # Empirical estimates from counts
        hats = _empirical_from_counts(n=cfg.n, n00=n00, n01=n01, n10=n10, n11=n11, rule=cfg.rule)
        for hk, hv in hats.items():
            out[f"{hk}_w{w}"] = float(hv)

    # -------- Derived empirical metrics across worlds --------
    pA0 = out.get("pA_hat_w0", float("nan"))
    pA1 = out.get("pA_hat_w1", float("nan"))
    pB0 = out.get("pB_hat_w0", float("nan"))
    pB1 = out.get("pB_hat_w1", float("nan"))
    pC0 = out.get("pC_hat_w0", float("nan"))
    pC1 = out.get("pC_hat_w1", float("nan"))

    JA_hat = abs(pA1 - pA0)
    JB_hat = abs(pB1 - pB0)
    Jbest_hat = max(JA_hat, JB_hat)
    dC_hat = (pC1 - pC0)
    JC_hat = abs(dC_hat)
    CC_hat = (JC_hat / Jbest_hat) if Jbest_hat > 0 else float("nan")

    out["JA_hat"] = float(JA_hat)
    out["JB_hat"] = float(JB_hat)
    out["Jbest_hat"] = float(Jbest_hat)
    out["dC_hat"] = float(dC_hat)
    out["JC_hat"] = float(JC_hat)
    out["CC_hat"] = float(CC_hat)

    # Empirical dependence summaries averaged across worlds
    phi0 = out.get("phi_hat_w0", float("nan"))
    phi1 = out.get("phi_hat_w1", float("nan"))
    tau0 = out.get("tau_hat_w0", float("nan"))
    tau1 = out.get("tau_hat_w1", float("nan"))
    out["phi_hat_avg"] = float(0.5 * (phi0 + phi1))
    out["tau_hat_avg"] = float(0.5 * (tau0 + tau1))

    # -------- Theory overlays --------
    # Copy only key theory quantities (keep table sane)
    out["CC_theory"] = float(theory["CC"])
    out["JC_theory"] = float(theory["JC"])
    out["dC_theory"] = float(theory["dC"])
    out["phi_theory_avg"] = float(theory["phi_avg"])
    out["tau_theory_avg"] = float(theory["tau_avg"])

    # -------- Envelope flagging --------
    # Compare empirical JC_hat to population envelope [jmin,jmax].
    # With finite n, small deviations are possible; we flag beyond tolerance.
    tol = float(cfg.envelope_tol)
    out["JC_env_violation"] = bool((JC_hat < (jmin - tol)) or (JC_hat > (jmax + tol)))
    out["JC_env_gap"] = float(
        0.0
        if not out["JC_env_violation"]
        else max(jmin - JC_hat, JC_hat - jmax)
    )

    return out


def simulate_grid(cfg: SimConfig) -> pd.DataFrame:
    """
    Run simulation across all lambdas and replicates.

    Returns
    -------
    df_long : pd.DataFrame
        One row per (lambda, rep).
    """
    if cfg.n <= 0:
        raise ValueError("n must be positive.")
    if cfg.n_reps <= 0:
        raise ValueError("n_reps must be positive.")
    if len(cfg.lambdas) < 1:
        raise ValueError("lambdas must be non-empty.")

    # Use a single RNG for reproducibility
    rng = np.random.default_rng(int(cfg.seed))

    rows: list[Dict[str, Any]] = []
    for rep in range(cfg.n_reps):
        for lam in cfg.lambdas:
            rows.append(simulate_replicate_at_lambda(cfg, lam=float(lam), rep=rep, rng=rng))

    df = pd.DataFrame(rows).sort_values(["lambda", "rep"]).reset_index(drop=True)
    return df


def summarize_simulation(
    df_long: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """
    Aggregate replicate-level results to produce per-lambda summaries.

    Summary columns include:
      - mean/std for CC_hat, JC_hat, etc.
      - quantiles for CC_hat and JC_hat
      - violation rate for envelope flag

    Returns
    -------
    df_sum : pd.DataFrame
        One row per lambda.
    """
    if df_long.empty:
        return df_long.copy()

    q = list(quantiles)

    def qcols(s: pd.Series, prefix: str) -> Dict[str, float]:
        qs = s.quantile(q)
        return {f"{prefix}_q{int(qq*1000):04d}": float(qs.loc[qq]) for qq in q}

    groups = []
    for lam, g in df_long.groupby("lambda", sort=True):
        row: Dict[str, Any] = {"lambda": float(lam)}
        for col in ["CC_hat", "JC_hat", "dC_hat", "JA_hat", "JB_hat", "Jbest_hat", "phi_hat_avg", "tau_hat_avg"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else float("nan")
        # Quantiles for headline stats
        if "CC_hat" in g.columns:
            row.update(qcols(g["CC_hat"], "CC_hat"))
        if "JC_hat" in g.columns:
            row.update(qcols(g["JC_hat"], "JC_hat"))

        # Theory (identical across reps; take first)
        for col in ["CC_theory", "JC_theory", "dC_theory", "phi_theory_avg", "tau_theory_avg", "JC_env_min", "JC_env_max"]:
            if col in g.columns:
                row[col] = float(g[col].iloc[0])

        # Envelope violation rate
        if "JC_env_violation" in g.columns:
            row["JC_env_violation_rate"] = float(g["JC_env_violation"].mean())

        groups.append(row)

    return pd.DataFrame(groups).sort_values("lambda").reset_index(drop=True)


# ----------------------------
# Convenience: grid builder
# ----------------------------
def build_linear_lambda_grid(num: int, *, include_endpoints: bool = True) -> np.ndarray:
    """
    Build a linear lambda grid in [0,1].

    If include_endpoints=True, returns np.linspace(0,1,num).
    Else returns interior points only.
    """
    if num < 2:
        raise ValueError("num must be >= 2 for a usable grid.")
    grid = np.linspace(0.0, 1.0, num=num, dtype=float)
    if include_endpoints:
        return grid
    return grid[1:-1]


# ----------------------------
# CLI
# ----------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError("pyyaml is required for --config usage") from e

    with open(path, "r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f))


def _cfg_from_dict(d: Dict[str, Any]) -> SimConfig:
    """
    Build SimConfig from a YAML-like dict.

    Expected schema (minimal):
      marginals:
        w0: {pA: ..., pB: ...}
        w1: {pA: ..., pB: ...}
      rule: "OR"|"AND"
      n: int
      n_reps: int
      seed: int
      lambdas: [ ... ]  OR  lambda_grid: {num: int}
      path: "fh_linear"|"fh_power"|"fh_scurve"|"gaussian_tau"
      path_params: {...}
    """
    md = d.get("marginals", {})
    w0 = md.get("w0", {})
    w1 = md.get("w1", {})

    marg = TwoWorldMarginals(
        w0=WorldMarginals(pA=float(w0["pA"]), pB=float(w0["pB"])),
        w1=WorldMarginals(pA=float(w1["pA"]), pB=float(w1["pB"])),
    )

    if "lambdas" in d:
        lambdas = [float(x) for x in d["lambdas"]]
    else:
        lg = d.get("lambda_grid", {"num": 21})
        num = int(lg.get("num", 21))
        lambdas = list(build_linear_lambda_grid(num).tolist())

    return SimConfig(
        marginals=marg,
        rule=str(d.get("rule", "OR")).upper(),  # type: ignore
        lambdas=lambdas,
        n=int(d["n"]),
        n_reps=int(d.get("n_reps", 1)),
        seed=int(d.get("seed", 0)),
        path=str(d.get("path", "fh_linear")),  # type: ignore
        path_params=dict(d.get("path_params", {})),
        envelope_tol=float(d.get("envelope_tol", 5e-3)),
        hard_fail_on_invalid=bool(d.get("hard_fail_on_invalid", True)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser(description="Run correlation cliff simulation grid.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    ap.add_argument("--out_csv", type=str, default=None, help="Write replicate-level rows to CSV.")
    ap.add_argument("--out_summary_csv", type=str, default=None, help="Write per-lambda summary to CSV.")
    ap.add_argument("--print_head", type=int, default=5, help="Print first N rows of summary.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.config is None:
        raise SystemExit("Please provide --config path/to/config.yaml (or call simulate_grid() from Python).")

    cfg_dict = _load_yaml(args.config)
    cfg = _cfg_from_dict(cfg_dict)

    df_long = simulate_grid(cfg)
    df_sum = summarize_simulation(df_long)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df_long.to_csv(args.out_csv, index=False)

    if args.out_summary_csv:
        os.makedirs(os.path.dirname(args.out_summary_csv) or ".", exist_ok=True)
        df_sum.to_csv(args.out_summary_csv, index=False)

    if args.print_head and args.print_head > 0:
        with pd.option_context("display.width", 160, "display.max_columns", 200):
            print(df_sum.head(int(args.print_head)))

    # Minimal diagnostics
    vio_rate = float(df_long["JC_env_violation"].mean()) if "JC_env_violation" in df_long.columns else float("nan")
    print(json.dumps({"rows": int(len(df_long)), "lambda_points": int(len(set(df_long["lambda"]))), "env_violation_rate": vio_rate}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
