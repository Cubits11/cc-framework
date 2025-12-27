from __future__ import annotations

"""
simulate.core
=============

Core simulation harness:
- simulate_replicate_at_lambda
- simulate_grid
- summarize_simulation

This is the heart of the experiment: construct joint -> sample -> compute estimates -> diagnostics.
"""

from typing import Any, Dict, List, Optional, Sequence

import math
import numpy as np
import pandas as pd

from .config import SimConfig
from .config import validate_cfg
from .paths import p11_from_path
from .sampling import draw_joint_counts, empirical_from_counts, rng_for_cell, validate_cell_probs
from . import utils as U


def simulate_replicate_at_lambda(
    cfg: SimConfig,
    *,
    lam: float,
    lam_index: int,
    rep: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Simulate one replicate at a given lambda.
    """
    lam_f = float(lam)
    if not (0.0 <= lam_f <= 1.0) or not math.isfinite(lam_f):
        raise ValueError(f"lambda must be finite and in [0,1], got {lam!r}")
    li = int(lam_index)
    rp = int(rep)
    if li < 0:
        raise ValueError(f"lambda_index must be >=0, got {lam_index!r}")
    if rp < 0:
        raise ValueError(f"rep must be >=0, got {rep!r}")

    out: Dict[str, Any] = {
        "lambda": lam_f,
        "lambda_index": li,
        "rep": rp,
        "rule": cfg.rule,
        "path": cfg.path,
        "seed": int(cfg.seed),
        "seed_policy": cfg.seed_policy,
        "n_per_world": int(cfg.n),
        "hard_fail_on_invalid": bool(cfg.hard_fail_on_invalid),
    }

    jmin, jmax = U.compute_fh_jc_envelope(cfg.marginals, cfg.rule)
    out["JC_env_min"] = float(jmin)
    out["JC_env_max"] = float(jmax)

    _WORLD_NUM_FIELDS = (
        "pA_true", "pB_true", "p00_true", "p01_true", "p10_true", "p11_true",
        "n00", "n01", "n10", "n11",
        "p00_hat", "p01_hat", "p10_hat", "p11_hat",
        "pA_hat", "pB_hat", "pC_hat",
        "phi_hat", "tau_hat",
        "degenerate_A", "degenerate_B", "phi_finite", "tau_finite",
        "pC_true", "phi_true", "tau_true",
    )

    def _prime_world_schema(w: int) -> None:
        out[f"world_valid_w{w}"] = True
        out[f"world_error_stage_w{w}"] = ""
        out[f"world_error_msg_w{w}"] = ""
        for base in _WORLD_NUM_FIELDS:
            out[f"{base}_w{w}"] = float("nan")

    def _mark_world_invalid(w: int, *, stage: str, msg: str) -> None:
        out[f"world_valid_w{w}"] = False
        out[f"world_error_stage_w{w}"] = str(stage)
        out[f"world_error_msg_w{w}"] = str(msg)

    for w in (0, 1):
        _prime_world_schema(w)

    for w, wm in ((0, cfg.marginals.w0), (1, cfg.marginals.w1)):
        out[f"pA_true_w{w}"] = float(wm.pA)
        out[f"pB_true_w{w}"] = float(wm.pB)

        rng_w = rng if cfg.seed_policy == "sequential" else rng_for_cell(cfg.seed, rp, li, w)

        try:
            p11, meta = p11_from_path(wm.pA, wm.pB, lam_f, path=cfg.path, path_params=cfg.path_params)
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="p11_from_path", msg=f"{type(e).__name__}: {e}")
            continue

        for mk, mv in meta.items():
            out[f"{mk}_w{w}"] = float(mv)

        try:
            cells = U.joint_cells_from_marginals(wm.pA, wm.pB, float(p11))
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="joint_cells_from_marginals", msg=f"{type(e).__name__}: {e}")
            continue

        try:
            _ = validate_cell_probs(
                np.array([cells["p00"], cells["p01"], cells["p10"], cells["p11"]], dtype=float),
                prob_tol=cfg.prob_tol,
                allow_tiny_negative=cfg.allow_tiny_negative,
                tiny_negative_eps=cfg.tiny_negative_eps,
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="validate_joint_probs", msg=f"{type(e).__name__}: {e}")
            continue

        out[f"p00_true_w{w}"] = float(cells["p00"])
        out[f"p01_true_w{w}"] = float(cells["p01"])
        out[f"p10_true_w{w}"] = float(cells["p10"])
        out[f"p11_true_w{w}"] = float(cells["p11"])

        try:
            n00, n01, n10, n11 = draw_joint_counts(
                rng_w,
                n=cfg.n,
                p00=float(cells["p00"]),
                p01=float(cells["p01"]),
                p10=float(cells["p10"]),
                p11=float(cells["p11"]),
                prob_tol=cfg.prob_tol,
                allow_tiny_negative=cfg.allow_tiny_negative,
                tiny_negative_eps=cfg.tiny_negative_eps,
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="draw_joint_counts", msg=f"{type(e).__name__}: {e}")
            continue

        out[f"n00_w{w}"] = int(n00)
        out[f"n01_w{w}"] = int(n01)
        out[f"n10_w{w}"] = int(n10)
        out[f"n11_w{w}"] = int(n11)

        try:
            hats = empirical_from_counts(
                n=cfg.n,
                n00=n00,
                n01=n01,
                n10=n10,
                n11=n11,
                rule=cfg.rule,
                context=f"(lam={lam_f:.6g}, idx={li}, rep={rp}, w={w})",
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="empirical_from_counts", msg=f"{type(e).__name__}: {e}")
            continue

        _COLLIDE = {"n", "n00", "n01", "n10", "n11"}
        for hk, hv in hats.items():
            if hk in _COLLIDE:
                continue
            out[f"{hk}_w{w}"] = float(hv)

        try:
            pC_true = float(U.pC_from_joint(cfg.rule, cells, pA=float(wm.pA), pB=float(wm.pB)))
            phi_true = float(U.phi_from_joint(float(wm.pA), float(wm.pB), float(cells["p11"])))
            tau_true = float(U.kendall_tau_a_from_joint(cells))
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="population_overlays", msg=f"{type(e).__name__}: {e}")
            continue

        out[f"pC_true_w{w}"] = float(pC_true)
        out[f"phi_true_w{w}"] = float(phi_true)
        out[f"tau_true_w{w}"] = float(tau_true)

    w0_ok = bool(out.get("world_valid_w0", False))
    w1_ok = bool(out.get("world_valid_w1", False))
    out["worlds_valid"] = bool(w0_ok and w1_ok)

    pA0 = float(out.get("pA_hat_w0", float("nan")))
    pA1 = float(out.get("pA_hat_w1", float("nan")))
    pB0 = float(out.get("pB_hat_w0", float("nan")))
    pB1 = float(out.get("pB_hat_w1", float("nan")))
    pC0 = float(out.get("pC_hat_w0", float("nan")))
    pC1 = float(out.get("pC_hat_w1", float("nan")))

    JA_hat = abs(pA1 - pA0)
    JB_hat = abs(pB1 - pB0)
    Jbest_hat = max(JA_hat, JB_hat)
    dC_hat = (pC1 - pC0)
    JC_hat = abs(dC_hat)
    CC_hat = (JC_hat / Jbest_hat) if (math.isfinite(Jbest_hat) and Jbest_hat > 0.0) else float("nan")

    out["JA_hat"] = float(JA_hat)
    out["JB_hat"] = float(JB_hat)
    out["Jbest_hat"] = float(Jbest_hat)
    out["dC_hat"] = float(dC_hat)
    out["JC_hat"] = float(JC_hat)
    out["CC_hat"] = float(CC_hat)

    def _nan_robust_avg(a: float, b: float) -> float:
        vals = [a, b]
        valid = [v for v in vals if not math.isnan(v)]
        if not valid:
            return float("nan")
        return float(sum(valid) / len(valid))

    phi0 = float(out.get("phi_hat_w0", float("nan")))
    phi1 = float(out.get("phi_hat_w1", float("nan")))
    tau0 = float(out.get("tau_hat_w0", float("nan")))
    tau1 = float(out.get("tau_hat_w1", float("nan")))
    out["phi_hat_avg"] = _nan_robust_avg(phi0, phi1)
    out["tau_hat_avg"] = _nan_robust_avg(tau0, tau1)

    pC0_true = float(out.get("pC_true_w0", float("nan")))
    pC1_true = float(out.get("pC_true_w1", float("nan")))
    dC_pop = pC1_true - pC0_true
    JC_pop = abs(dC_pop)

    JA_pop = abs(float(cfg.marginals.w1.pA) - float(cfg.marginals.w0.pA))
    JB_pop = abs(float(cfg.marginals.w1.pB) - float(cfg.marginals.w0.pB))
    Jbest_pop = max(JA_pop, JB_pop)
    CC_pop = (JC_pop / Jbest_pop) if (Jbest_pop > 0.0 and math.isfinite(Jbest_pop)) else float("nan")

    out["dC_pop"] = float(dC_pop)
    out["JC_pop"] = float(JC_pop)
    out["JA_pop"] = float(JA_pop)
    out["JB_pop"] = float(JB_pop)
    out["Jbest_pop"] = float(Jbest_pop)
    out["CC_pop"] = float(CC_pop)

    phi0_true = float(out.get("phi_true_w0", float("nan")))
    phi1_true = float(out.get("phi_true_w1", float("nan")))
    tau0_true = float(out.get("tau_true_w0", float("nan")))
    tau1_true = float(out.get("tau_true_w1", float("nan")))
    out["phi_pop_avg"] = float(0.5 * (phi0_true + phi1_true))
    out["tau_pop_avg"] = float(0.5 * (tau0_true + tau1_true))

    # Optional: theory reference overlays (separate, explicitly labeled)
    if cfg.include_theory_reference and callable(U.compute_metrics_for_lambda):
        try:
            theory = U.compute_metrics_for_lambda(cfg.marginals, cfg.rule, lam_f)  # type: ignore[misc]
            out["CC_theory_ref"] = float(theory.get("CC", float("nan")))
            out["JC_theory_ref"] = float(theory.get("JC", float("nan")))
            out["dC_theory_ref"] = float(theory.get("dC", float("nan")))
            out["phi_theory_ref_avg"] = float(theory.get("phi_avg", float("nan")))
            out["tau_theory_ref_avg"] = float(theory.get("tau_avg", float("nan")))
            out["CC_ref_minus_pop"] = float(out["CC_theory_ref"] - out["CC_pop"])
            out["JC_ref_minus_pop"] = float(out["JC_theory_ref"] - out["JC_pop"])
        except Exception as e:
            out["theory_ref_error"] = f"{type(e).__name__}: {e}"

    tol = float(cfg.envelope_tol)
    if math.isfinite(JC_hat) and math.isfinite(float(jmin)) and math.isfinite(float(jmax)):
        low = float(jmin) - tol
        high = float(jmax) + tol
        violated_low = JC_hat < low
        violated_high = JC_hat > high
        violated = bool(violated_low or violated_high)
        out["JC_env_violation"] = violated
        out["JC_env_violation_low"] = bool(violated_low)
        out["JC_env_violation_high"] = bool(violated_high)
        if violated_low:
            out["JC_env_gap"] = float(low - JC_hat)
        elif violated_high:
            out["JC_env_gap"] = float(JC_hat - high)
        else:
            out["JC_env_gap"] = 0.0
    else:
        out["JC_env_violation"] = False
        out["JC_env_violation_low"] = False
        out["JC_env_violation_high"] = False
        out["JC_env_gap"] = float("nan")

    return out


def simulate_grid(cfg: SimConfig) -> pd.DataFrame:
    """
    Run simulation across all lambdas and replicates.

    IMPORTANT: stable_per_cell seeding uses cfg.lambda_index_for_seed(lam) so results
    do not change if lambdas are re-ordered.
    """
    validate_cfg(cfg)

    lambdas_list = [float(x) for x in cfg.lambdas]
    if len(lambdas_list) == 0:
        raise ValueError("cfg.lambdas must be non-empty.")
    for i, lam in enumerate(lambdas_list):
        if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
            raise ValueError(f"Invalid lambda at index {i}: got {lam!r}")

    if cfg.seed_policy == "sequential":
        base_rng = np.random.default_rng(int(cfg.seed))
    else:
        base_rng = np.random.default_rng(0)  # intentionally unused

    total = int(cfg.n_reps) * int(len(lambdas_list))
    rows: list[Dict[str, Any]] = []
    rows_append = rows.append

    for rep in range(int(cfg.n_reps)):
        for lam in lambdas_list:
            # canonical lambda index for order-invariance
            lam_index = cfg.lambda_index_for_seed(lam)
            try:
                row = simulate_replicate_at_lambda(
                    cfg,
                    lam=float(lam),
                    lam_index=int(lam_index),
                    rep=int(rep),
                    rng=base_rng,
                )
                row.setdefault("row_ok", True)
                row.setdefault("row_error_stage", "")
                row.setdefault("row_error_msg", "")
                rows_append(row)
            except Exception as e:
                if cfg.hard_fail_on_invalid:
                    raise
                rows_append(
                    {
                        "lambda": float(lam),
                        "lambda_index": int(lam_index),
                        "rep": int(rep),
                        "rule": cfg.rule,
                        "path": cfg.path,
                        "seed": int(cfg.seed),
                        "seed_policy": cfg.seed_policy,
                        "n_per_world": int(cfg.n),
                        "row_ok": False,
                        "row_error_stage": "simulate_replicate_at_lambda",
                        "row_error_msg": f"{type(e).__name__}: {e}",
                    }
                )

    df = pd.DataFrame.from_records(rows)
    sort_cols = [c for c in ("lambda", "rep") if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True) if sort_cols else df.reset_index(drop=True)

    if len(df) != total and cfg.hard_fail_on_invalid:
        raise RuntimeError(f"simulate_grid produced {len(df)} rows, expected {total}")

    return df


def summarize_simulation(
    df_long: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """
    Aggregate replicate-level results to produce per-lambda summaries.
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    if "lambda" not in df_long.columns:
        raise ValueError("df_long must contain a 'lambda' column.")

    q_raw = [float(x) for x in quantiles]
    if len(q_raw) == 0:
        raise ValueError("quantiles must be non-empty.")
    for qq in q_raw:
        if not math.isfinite(qq) or not (0.0 <= qq <= 1.0):
            raise ValueError(f"Invalid quantile {qq!r}")

    q = sorted({round(qq, 12) for qq in q_raw})
    if len(q) != len(q_raw):
        raise ValueError(f"quantiles contains duplicates after rounding: {quantiles!r}")

    def _q_label(qq: float) -> str:
        return f"q{int(round(qq * 1000.0)):04d}"

    def _qcols(s: pd.Series, prefix: str) -> Dict[str, float]:
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        if s2.empty:
            return {f"{prefix}_{_q_label(qq)}": float("nan") for qq in q}
        qs = s2.quantile(q)
        out: Dict[str, float] = {}
        for qq in q:
            out[f"{prefix}_{_q_label(qq)}"] = float(qs.loc[qq])
        return out

    group_keys: List[str] = ["lambda"]
    for k in ("rule", "path"):
        if k in df_long.columns:
            group_keys.append(k)

    df = df_long.copy()
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    if df["lambda"].isna().any():
        bad = df_long.loc[df["lambda"].isna(), "lambda"].head(5).tolist()
        raise ValueError(f"Found non-numeric lambda values (first few): {bad!r}")

    core_cols = (
        "CC_hat", "JC_hat", "dC_hat", "JA_hat", "JB_hat", "Jbest_hat",
        "phi_hat_avg", "tau_hat_avg",
    )

    pop_cols = (
        "CC_pop", "JC_pop", "dC_pop", "phi_pop_avg", "tau_pop_avg",
        "JC_env_min", "JC_env_max",
    )

    theory_cols = (
        "CC_theory_ref", "JC_theory_ref", "dC_theory_ref",
        "phi_theory_ref_avg", "tau_theory_ref_avg",
    )

    groups: list[Dict[str, Any]] = []
    gb = df.groupby(group_keys, sort=True, dropna=False)

    for key, g in gb:
        row: Dict[str, Any] = {}
        if isinstance(key, tuple):
            for kname, kval in zip(group_keys, key):
                row[kname] = float(kval) if kname == "lambda" else (None if pd.isna(kval) else str(kval))
        else:
            row["lambda"] = float(key)

        row["n_rows"] = int(len(g))
        row["n_reps"] = int(pd.to_numeric(g["rep"], errors="coerce").nunique(dropna=True)) if "rep" in g.columns else int(len(g))

        if "row_ok" in g.columns:
            ok = g["row_ok"].fillna(False).astype(bool)
            row["row_ok_rate"] = float(ok.mean()) if len(ok) > 0 else float("nan")
            row["n_row_ok"] = int(ok.sum())
            row["n_row_fail"] = int((~ok).sum())
        else:
            row["row_ok_rate"] = float("nan")
            row["n_row_ok"] = int(len(g))
            row["n_row_fail"] = 0

        for col in core_cols:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(s.mean())
                row[f"{col}_std"] = float(s.std(ddof=1)) if s.notna().sum() >= 2 else float("nan")
                row[f"{col}_n"] = int(s.notna().sum())
                row[f"{col}_nan_rate"] = float(s.isna().mean())

        if "CC_hat" in g.columns:
            row.update(_qcols(g["CC_hat"], "CC_hat"))
        if "JC_hat" in g.columns:
            row.update(_qcols(g["JC_hat"], "JC_hat"))

        if "JC_env_violation" in g.columns:
            v = g["JC_env_violation"].fillna(False).astype(bool)
            row["JC_env_violation_rate"] = float(v.mean()) if len(v) > 0 else float("nan")
            row["JC_env_violation_n"] = int(v.sum())
        else:
            row["JC_env_violation_rate"] = float("nan")
            row["JC_env_violation_n"] = 0

        inv_cols = [c for c in g.columns if c.startswith("invalid_joint_w")]
        for c in inv_cols:
            v = g[c].fillna(False).astype(bool)
            row[f"{c}_rate"] = float(v.mean()) if len(v) > 0 else float("nan")
            row[f"{c}_n"] = int(v.sum())

        for col in pop_cols:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                if s.empty:
                    row[col] = float("nan")
                    row[f"{col}_drift"] = float("nan")
                    row[f"{col}_nonconstant"] = False
                else:
                    v0 = float(s.iloc[0])
                    drift = float(s.max() - s.min()) if len(s) > 1 else 0.0
                    row[col] = v0
                    row[f"{col}_drift"] = drift
                    row[f"{col}_nonconstant"] = bool(drift != 0.0)

        for col in theory_cols:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                if s.empty:
                    row[col] = float("nan")
                    row[f"{col}_drift"] = float("nan")
                    row[f"{col}_nonconstant"] = False
                else:
                    v0 = float(s.iloc[0])
                    drift = float(s.max() - s.min()) if len(s) > 1 else 0.0
                    row[col] = v0
                    row[f"{col}_drift"] = drift
                    row[f"{col}_nonconstant"] = bool(drift != 0.0)

        groups.append(row)

    df_sum = pd.DataFrame(groups)
    sort_cols = [c for c in ("rule", "path", "lambda") if c in df_sum.columns]
    df_sum = df_sum.sort_values(sort_cols, kind="mergesort").reset_index(drop=True) if sort_cols else df_sum.sort_values("lambda", kind="mergesort").reset_index(drop=True)
    return df_sum
