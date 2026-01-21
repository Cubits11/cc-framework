from __future__ import annotations

"""
simulate.core
=============

Core simulation harness:
- simulate_replicate_at_lambda
- simulate_grid
- summarize_simulation

Research-OS guarantees:
- Deterministic results under stable_per_cell seeding, invariant to lambda ordering.
- Sequential seeding is deterministic for a given iteration order, but order-dependent by design.
- Explicit per-world validity flags + stage + message.
- Stable output schema with no meta-key collisions.

Design notes (Correlation Cliff):
- Two worlds w∈{0,1} with fixed marginals (pA,pB) per world.
- Dependence enters only via p11 = P(A=1,B=1), chosen via p11_from_path(...).
- For each world, we construct a valid 2x2 joint, sample multinomial counts,
  compute empirical metrics, and overlay population (“true”) metrics.
- Across worlds, we compute CC_hat and CC_pop:
    JA = |pA1 - pA0|, JB = |pB1 - pB0|, Jbest = max(JA, JB)
    JC = |pC1 - pC0|, CC = JC / Jbest
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import utils as U
from .config import SimConfig, validate_cfg
from .paths import p11_from_path
from .sampling import draw_joint_counts, empirical_from_counts, rng_for_cell, validate_cell_probs


# ----------------------------
# Local helpers (strictness)
# ----------------------------
def _strict_nonneg_int(x: Any, name: str) -> int:
    if isinstance(x, bool):
        raise TypeError(f"{name} must be an int >= 0, got bool {x!r}")
    if not isinstance(x, int):
        raise TypeError(f"{name} must be an int >= 0, got {type(x).__name__}={x!r}")
    if x < 0:
        raise ValueError(f"{name} must be >= 0, got {x}")
    return x


def _finite_unit_interval(x: Any, name: str) -> float:
    if isinstance(x, bool):
        raise TypeError(f"{name} must be a finite float in [0,1], got bool {x!r}")
    xf = float(x)
    if not (math.isfinite(xf) and 0.0 <= xf <= 1.0):
        raise ValueError(f"{name} must be finite and in [0,1], got {x!r}")
    return xf


def _nan_robust_avg(a: float, b: float) -> float:
    vals = [a, b]
    valid = [v for v in vals if math.isfinite(v) and not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _bool_to_float(b: Any) -> float:
    try:
        return 1.0 if bool(b) else 0.0
    except Exception:
        return 0.0


# ----------------------------
# Per-replicate simulation
# ----------------------------
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

    Notes on RNG:
    - stable_per_cell: ignores `rng` and uses rng_for_cell(seed, rep, lam_index, world)
    - sequential: uses `rng` as a base stream, but derives per-world substreams to avoid
      world-to-world coupling while retaining order dependence.
    """
    lam_f = _finite_unit_interval(lam, "lambda")
    li = _strict_nonneg_int(lam_index, "lambda_index")
    rp = _strict_nonneg_int(rep, "rep")

    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")

    row_ctx = f"(lam={lam_f:.6g}, idx={li}, rep={rp})"

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
        "row_ok": True,  # refined after world runs
        "row_error_stage": "",
        "row_error_msg": "",
    }

    # Envelope is a population property; always computable from marginals + rule.
    jmin, jmax = U.compute_fh_jc_envelope(cfg.marginals, cfg.rule)
    out["JC_env_min"] = float(jmin)
    out["JC_env_max"] = float(jmax)

    # World schema: keep consistent column names across all rows.
    _WORLD_NUM_FIELDS: Tuple[str, ...] = (
        "pA_true",
        "pB_true",
        "p00_true",
        "p01_true",
        "p10_true",
        "p11_true",
        "n00",
        "n01",
        "n10",
        "n11",
        "p00_hat",
        "p01_hat",
        "p10_hat",
        "p11_hat",
        "pA_hat",
        "pB_hat",
        "pC_hat",
        "phi_hat",
        "tau_hat",
        "degenerate_A",
        "degenerate_B",
        "phi_finite",
        "tau_finite",
        "pC_true",
        "phi_true",
        "tau_true",
    )

    def _prime_world_schema(w: int) -> None:
        out[f"world_valid_w{w}"] = True
        out[f"invalid_joint_w{w}"] = False  # summarizer expects this prefix
        out[f"world_error_stage_w{w}"] = ""
        out[f"world_error_msg_w{w}"] = ""
        out[f"rng_seed_w{w}"] = float("nan")  # only populated for sequential
        for base in _WORLD_NUM_FIELDS:
            out[f"{base}_w{w}"] = float("nan")

    def _mark_world_invalid(w: int, *, stage: str, msg: str) -> None:
        out[f"world_valid_w{w}"] = False
        out[f"world_error_stage_w{w}"] = str(stage)
        out[f"world_error_msg_w{w}"] = str(msg)

        # Flag as "invalid_joint" for any upstream joint/sampling issues.
        out[f"invalid_joint_w{w}"] = stage in (
            "p11_from_path",
            "joint_cells_from_marginals",
            "validate_joint_probs",
            "draw_joint_counts",
        )

    for w in (0, 1):
        _prime_world_schema(w)

    # ----------------------------
    # World loop
    # ----------------------------
    for w, wm in ((0, cfg.marginals.w0), (1, cfg.marginals.w1)):
        w_ctx = f"{row_ctx}, w={w}"
        out[f"pA_true_w{w}"] = float(wm.pA)
        out[f"pB_true_w{w}"] = float(wm.pB)

        # RNG policy:
        if cfg.seed_policy == "stable_per_cell":
            rng_w = rng_for_cell(cfg.seed, rp, li, w)
        else:
            # Derive per-world substream seed from the sequential base RNG (order-dependent by design),
            # to reduce artificial cross-world coupling.
            seed_w = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
            out[f"rng_seed_w{w}"] = float(seed_w)
            rng_w = np.random.default_rng(seed_w)

        # p11 from chosen path
        try:
            p11, meta = p11_from_path(
                float(wm.pA),
                float(wm.pB),
                lam_f,
                path=cfg.path,
                path_params=cfg.path_params,
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="p11_from_path", msg=f"{type(e).__name__}: {e}")
            continue

        # Meta: prefix to avoid collisions with real columns.
        if isinstance(meta, dict):
            for mk, mv in meta.items():
                try:
                    out[f"pathmeta_{mk}_w{w}"] = float(mv)
                except Exception:
                    # Keep meta robust; never allow it to crash the simulation.
                    out[f"pathmeta_{mk}_w{w}"] = float("nan")

        # Build joint cells
        try:
            cells = U.joint_cells_from_marginals(float(wm.pA), float(wm.pB), float(p11))
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="joint_cells_from_marginals", msg=f"{type(e).__name__}: {e}")
            continue

        # Validate joint probabilities (no renorm; tiny clip allowed)
        try:
            _ = validate_cell_probs(
                np.array(
                    [cells["p00"], cells["p01"], cells["p10"], cells["p11"]], dtype=np.float64
                ),
                prob_tol=cfg.prob_tol,
                allow_tiny_negative=cfg.allow_tiny_negative,
                tiny_negative_eps=cfg.tiny_negative_eps,
                context=w_ctx,
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

        # Sample multinomial counts
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
                context=w_ctx,
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

        # Empirical estimates from counts
        try:
            hats = empirical_from_counts(
                n=cfg.n,
                n00=n00,
                n01=n01,
                n10=n10,
                n11=n11,
                rule=cfg.rule,
                context=w_ctx,
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="empirical_from_counts", msg=f"{type(e).__name__}: {e}")
            continue

        # Avoid collisions with raw count fields already stored in out.
        _COLLIDE = {"n", "n00", "n01", "n10", "n11"}
        for hk, hv in hats.items():
            if hk in _COLLIDE:
                continue
            out[f"{hk}_w{w}"] = float(hv)

        # Population overlays (true metrics)
        try:
            pC_true = float(U.pC_from_joint(cfg.rule, cells, pA=float(wm.pA), pB=float(wm.pB)))
            phi_true = float(U.phi_from_joint(float(wm.pA), float(wm.pB), float(cells["p11"])))
            tau_true = float(U.kendall_tau_a_from_joint(cells))
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="population_overlays", msg=f"{type(e).__name__}: {e}")
            continue

        out[f"pC_true_w{w}"] = pC_true
        out[f"phi_true_w{w}"] = phi_true
        out[f"tau_true_w{w}"] = tau_true

    # ----------------------------
    # Cross-world aggregate metrics
    # ----------------------------
    w0_ok = bool(out.get("world_valid_w0", False))
    w1_ok = bool(out.get("world_valid_w1", False))
    worlds_valid = bool(w0_ok and w1_ok)
    out["worlds_valid"] = worlds_valid

    # Row OK means: replicate is fully valid (both worlds valid)
    out["row_ok"] = worlds_valid

    if worlds_valid:
        pA0 = float(out["pA_hat_w0"])
        pA1 = float(out["pA_hat_w1"])
        pB0 = float(out["pB_hat_w0"])
        pB1 = float(out["pB_hat_w1"])
        pC0 = float(out["pC_hat_w0"])
        pC1 = float(out["pC_hat_w1"])

        JA_hat = abs(pA1 - pA0)
        JB_hat = abs(pB1 - pB0)
        Jbest_hat = max(JA_hat, JB_hat)
        dC_hat = pC1 - pC0
        JC_hat = abs(dC_hat)
        CC_hat = (
            (JC_hat / Jbest_hat) if (math.isfinite(Jbest_hat) and Jbest_hat > 0.0) else float("nan")
        )

        out["JA_hat"] = float(JA_hat)
        out["JB_hat"] = float(JB_hat)
        out["Jbest_hat"] = float(Jbest_hat)
        out["dC_hat"] = float(dC_hat)
        out["JC_hat"] = float(JC_hat)
        out["CC_hat"] = float(CC_hat)

        phi0 = float(out.get("phi_hat_w0", float("nan")))
        phi1 = float(out.get("phi_hat_w1", float("nan")))
        tau0 = float(out.get("tau_hat_w0", float("nan")))
        tau1 = float(out.get("tau_hat_w1", float("nan")))
        out["phi_hat_avg"] = _nan_robust_avg(phi0, phi1)
        out["tau_hat_avg"] = _nan_robust_avg(tau0, tau1)
    else:
        # Explicitly mark cross-world outputs as NaN
        for k in (
            "JA_hat",
            "JB_hat",
            "Jbest_hat",
            "dC_hat",
            "JC_hat",
            "CC_hat",
            "phi_hat_avg",
            "tau_hat_avg",
        ):
            out[k] = float("nan")

    # Population-level cross-world baselines (JA/JB from marginals always computable)
    JA_pop = abs(float(cfg.marginals.w1.pA) - float(cfg.marginals.w0.pA))
    JB_pop = abs(float(cfg.marginals.w1.pB) - float(cfg.marginals.w0.pB))
    Jbest_pop = max(JA_pop, JB_pop)
    out["JA_pop"] = float(JA_pop)
    out["JB_pop"] = float(JB_pop)
    out["Jbest_pop"] = float(Jbest_pop)

    pC0_true = float(out.get("pC_true_w0", float("nan")))
    pC1_true = float(out.get("pC_true_w1", float("nan")))
    dC_pop = pC1_true - pC0_true
    JC_pop = abs(dC_pop)
    CC_pop = (
        (JC_pop / Jbest_pop) if (Jbest_pop > 0.0 and math.isfinite(Jbest_pop)) else float("nan")
    )

    out["dC_pop"] = float(dC_pop)
    out["JC_pop"] = float(JC_pop)
    out["CC_pop"] = float(CC_pop)

    phi0_true = float(out.get("phi_true_w0", float("nan")))
    phi1_true = float(out.get("phi_true_w1", float("nan")))
    tau0_true = float(out.get("tau_true_w0", float("nan")))
    tau1_true = float(out.get("tau_true_w1", float("nan")))
    out["phi_pop_avg"] = _nan_robust_avg(phi0_true, phi1_true)
    out["tau_pop_avg"] = _nan_robust_avg(tau0_true, tau1_true)

    # Optional: theory reference overlays (separate, explicitly labeled)
    if cfg.include_theory_reference and callable(getattr(U, "compute_metrics_for_lambda", None)):
        try:
            theory = U.compute_metrics_for_lambda(
                cfg.marginals,
                cfg.rule,
                lam_f,
                path=cfg.path,
                path_params=cfg.path_params,
            )  # type: ignore[misc]
            out["CC_theory_ref"] = float(theory.get("CC", float("nan")))
            out["JC_theory_ref"] = float(theory.get("JC", float("nan")))
            out["dC_theory_ref"] = float(theory.get("dC", float("nan")))
            out["phi_theory_ref_avg"] = float(theory.get("phi_avg", float("nan")))
            out["tau_theory_ref_avg"] = float(theory.get("tau_avg", float("nan")))
            out["CC_ref_minus_pop"] = float(out["CC_theory_ref"] - out["CC_pop"])
            out["JC_ref_minus_pop"] = float(out["JC_theory_ref"] - out["JC_pop"])
        except Exception as e:
            out["theory_ref_error"] = f"{type(e).__name__}: {e}"

    # Envelope violation check on JC_hat (only meaningful if worlds valid)
    tol = float(cfg.envelope_tol)
    JC_hat = float(out.get("JC_hat", float("nan")))
    if (
        worlds_valid
        and math.isfinite(JC_hat)
        and math.isfinite(float(jmin))
        and math.isfinite(float(jmax))
    ):
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


# ----------------------------
# Grid simulation
# ----------------------------
def simulate_grid(cfg: SimConfig) -> pd.DataFrame:
    """
    Run simulation across all lambdas and replicates.

    stable_per_cell:
      - uses cfg.lambda_index_for_seed(lam) + rng_for_cell(...) => order-invariant
    sequential:
      - draws from a base RNG seeded by cfg.seed => order-dependent by design
    """
    validate_cfg(cfg)

    lambdas_list = [float(x) for x in cfg.lambdas]
    if len(lambdas_list) == 0:
        raise ValueError("cfg.lambdas must be non-empty.")

    for i, lam in enumerate(lambdas_list):
        if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
            raise ValueError(f"Invalid lambda at index {i}: got {lam!r}")

    base_rng = (
        np.random.default_rng(int(cfg.seed))
        if cfg.seed_policy == "sequential"
        else np.random.default_rng(0)
    )

    total = int(cfg.n_reps) * int(len(lambdas_list))
    rows: list[Dict[str, Any]] = []
    rows_append = rows.append

    for rep in range(int(cfg.n_reps)):
        for lam in lambdas_list:
            lam_index = cfg.lambda_index_for_seed(lam)  # canonical for stable_per_cell seeding
            try:
                row = simulate_replicate_at_lambda(
                    cfg,
                    lam=float(lam),
                    lam_index=int(lam_index),
                    rep=int(rep),
                    rng=base_rng,
                )
                # Ensure row-level fields exist even if older code paths forget them
                row.setdefault("worlds_valid", bool(row.get("world_valid_w0", False) and row.get("world_valid_w1", False)))
                row.setdefault("row_ok", bool(row.get("worlds_valid", False)))
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
                        "hard_fail_on_invalid": bool(cfg.hard_fail_on_invalid),
                        "worlds_valid": False,
                        "row_ok": False,
                        "row_error_stage": "simulate_replicate_at_lambda",
                        "row_error_msg": f"{type(e).__name__}: {e}",
                    }
                )

    df = pd.DataFrame.from_records(rows)
    sort_cols = [c for c in ("lambda", "rep") if c in df.columns]
    df = (
        df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        if sort_cols
        else df.reset_index(drop=True)
    )

    if len(df) != total and cfg.hard_fail_on_invalid:
        raise RuntimeError(f"simulate_grid produced {len(df)} rows, expected {total}")

    return df


# ----------------------------
# Summaries
# ----------------------------
def summarize_simulation(
    df_long: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """
    Aggregate replicate-level results to produce per-lambda summaries.

    Outputs are designed to be:
    - plot-ready (mean/std/quantiles),
    - audit-ready (ok-rate, invalid-joint rates, envelope violation rates),
    - reviewer-proof (NaN rates, population drift diagnostics).
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

    # round + dedupe with explicit error if user gave duplicates
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

    # Core replicate outputs (empirical)
    core_cols = (
        "CC_hat",
        "JC_hat",
        "dC_hat",
        "JA_hat",
        "JB_hat",
        "Jbest_hat",
        "phi_hat_avg",
        "tau_hat_avg",
    )

    # Population/theory columns that should be constant within a lambda group
    pop_cols = (
        "CC_pop",
        "JC_pop",
        "dC_pop",
        "phi_pop_avg",
        "tau_pop_avg",
        "JC_env_min",
        "JC_env_max",
        "JA_pop",
        "JB_pop",
        "Jbest_pop",
    )

    gb = df.groupby(group_keys, sort=True, dropna=False)
    rows: List[Dict[str, Any]] = []

    def _constancy_block(g: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Report population constancy and drift within a group."""
        if col not in g.columns:
            return {}
        s = pd.to_numeric(g[col], errors="coerce")
        nn = int(s.notna().sum())
        if nn == 0:
            return {
                f"{col}_value": float("nan"),
                f"{col}_n": 0,
                f"{col}_std": float("nan"),
                f"{col}_drift": False,
                f"{col}_unique": 0,
            }
        # Use std as drift proxy; also count unique rounded values
        std = float(s.std(ddof=0)) if nn >= 2 else 0.0
        # Unique after rounding to avoid float micro-noise
        uniq = int(pd.Series(np.round(s.dropna().to_numpy(dtype=float), 15)).nunique())
        drift = bool((std > 1e-15) or (uniq > 1))
        # “Value” as median (robust) since it should be constant anyway
        val = float(s.median())
        return {
            f"{col}_value": val,
            f"{col}_n": nn,
            f"{col}_std": std,
            f"{col}_drift": drift,
            f"{col}_unique": uniq,
        }

    for key, g in gb:
        row: Dict[str, Any] = {}

        # Materialize group identifiers
        if isinstance(key, tuple):
            for kname, kval in zip(group_keys, key):
                if kname == "lambda":
                    row[kname] = float(kval)
                else:
                    row[kname] = None if pd.isna(kval) else str(kval)
        else:
            row["lambda"] = float(key)

        row["n_rows"] = int(len(g))
        row["n_reps"] = (
            int(pd.to_numeric(g["rep"], errors="coerce").nunique(dropna=True))
            if "rep" in g.columns
            else int(len(g))
        )

        # Row validity
        if "row_ok" in g.columns:
            ok = g["row_ok"].fillna(False).astype(bool)
            row["row_ok_rate"] = float(ok.mean()) if len(ok) > 0 else float("nan")
            row["n_row_ok"] = int(ok.sum())
            row["n_row_fail"] = int((~ok).sum())
        else:
            row["row_ok_rate"] = float("nan")
            row["n_row_ok"] = int(len(g))
            row["n_row_fail"] = 0

        # Core stats
        for col in core_cols:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(s.mean())
                row[f"{col}_std"] = float(s.std(ddof=1)) if s.notna().sum() >= 2 else float("nan")
                row[f"{col}_n"] = int(s.notna().sum())
                row[f"{col}_nan_rate"] = float(s.isna().mean())

        # Quantiles for headline curves
        if "CC_hat" in g.columns:
            row.update(_qcols(g["CC_hat"], "CC_hat"))
        if "JC_hat" in g.columns:
            row.update(_qcols(g["JC_hat"], "JC_hat"))

        # Envelope violation summary
        if "JC_env_violation" in g.columns:
            v = g["JC_env_violation"].fillna(False).astype(bool)
            row["JC_env_violation_rate"] = float(v.mean()) if len(v) > 0 else float("nan")
            row["JC_env_violation_n"] = int(v.sum())
        else:
            row["JC_env_violation_rate"] = float("nan")
            row["JC_env_violation_n"] = 0

        # Invalid-joint flags emitted by simulate_replicate_at_lambda
        inv_cols = [c for c in g.columns if c.startswith("invalid_joint_w")]
        for c in inv_cols:
            v = g[c].fillna(False).astype(bool)
            row[f"{c}_rate"] = float(v.mean()) if len(v) > 0 else float("nan")
            row[f"{c}_n"] = int(v.sum())

        # Population constancy / drift diagnostics
        for col in pop_cols:
            row.update(_constancy_block(g, col))

        # Optional: theory ref overlays constancy
        for col in ("CC_theory_ref", "JC_theory_ref", "dC_theory_ref", "phi_theory_ref_avg", "tau_theory_ref_avg", "CC_ref_minus_pop", "JC_ref_minus_pop"):
            row.update(_constancy_block(g, col))

        # Optional: report most common failure stages (helps debugging experiment validity)
        if "row_error_stage" in g.columns:
            stages = g["row_error_stage"].fillna("").astype(str)
            # exclude empty
            bad = stages[stages != ""]
            if len(bad) > 0:
                top = bad.value_counts().head(3)
                for i, (stage, cnt) in enumerate(top.items(), start=1):
                    row[f"top_fail_stage_{i}"] = str(stage)
                    row[f"top_fail_stage_{i}_n"] = int(cnt)

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Sort by lambda (and rule/path if present)
    sort_cols = [c for c in ("lambda", "rule", "path") if c in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df_out = df_out.reset_index(drop=True)

    return df_out