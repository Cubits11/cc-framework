"""
Enterprise-scale emergent composition sweep (Gaussian copula).

Goal
----
Simulate *correlated* guardrail decisions for two guardrails A and B under a
latent Gaussian copula model, then sweep correlation (rho) and guardrail strength
(TPR_B) to observe how dependence drives emergent composition behavior.

Key properties
--------------
- Marginals: each guardrail's FPR and TPR are calibrated by construction using
  thresholding of N(0,1) (negative class) and N(mu,1) (positive class).
- Dependence: cross-guardrail dependence is induced via a *bivariate normal*
  latent variable with correlation rho, applied separately to negative and
  positive distributions (same rho by default). You can override rho_neg/rho_pos.
- Outputs: CSV grid with metrics + sanity diagnostics, plus heatmap figures.

Why this matters
----------------
This sweep is a controlled "physics engine" for guardrail composition:
it isolates the dependence structure and shows when "independence assumptions"
collapse, producing emergent deficits/surpluses in composed performance.

Run
---
python -m cc.analysis.emergent_sweep --self-check
python -m cc.analysis.emergent_sweep --n-samples 20000 --rho-steps 21 --tprb-steps 13

Notes
-----
- This module intentionally stays numpy + matplotlib only (matplotlib imported lazily).
- For heavier reporting (phase-space contours, stats tests), use a separate analysis module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from cc.core.metrics import cc_max, delta_add, youden_j

# -------------------------
# Data models
# -------------------------


@dataclass(frozen=True)
class GuardrailSpec:
    """Guardrail operating point in (TPR, FPR)."""

    tpr: float
    fpr: float


@dataclass(frozen=True)
class SweepConfig:
    """Sweep configuration."""

    n_samples: int
    rho_grid: np.ndarray
    tpr_b_grid: np.ndarray
    guardrail_a: GuardrailSpec
    guardrail_b_fpr: float
    comp: str
    seed: int
    replicates: int = 1
    jobs: int = 1
    executor: str = "process"  # "process" or "thread"
    rho_pos: float | None = None  # override rho for positive class only
    rho_neg: float | None = None  # override rho for negative class only


@dataclass(frozen=True)
class RunMetadata:
    """Reproducibility + provenance details for a sweep."""

    run_id: str
    created_utc: str
    python: str
    platform: str
    numpy: str
    argv: list[str]
    config: dict[str, Any]
    elapsed_sec: float


# -------------------------
# Validation helpers
# -------------------------


def _clip_prob(p: float, eps: float = 1e-9) -> float:
    return float(min(max(float(p), eps), 1.0 - eps))


def _validate_guardrail(spec: GuardrailSpec, name: str) -> None:
    tpr = float(spec.tpr)
    fpr = float(spec.fpr)
    if not (0.0 < tpr < 1.0):
        raise ValueError(f"{name}.tpr must be in (0,1). Got {tpr}")
    if not (0.0 < fpr < 1.0):
        raise ValueError(f"{name}.fpr must be in (0,1). Got {fpr}")


def _validate_comp(comp: str) -> str:
    comp_u = str(comp).upper().strip()
    if comp_u not in {"AND", "OR"}:
        raise ValueError(f"Unknown composition rule: {comp!r}. Expected AND or OR.")
    return comp_u


def _validate_rho(rho: float) -> float:
    rho = float(rho)
    if math.isnan(rho) or math.isinf(rho):
        raise ValueError(f"rho must be finite. Got {rho}")
    # Avoid numerical issues at exactly +/-1.
    return float(np.clip(rho, -0.999, 0.999))


def _validate_positive_int(x: int, name: str, min_value: int = 1) -> int:
    xi = int(x)
    if xi < min_value:
        raise ValueError(f"{name} must be >= {min_value}. Got {xi}")
    return xi


def _validate_executor(executor: str) -> str:
    ex = str(executor).lower().strip()
    if ex not in {"process", "thread"}:
        raise ValueError("--executor must be 'process' or 'thread'")
    return ex


# -------------------------
# Copula calibration
# -------------------------


def _threshold_from_fpr(fpr: float) -> float:
    """
    For negative class score ~ N(0,1), threshold t is chosen so:
        FPR = P(score >= t) = 1 - Phi(t)
    => t = Phi^{-1}(1 - FPR)
    """
    fpr = _clip_prob(fpr)
    return float(NormalDist().inv_cdf(1.0 - fpr))


def _mu_pos_from_tpr(tpr: float, threshold: float) -> float:
    """
    For positive class score ~ N(mu,1), threshold t is chosen so:
        TPR = P(score >= t) = 1 - Phi(t - mu)
    => mu = t - Phi^{-1}(1 - TPR)
    """
    tpr = _clip_prob(tpr)
    z = NormalDist().inv_cdf(1.0 - tpr)
    return float(threshold - z)


# -------------------------
# Simulation core
# -------------------------


def _bvn_scores(n: int, rho: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample (X, Y) ~ BVN(0,0,1,1,rho).
    Efficient construction:
        X = Z1
        Y = rho*Z1 + sqrt(1-rho^2)*Z2
    """
    rho = _validate_rho(rho)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    y = rho * z1 + math.sqrt(1.0 - rho * rho) * z2
    return z1, y


def _simulate_pair_decisions(
    n: int,
    thr_a: float,
    thr_b: float,
    mu_a: float,
    mu_b: float,
    rho_neg: float,
    rho_pos: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Simulate correlated scores for A and B for negative and positive classes.

    Negative:
        (S_a0, S_b0) ~ BVN(0,0,1,1,rho_neg)
    Positive:
        (S_a1, S_b1) ~ BVN(mu_a,mu_b,1,1,rho_pos)

    Decisions are thresholds on these latent scores.
    """
    a0, b0 = _bvn_scores(n, rho_neg, rng)
    a1, b1 = _bvn_scores(n, rho_pos, rng)
    a1 = a1 + mu_a
    b1 = b1 + mu_b

    return {
        "a_neg_score": a0,
        "b_neg_score": b0,
        "a_pos_score": a1,
        "b_pos_score": b1,
        "a_neg": (a0 >= thr_a),
        "b_neg": (b0 >= thr_b),
        "a_pos": (a1 >= thr_a),
        "b_pos": (b1 >= thr_b),
    }


def _compose(a: np.ndarray, b: np.ndarray, comp: str) -> np.ndarray:
    comp_u = _validate_comp(comp)
    if comp_u == "AND":
        return np.logical_and(a, b)
    return np.logical_or(a, b)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns 0 if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = float(x.std())
    sy = float(y.std())
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _measure_one(
    a: GuardrailSpec,
    b: GuardrailSpec,
    rho: float,
    n_samples: int,
    comp: str,
    rng: np.random.Generator,
    rho_pos_override: float | None = None,
    rho_neg_override: float | None = None,
) -> dict[str, Any]:
    """
    Measure composed metrics for one configuration.
    Returns a dict of JSON/CSV-safe scalars.
    """
    _validate_guardrail(a, "guardrail_a")
    _validate_guardrail(b, "guardrail_b")
    comp_u = _validate_comp(comp)

    # Calibrate thresholds / shifts
    thr_a = _threshold_from_fpr(a.fpr)
    thr_b = _threshold_from_fpr(b.fpr)
    mu_a = _mu_pos_from_tpr(a.tpr, thr_a)
    mu_b = _mu_pos_from_tpr(b.tpr, thr_b)

    rho = _validate_rho(rho)
    rho_neg = _validate_rho(rho_neg_override if rho_neg_override is not None else rho)
    rho_pos = _validate_rho(rho_pos_override if rho_pos_override is not None else rho)

    sim = _simulate_pair_decisions(
        n=int(n_samples),
        thr_a=thr_a,
        thr_b=thr_b,
        mu_a=mu_a,
        mu_b=mu_b,
        rho_neg=rho_neg,
        rho_pos=rho_pos,
        rng=rng,
    )

    a_neg = sim["a_neg"]
    b_neg = sim["b_neg"]
    a_pos = sim["a_pos"]
    b_pos = sim["b_pos"]

    comp_neg = _compose(a_neg, b_neg, comp_u)
    comp_pos = _compose(a_pos, b_pos, comp_u)

    # Empirical rates
    tpr_a = float(np.mean(a_pos))
    fpr_a = float(np.mean(a_neg))
    tpr_b = float(np.mean(b_pos))
    fpr_b = float(np.mean(b_neg))
    tpr_c = float(np.mean(comp_pos))
    fpr_c = float(np.mean(comp_neg))

    # Core metrics (framework)
    j_a = float(youden_j(tpr_a, fpr_a))
    j_b = float(youden_j(tpr_b, fpr_b))
    j_comp = float(youden_j(tpr_c, fpr_c))
    cc = float(cc_max(j_comp, j_a, j_b))
    d_add = float(delta_add(j_comp, j_a, j_b))

    # Diagnostics: did we actually induce dependence?
    rho_score_neg_hat = _corr(sim["a_neg_score"], sim["b_neg_score"])
    rho_score_pos_hat = _corr(sim["a_pos_score"], sim["b_pos_score"])
    rho_dec_neg_hat = _corr(a_neg.astype(float), b_neg.astype(float))
    rho_dec_pos_hat = _corr(a_pos.astype(float), b_pos.astype(float))

    return {
        # requested inputs
        "rho": float(rho),
        "rho_neg": float(rho_neg),
        "rho_pos": float(rho_pos),
        "tpr_a_target": float(a.tpr),
        "fpr_a_target": float(a.fpr),
        "tpr_b_target": float(b.tpr),
        "fpr_b_target": float(b.fpr),
        "comp": comp_u,
        "n_samples": int(n_samples),
        # calibrated latent params
        "thr_a": float(thr_a),
        "thr_b": float(thr_b),
        "mu_a": float(mu_a),
        "mu_b": float(mu_b),
        # empirical marginals
        "tpr_a": float(tpr_a),
        "fpr_a": float(fpr_a),
        "tpr_b": float(tpr_b),
        "fpr_b": float(fpr_b),
        "tpr_comp": float(tpr_c),
        "fpr_comp": float(fpr_c),
        # framework metrics
        "j_a": float(j_a),
        "j_b": float(j_b),
        "j_comp": float(j_comp),
        "cc_max": float(cc),
        "delta_add": float(d_add),
        # dependence sanity
        "rho_score_neg_hat": float(rho_score_neg_hat),
        "rho_score_pos_hat": float(rho_score_pos_hat),
        "rho_dec_neg_hat": float(rho_dec_neg_hat),
        "rho_dec_pos_hat": float(rho_dec_pos_hat),
    }


# -------------------------
# Replicate aggregation
# -------------------------


def _aggregate_replicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate replicate rows into mean + std columns.
    Keeps deterministic identifiers from the first replicate.
    """
    if not rows:
        raise ValueError("No replicate rows provided to aggregate.")

    base = rows[0]
    out: dict[str, Any] = {}

    # Deterministic identifiers (should be identical across replicates)
    keep = [
        "rho",
        "rho_neg",
        "rho_pos",
        "tpr_a_target",
        "fpr_a_target",
        "tpr_b_target",
        "fpr_b_target",
        "comp",
        "n_samples",
        "thr_a",
        "thr_b",
        "mu_a",
        "mu_b",
    ]
    for k in keep:
        if k in base:
            out[k] = base[k]

    # Numeric metrics to aggregate (can vary slightly across replicates)
    agg = [
        "tpr_a",
        "fpr_a",
        "tpr_b",
        "fpr_b",
        "tpr_comp",
        "fpr_comp",
        "j_a",
        "j_b",
        "j_comp",
        "cc_max",
        "delta_add",
        "rho_score_neg_hat",
        "rho_score_pos_hat",
        "rho_dec_neg_hat",
        "rho_dec_pos_hat",
    ]
    for k in agg:
        vals = np.array([float(r[k]) for r in rows], dtype=float)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    out["replicates"] = len(rows)
    return out


# -------------------------
# Parallel-safe worker (must be top-level for ProcessPool pickling)
# -------------------------


def _run_one_cell_worker(
    i: int,
    j: int,
    rho: float,
    tpr_b: float,
    *,
    n_samples: int,
    comp: str,
    tpr_a: float,
    fpr_a: float,
    fpr_b: float,
    seed: int,
    replicates: int,
    rho_pos: float | None,
    rho_neg: float | None,
) -> dict[str, Any]:
    a = GuardrailSpec(tpr=float(tpr_a), fpr=float(fpr_a))
    b = GuardrailSpec(tpr=float(tpr_b), fpr=float(fpr_b))

    reps_n = max(1, int(replicates))
    reps: list[dict[str, Any]] = []
    for r in range(reps_n):
        # Deterministic per-cell, per-replicate RNG (stable across process/thread execution)
        ss = np.random.SeedSequence([int(seed), int(i), int(j), int(r)])
        rng = np.random.default_rng(ss)
        reps.append(
            _measure_one(
                a=a,
                b=b,
                rho=float(rho),
                n_samples=int(n_samples),
                comp=str(comp),
                rng=rng,
                rho_pos_override=rho_pos,
                rho_neg_override=rho_neg,
            )
        )
    return _aggregate_replicates(reps)


# -------------------------
# Sweep runner
# -------------------------


def run_sweep(cfg: SweepConfig) -> list[dict[str, Any]]:
    """
    Run the sweep over rho_grid x tpr_b_grid with deterministic per-cell seeding.
    Supports process-parallel execution on macOS by using a top-level worker.
    """
    _validate_guardrail(cfg.guardrail_a, "guardrail_a")
    comp_u = _validate_comp(cfg.comp)
    jobs = _validate_positive_int(cfg.jobs, "jobs", 1)
    reps = _validate_positive_int(cfg.replicates, "replicates", 1)
    ex = _validate_executor(cfg.executor)

    rho_vals = [float(x) for x in np.asarray(cfg.rho_grid, dtype=float).tolist()]
    tpr_vals = [float(x) for x in np.asarray(cfg.tpr_b_grid, dtype=float).tolist()]

    # Single-process path (also used as fallback)
    if jobs <= 1:
        rows: list[dict[str, Any]] = []
        for i, rho in enumerate(rho_vals):
            for j, tpr_b in enumerate(tpr_vals):
                rows.append(
                    _run_one_cell_worker(
                        i,
                        j,
                        rho,
                        tpr_b,
                        n_samples=cfg.n_samples,
                        comp=comp_u,
                        tpr_a=cfg.guardrail_a.tpr,
                        fpr_a=cfg.guardrail_a.fpr,
                        fpr_b=cfg.guardrail_b_fpr,
                        seed=cfg.seed,
                        replicates=reps,
                        rho_pos=cfg.rho_pos,
                        rho_neg=cfg.rho_neg,
                    )
                )
        rows.sort(key=lambda r: (float(r["rho"]), float(r["tpr_b_target"])))
        return rows

    # Parallel path
    if ex == "thread":
        from concurrent.futures import ThreadPoolExecutor as Executor
        from concurrent.futures import as_completed
    else:
        from concurrent.futures import ProcessPoolExecutor as Executor
        from concurrent.futures import as_completed

    tasks: list[tuple[int, int, float, float]] = []
    for i, rho in enumerate(rho_vals):
        for j, tpr_b in enumerate(tpr_vals):
            tasks.append((i, j, rho, tpr_b))

    rows_out: list[dict[str, Any]] = []
    with Executor(max_workers=jobs) as pool:
        futs = [
            pool.submit(
                _run_one_cell_worker,
                i,
                j,
                rho,
                tpr_b,
                n_samples=cfg.n_samples,
                comp=comp_u,
                tpr_a=cfg.guardrail_a.tpr,
                fpr_a=cfg.guardrail_a.fpr,
                fpr_b=cfg.guardrail_b_fpr,
                seed=cfg.seed,
                replicates=reps,
                rho_pos=cfg.rho_pos,
                rho_neg=cfg.rho_neg,
            )
            for (i, j, rho, tpr_b) in tasks
        ]
        for fut in as_completed(futs):
            rows_out.append(fut.result())

    rows_out.sort(key=lambda r: (float(r["rho"]), float(r["tpr_b_target"])))
    return rows_out


# -------------------------
# Output helpers
# -------------------------


def _coerce_csv_value(v: Any) -> Any:
    """Convert numpy scalars / odd types into CSV-friendly python types."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows_l = list(rows)
    if not rows_l:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "rho",
        "rho_neg",
        "rho_pos",
        "tpr_b_target",
        "fpr_b_target",
        "tpr_a_target",
        "fpr_a_target",
        "comp",
        "n_samples",
        "replicates",
        "thr_a",
        "thr_b",
        "mu_a",
        "mu_b",
        "tpr_a",
        "tpr_a_std",
        "fpr_a",
        "fpr_a_std",
        "tpr_b",
        "tpr_b_std",
        "fpr_b",
        "fpr_b_std",
        "tpr_comp",
        "tpr_comp_std",
        "fpr_comp",
        "fpr_comp_std",
        "j_a",
        "j_a_std",
        "j_b",
        "j_b_std",
        "j_comp",
        "j_comp_std",
        "cc_max",
        "cc_max_std",
        "delta_add",
        "delta_add_std",
        "rho_score_neg_hat",
        "rho_score_neg_hat_std",
        "rho_score_pos_hat",
        "rho_score_pos_hat_std",
        "rho_dec_neg_hat",
        "rho_dec_neg_hat_std",
        "rho_dec_pos_hat",
        "rho_dec_pos_hat_std",
    ]

    all_keys = list(rows_l[0].keys())
    extras = [k for k in all_keys if k not in preferred]
    fieldnames = [k for k in preferred if k in all_keys] + sorted(extras)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_l:
            w.writerow({k: _coerce_csv_value(r.get(k)) for k in fieldnames})


def _plot_heatmap(
    rows: list[dict[str, Any]],
    out_path: Path,
    value_key: str,
    title: str,
    xlabel: str = "correlation (rho)",
    ylabel: str = "TPR_B",
) -> None:
    import matplotlib.pyplot as plt

    rhos = sorted({float(r["rho"]) for r in rows})
    tprs = sorted({float(r["tpr_b_target"]) for r in rows})
    rho_idx = {v: i for i, v in enumerate(rhos)}
    tpr_idx = {v: i for i, v in enumerate(tprs)}

    grid = np.full((len(tprs), len(rhos)), np.nan, dtype=float)
    for r in rows:
        i = tpr_idx[float(r["tpr_b_target"])]
        j = rho_idx[float(r["rho"])]
        grid[i, j] = float(r[value_key])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="cividis",
        extent=[min(rhos), max(rhos), min(tprs), max(tprs)],
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _hash_config_for_run_id(cfg: SweepConfig, argv: Sequence[str]) -> str:
    # Keep it stable + human-usable; include argv (since it encodes CLI intent)
    h = hashlib.blake2b(digest_size=10)
    h.update(json.dumps(argv, sort_keys=False).encode("utf-8"))
    # For ndarray fields, hash a compact representation
    h.update(np.asarray(cfg.rho_grid, dtype=float).tobytes())
    h.update(np.asarray(cfg.tpr_b_grid, dtype=float).tobytes())
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert objects into JSON-serializable Python types."""
    # numpy
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # pathlib
    if isinstance(obj, Path):
        return str(obj)

    # containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # fall-through
    return obj


def _save_metadata(meta: RunMetadata, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(asdict(meta))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# -------------------------
# Self-checks (rigorous sanity)
# -------------------------


def _approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def run_self_checks() -> None:
    """
    Fast, deterministic checks that validate:
    1) Marginals match target TPR/FPR (within tolerance).
    2) rho=0 reproduces independence for AND/OR (within tolerance).
    3) Dependence changes composed rates in correct direction.
    4) Induced latent-score correlation tracks the requested rho.
    """
    seed = 123
    n = 150_000  # large enough to be meaningful but still tractable
    a = GuardrailSpec(tpr=0.82, fpr=0.08)
    b = GuardrailSpec(tpr=0.70, fpr=0.08)

    def measure(rho: float, comp: str) -> dict[str, Any]:
        # deterministic but distinct across (rho, comp)
        bump = round((rho + 1.0) * 10_000)
        bump += 0 if comp.upper() == "AND" else 50_000
        rng = np.random.default_rng(seed + bump)
        return _measure_one(a=a, b=b, rho=rho, n_samples=n, comp=comp, rng=rng)

    # 1) Marginals
    m0 = measure(0.0, "AND")
    # tolerance for proportions with n=150k: strict but realistic
    if not _approx(float(m0["tpr_a"]), a.tpr, 0.004):
        raise AssertionError(f"TPR_A mismatch: got {m0['tpr_a']:.4f} vs target {a.tpr:.4f}")
    if not _approx(float(m0["fpr_a"]), a.fpr, 0.0025):
        raise AssertionError(f"FPR_A mismatch: got {m0['fpr_a']:.4f} vs target {a.fpr:.4f}")
    if not _approx(float(m0["tpr_b"]), b.tpr, 0.004):
        raise AssertionError(f"TPR_B mismatch: got {m0['tpr_b']:.4f} vs target {b.tpr:.4f}")
    if not _approx(float(m0["fpr_b"]), b.fpr, 0.0025):
        raise AssertionError(f"FPR_B mismatch: got {m0['fpr_b']:.4f} vs target {b.fpr:.4f}")

    # 2) Independence baseline at rho=0
    # AND: p = pA*pB
    tpr_and_theory = a.tpr * b.tpr
    fpr_and_theory = a.fpr * b.fpr
    if not _approx(float(m0["tpr_comp"]), tpr_and_theory, 0.006):
        raise AssertionError(
            f"rho=0 AND TPR_comp mismatch: got {m0['tpr_comp']:.4f} vs {tpr_and_theory:.4f}"
        )
    if not _approx(float(m0["fpr_comp"]), fpr_and_theory, 0.002):
        raise AssertionError(
            f"rho=0 AND FPR_comp mismatch: got {m0['fpr_comp']:.6f} vs {fpr_and_theory:.6f}"
        )

    # OR: p = pA + pB - pA*pB
    m0_or = measure(0.0, "OR")
    tpr_or_theory = a.tpr + b.tpr - (a.tpr * b.tpr)
    fpr_or_theory = a.fpr + b.fpr - (a.fpr * b.fpr)
    if not _approx(float(m0_or["tpr_comp"]), tpr_or_theory, 0.006):
        raise AssertionError(
            f"rho=0 OR TPR_comp mismatch: got {m0_or['tpr_comp']:.4f} vs {tpr_or_theory:.4f}"
        )
    if not _approx(float(m0_or["fpr_comp"]), fpr_or_theory, 0.0025):
        raise AssertionError(
            f"rho=0 OR FPR_comp mismatch: got {m0_or['fpr_comp']:.4f} vs {fpr_or_theory:.4f}"
        )

    # 3) Directional dependence effects
    m_pos = measure(+0.6, "AND")
    m_neg = measure(-0.6, "AND")
    # AND: positive correlation increases intersection rates; negative decreases
    if not (float(m_pos["fpr_comp"]) > float(m0["fpr_comp"])):
        raise AssertionError("AND: expected +rho to increase FPR_comp vs rho=0")
    if not (float(m_neg["fpr_comp"]) < float(m0["fpr_comp"])):
        raise AssertionError("AND: expected -rho to decrease FPR_comp vs rho=0")

    # OR: positive correlation decreases union rates (more overlap); negative increases
    m_pos_or = measure(+0.6, "OR")
    m_neg_or = measure(-0.6, "OR")
    if not (float(m_pos_or["tpr_comp"]) < float(m0_or["tpr_comp"])):
        raise AssertionError("OR: expected +rho to decrease TPR_comp vs rho=0")
    if not (float(m_neg_or["tpr_comp"]) > float(m0_or["tpr_comp"])):
        raise AssertionError("OR: expected -rho to increase TPR_comp vs rho=0")

    # 4) Induced dependence sanity (scores should reflect rho strongly)
    if not (float(m_pos["rho_score_neg_hat"]) > 0.45):
        raise AssertionError("Expected rho_score_neg_hat to be strongly positive at rho=+0.6")
    if not (float(m_neg["rho_score_neg_hat"]) < -0.45):
        raise AssertionError("Expected rho_score_neg_hat to be strongly negative at rho=-0.6")

    print("✓ self-checks passed (marginals, independence, dependence direction, rho diagnostics).")


# -------------------------
# CLI
# -------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run emergent composition sweep (Gaussian copula).")

    p.add_argument("--n-samples", type=int, default=40_000)
    p.add_argument("--rho-min", type=float, default=-0.6)
    p.add_argument("--rho-max", type=float, default=0.6)
    p.add_argument("--rho-steps", type=int, default=25)
    p.add_argument("--tprb-min", type=float, default=0.6)
    p.add_argument("--tprb-max", type=float, default=0.9)
    p.add_argument("--tprb-steps", type=int, default=16)

    p.add_argument("--tpr-a", type=float, default=0.82)
    p.add_argument("--fpr-a", type=float, default=0.08)
    p.add_argument("--fpr-b", type=float, default=0.08)

    p.add_argument("--comp", type=str, default="AND", choices=["AND", "OR"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--replicates", type=int, default=1, help="Repeat each cell and aggregate mean/std."
    )
    p.add_argument(
        "--jobs", type=int, default=1, help="Parallel workers. Use >1 for faster sweeps."
    )
    p.add_argument(
        "--executor",
        type=str,
        default="process",
        choices=["process", "thread"],
        help="Parallel backend. 'process' is fastest; 'thread' avoids pickling issues.",
    )

    # advanced: allow different dependence in pos vs neg classes
    p.add_argument(
        "--rho-neg", type=float, default=None, help="Override rho for negative class only."
    )
    p.add_argument(
        "--rho-pos", type=float, default=None, help="Override rho for positive class only."
    )

    p.add_argument("--out-csv", type=Path, default=Path("results/emergent/sweep.csv"))
    p.add_argument(
        "--out-meta",
        type=Path,
        default=Path("results/emergent/run_meta.json"),
        help="Write run metadata JSON for reproducibility.",
    )
    p.add_argument("--out-fig", type=Path, default=Path("figures/emergent/cc_max_heatmap.png"))
    p.add_argument(
        "--out-fig-delta", type=Path, default=Path("figures/emergent/delta_add_heatmap.png")
    )
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.add_argument("--self-check", action="store_true", help="Run rigorous sanity checks and exit.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.self_check:
        run_self_checks()
        return

    # Build grids
    rho_grid = np.linspace(
        float(args.rho_min),
        float(args.rho_max),
        _validate_positive_int(args.rho_steps, "rho_steps", 1),
    )
    tpr_b_grid = np.linspace(
        float(args.tprb_min),
        float(args.tprb_max),
        _validate_positive_int(args.tprb_steps, "tprb_steps", 1),
    )

    cfg = SweepConfig(
        n_samples=_validate_positive_int(args.n_samples, "n_samples", 1),
        rho_grid=rho_grid,
        tpr_b_grid=tpr_b_grid,
        guardrail_a=GuardrailSpec(tpr=float(args.tpr_a), fpr=float(args.fpr_a)),
        guardrail_b_fpr=float(args.fpr_b),
        comp=str(args.comp),
        seed=int(args.seed),
        replicates=_validate_positive_int(args.replicates, "replicates", 1),
        jobs=_validate_positive_int(args.jobs, "jobs", 1),
        executor=_validate_executor(args.executor),
        rho_neg=args.rho_neg,
        rho_pos=args.rho_pos,
    )

    _validate_guardrail(cfg.guardrail_a, "guardrail_a")
    _clip_prob(cfg.guardrail_b_fpr)  # just ensure it's sane

    # Run sweep
    t0 = time.time()
    rows = run_sweep(cfg)
    elapsed = time.time() - t0

    # Write outputs
    _write_csv(rows, args.out_csv)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")

    run_id = _hash_config_for_run_id(cfg, sys.argv)
    meta = RunMetadata(
        run_id=run_id,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python=sys.version.split()[0],
        platform=f"{platform.system()}-{platform.release()} ({platform.machine()})",
        numpy=getattr(np, "__version__", "unknown"),
        argv=list(sys.argv),
        config=_to_jsonable(asdict(cfg)),
        elapsed_sec=float(elapsed),
    )
    _save_metadata(meta, args.out_meta)
    print(f"Wrote metadata to {args.out_meta} (run_id={run_id})")

    if not args.no_plots:
        _plot_heatmap(
            rows,
            args.out_fig,
            value_key="cc_max",
            title="Emergent composition sweep: CC_max",
        )
        print(f"Wrote figure to {args.out_fig}")

        _plot_heatmap(
            rows,
            args.out_fig_delta,
            value_key="delta_add",
            title="Emergent composition sweep: delta_add",
        )
        print(f"Wrote figure to {args.out_fig_delta}")


if __name__ == "__main__":
    main()
