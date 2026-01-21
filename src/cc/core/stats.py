# src/cc/core/stats.py
"""
Statistical methods for CC framework with proper two-world handling.
Implements bootstrap CIs, analytical bounds, and composability metrics.

Author: Pranav Bhave
Date: 2026-01-01 (Enterprise upgrade)
Institution: Penn State University

================================================================================
STATISTICAL CONTRACT (Enterprise / Audit-Safe)
================================================================================

This module is the CC framework's "truth engine": it defines *what* is being
measured, *how* uncertainty is quantified, and *how* failures are surfaced.

Core definitions (hard contract):

- Worlds:
  - World 0 (baseline, no protection): success rate p0 = E[success | world=0] ∈ [0,1]
  - World 1 (protected):               success rate p1 = E[success | world=1] ∈ [0,1]

- Success:
  - Binary outcome: success ∈ {0,1}. (We coerce bool/int; anything else raises.)

- J-statistic (Youden-style difference, repurposed):
  - J = p0 - p1 ∈ [-1, 1]
  - J > 0  => protection reduces success rate (good)
  - J = 0  => no effect
  - J < 0  => protection increases success rate (regression / harmful)

  IMPORTANT: Negative J is valid and MUST NOT be clamped away.

- Composability Coefficients (CC):
  Let J_comp be the composed system's J, and {J_i} be individual guardrails' J.
  Define:
    J_max = max_i J_i
    Delta_add = J_comp - J_max

    CC_max is only *meaningful* if J_max > 0. If J_max <= 0, the ratio is
    undefined as a synergy metric (there is no "effective baseline guardrail").
    We return cc_max = NaN and set evidence flags.

  Heuristic (NOT a theorem) composition "bounds":
    - j_theoretical_max / j_theoretical_add are retained for backward compatibility,
      but are explicitly heuristic. They require assumptions that are not proven
      in general in this repo. Use for exploratory analysis only.

Inference contract (audit-safe):

- Bootstrap confidence intervals:
  - Stratified by world by default (preserve world counts).
  - Resampling modes:
      * iid     : within-world iid bootstrap
      * block   : within-world moving block bootstrap (ONLY if data is ordered)
      * cluster : within-world cluster bootstrap (ONLY if clusters are provided)
      * pair    : paired bootstrap if a pairing key is provided and pairs are valid
  - CI methods:
      * percentile
      * basic
      * bca (full BCa: bias-correction + jackknife acceleration on the chosen unit)

- Failure semantics:
  - No silent drops.
  - We track:
      n_valid, n_failed, failure_rate, failure_reasons
  - If valid replicates < min_valid_frac * B (default 0.90),
    CIs are returned as (NaN, NaN) and result.valid=False.

- Dependence handling:
  - Preferred: cluster bootstrap (resample cluster units) when cluster labels exist.
  - Block bootstrap only if ordering is defensible (ordered=True or monotone timestamp).
  - Optional ICC diagnostics are computed per world when clusters exist.
    By default we DO NOT apply additional ICC widening on top of cluster bootstrap
    (double-conservative). ICC is reported for audit/diagnostics.

Hypothesis testing contract:

- We provide *defensible* classical tests for difference in proportions (unpaired):
  - Fisher exact test (2x2)
  - Two-proportion z-test (asymptotic)
- For paired data (pair_key with complete pairs):
  - McNemar exact test
- Optional permutation p-values can be computed (n_permutations > 0), but are
  only valid under exchangeability assumptions (paired swap is safest).

Provenance contract:

- We emit stable hashes:
  - input_hash: sha256 over an order-insensitive multiset fingerprint + config
  - order_hash: sha256 over an order-sensitive fingerprint (important for blocks)
  - The hash is robust to extra/unhashable fields on AttackResult: we only use a
    fixed allowlist + string-coercion fallback.

================================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

# Type aliases for clarity
AttackResults = list["AttackResult"]
ConfidenceInterval = tuple[float, float]


# =============================================================================
# Public result types
# =============================================================================


@dataclass
class JBootstrapCI:
    """Minimal return type for the runner's import (used in analysis + logging)."""

    ci_lower: float
    ci_upper: float
    method: str = "percentile"
    valid: bool = True
    n_valid: int = 0
    n_failed: int = 0
    failure_rate: float = 0.0


@dataclass
class BootstrapResult:
    """Comprehensive results of bootstrap analysis with CC metrics (audit-grade)."""

    # Primary statistics
    j_statistic: float
    p0: float  # Success rate in world 0
    p1: float  # Success rate in world 1

    # Composability metrics
    cc_max: float
    delta_add: float
    cc_multiplicative: float | None = None

    # Confidence intervals
    ci_j: ConfidenceInterval = (math.nan, math.nan)
    ci_cc_max: ConfidenceInterval = (math.nan, math.nan)
    ci_delta_add: ConfidenceInterval = (math.nan, math.nan)
    ci_width: float = math.nan

    # Bootstrap diagnostics
    bootstrap_samples: np.ndarray | None = None  # J bootstrap samples (optional)
    n_sessions: int = 0
    n_bootstrap: int = 0
    convergence_diagnostic: float = math.nan  # CV of bootstrap statistic
    effective_sample_size: float = math.nan  # global ESS diagnostic

    # NEW: audit-safe bootstrap accounting
    n_valid: int = 0
    n_failed: int = 0
    failure_rate: float = 0.0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    valid: bool = True

    # NEW: dependence diagnostics (if applicable)
    resample: str = "iid"  # iid|block|cluster|pair
    ordered: bool | None = None
    block_size: int | None = None
    cluster_key: str | None = None
    n_clusters_world0: int | None = None
    n_clusters_world1: int | None = None
    icc_world0: float | None = None
    icc_world1: float | None = None
    ess_world0: float | None = None
    ess_world1: float | None = None

    # Metadata
    method: str = "percentile"  # percentile|basic|bca
    tier: str = "Tier 0"  # Tier 0 / Tier 1 / Tier 2 (policy label)
    input_hash: str = ""
    order_hash: str = ""

    # Statistical tests
    hypothesis_tests: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (backward-compatible + extended)."""
        return {
            "j_statistic": float(self.j_statistic),
            "p0": float(self.p0),
            "p1": float(self.p1),
            "cc_max": float(self.cc_max) if np.isfinite(self.cc_max) else None,
            "delta_add": float(self.delta_add),
            "cc_multiplicative": float(self.cc_multiplicative)
            if self.cc_multiplicative is not None
            else None,
            "confidence_intervals": {
                "j": {"lower": float(self.ci_j[0]), "upper": float(self.ci_j[1])},
                "cc_max": {"lower": float(self.ci_cc_max[0]), "upper": float(self.ci_cc_max[1])},
                "delta_add": {
                    "lower": float(self.ci_delta_add[0]),
                    "upper": float(self.ci_delta_add[1]),
                },
            },
            "diagnostics": {
                "n_sessions": int(self.n_sessions),
                "n_bootstrap": int(self.n_bootstrap),
                "convergence": float(self.convergence_diagnostic),
                "effective_sample_size": float(self.effective_sample_size),
                "n_valid": int(self.n_valid),
                "n_failed": int(self.n_failed),
                "failure_rate": float(self.failure_rate),
                "failure_reasons": dict(self.failure_reasons),
                "valid": bool(self.valid),
                "resample": self.resample,
                "ordered": self.ordered,
                "block_size": self.block_size,
                "cluster_key": self.cluster_key,
                "n_clusters_world0": self.n_clusters_world0,
                "n_clusters_world1": self.n_clusters_world1,
                "icc_world0": self.icc_world0,
                "icc_world1": self.icc_world1,
                "ess_world0": self.ess_world0,
                "ess_world1": self.ess_world1,
            },
            "hypothesis_tests": self.hypothesis_tests,
            "method": self.method,
            "tier": self.tier,
            "input_hash": self.input_hash,
            "order_hash": self.order_hash,
        }


@dataclass
class CCInterpretation:
    """Interpretation of CC results"""

    regime: str  # "constructive", "independent", "destructive"
    confidence: float
    recommendation: str
    evidence: dict[str, Any]


# =============================================================================
# Public API (kept compatible)
# =============================================================================


def bootstrap_ci_j_statistic(
    w0: np.ndarray,
    w1: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    random_state: int = 1337,
) -> JBootstrapCI:
    """
    Percentile bootstrap CI for J = mean(w0) - mean(w1),
    where w0, w1 are 0/1 arrays of outcomes (world 0 / world 1).

    Enterprise changes vs legacy:
      - Tracks failures and returns validity flags.
      - Never clamps J.

    Args:
        w0, w1: binary outcomes (0/1 or bool)
        B: bootstrap replicates
        alpha: CI level
        random_state: seed

    Returns:
        JBootstrapCI
    """
    w0 = np.asarray(w0).ravel()
    w1 = np.asarray(w1).ravel()
    if w0.size == 0 or w1.size == 0:
        raise ValueError("bootstrap_ci_j_statistic: both worlds must be non-empty")

    y0 = _coerce_binary_array(w0, name="w0")
    y1 = _coerce_binary_array(w1, name="w1")

    n0, n1 = y0.size, y1.size
    rng = np.random.default_rng(np.random.SeedSequence(random_state))

    boot = np.empty(B, dtype=float)
    n_failed = 0
    reasons: dict[str, int] = {}

    for b in range(B):
        try:
            s0 = y0[rng.integers(0, n0, size=n0)]
            s1 = y1[rng.integers(0, n1, size=n1)]
            boot[b] = float(s0.mean() - s1.mean())
        except Exception as e:
            boot[b] = np.nan
            n_failed += 1
            k = type(e).__name__
            reasons[k] = reasons.get(k, 0) + 1

    boot = boot[np.isfinite(boot)]
    n_valid = int(boot.size)
    failure_rate = float(n_failed / max(B, 1))
    valid = n_valid >= int(0.90 * B)

    if not valid:
        warnings.warn(
            f"bootstrap_ci_j_statistic: insufficient valid replicates "
            f"(valid={n_valid}/{B}, failure_rate={failure_rate:.2%}). Returning NaN CI.",
            stacklevel=2,
        )
        return JBootstrapCI(
            ci_lower=math.nan,
            ci_upper=math.nan,
            method="percentile",
            valid=False,
            n_valid=n_valid,
            n_failed=n_failed,
            failure_rate=failure_rate,
        )

    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return JBootstrapCI(
        ci_lower=lo,
        ci_upper=hi,
        method="percentile",
        valid=True,
        n_valid=n_valid,
        n_failed=n_failed,
        failure_rate=failure_rate,
    )


def compute_j_statistic(results: AttackResults) -> tuple[float, float, float]:
    """
    Compute J-statistic (p0 - p1) from two-world AttackResult list.

    Requirements:
      - Each result must have .world_bit in {0,1}
      - Each result must have .success in {0,1} or bool

    Returns:
      (J, p0, p1) with J ∈ [-1,1].
    """
    if not results:
        raise ValueError("compute_j_statistic: Empty results list")

    world = np.array([_get_world_bit(r) for r in results], dtype=int)
    y = np.array([_get_success(r) for r in results], dtype=int)

    mask0 = world == 0
    mask1 = world == 1
    n0 = int(mask0.sum())
    n1 = int(mask1.sum())
    if n0 == 0 or n1 == 0:
        raise ValueError(f"compute_j_statistic: Need both worlds. Got n0={n0}, n1={n1}.")

    p0 = float(y[mask0].mean())
    p1 = float(y[mask1].mean())
    j = float(p0 - p1)
    return j, p0, p1


def compute_composability_coefficients(
    j_composed: float, j_individual: dict[str, float], p0: float
) -> dict[str, float]:
    """
    Compute composability coefficient variants.

    Enterprise semantics:
      - cc_max is NaN if max individual J <= 0 (ratio not meaningful).
      - "theoretical" keys are maintained for backward compatibility but are
        explicitly HEURISTIC (not proven general bounds).

    Returns dict with stable keys:
      cc_max, delta_add, cc_multiplicative, j_composed, j_max_individual,
      j_theoretical_max, j_theoretical_add, efficiency_ratio
    """
    if not j_individual:
        # Degenerate: no individuals, interpret composed as baseline.
        return {
            "cc_max": float("nan"),
            "delta_add": 0.0,
            "cc_multiplicative": None,
            "j_composed": float(j_composed),
            "j_max_individual": float("nan"),
            "j_theoretical_max": float(j_composed),  # heuristic
            "j_theoretical_add": float(j_composed),  # heuristic
            "efficiency_ratio": float("nan"),
            "cc_defined": False,
            "reason": "no_individual_j_provided",
        }

    j_vals = [float(v) for v in j_individual.values()]
    j_max = float(max(j_vals))

    delta_add = float(j_composed - j_max)

    # cc_max meaning requires a positive best individual effect.
    if j_max <= 0:
        cc_max = float("nan")
        cc_defined = False
        cc_reason = "j_max_individual<=0"
    else:
        cc_max = float(j_composed / j_max)
        cc_defined = True
        cc_reason = "ok"

    # Heuristic composition references (kept as legacy keys).
    # These are NOT proven general bounds without explicit assumptions.
    if len(j_vals) > 1:
        heuristic_max = float(min(sum(j_vals), p0))
        heuristic_add = float(min(sum(j_vals) - float(np.prod(j_vals)), p0))
        j_mult = float(np.prod(j_vals))
    else:
        heuristic_max = float(j_vals[0])
        heuristic_add = float(j_vals[0])
        j_mult = float(j_vals[0])

    cc_mult = float(j_composed / j_mult) if j_mult > 0 else None

    efficiency_ratio = float(j_composed / heuristic_max) if heuristic_max != 0 else float("nan")

    return {
        "cc_max": cc_max,
        "delta_add": delta_add,
        "cc_multiplicative": cc_mult,
        "j_composed": float(j_composed),
        "j_max_individual": j_max,
        "j_theoretical_max": heuristic_max,  # legacy name, heuristic in reality
        "j_theoretical_add": heuristic_add,  # legacy name, heuristic in reality
        "efficiency_ratio": efficiency_ratio,
        "cc_defined": cc_defined,
        "reason": cc_reason,
    }


def bootstrap_ci_with_cc(
    results: AttackResults,
    j_individual: dict[str, float] | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    block_size: int | None = None,
    method: str = "percentile",
    compute_cc: bool = True,
    random_seed: int = 42,
    # --- Enterprise additions (all optional; keep old signature behavior) ---
    cluster_ids: np.ndarray | None = None,
    cluster_key: str | None = None,
    pair_key: str | None = None,
    resample: str = "auto",  # auto|iid|block|cluster|pair
    ordered: bool | None = None,
    min_valid_frac: float = 0.90,
    n_permutations: int = 0,
    tier: str = "auto",
    return_bootstrap_samples: bool = True,
) -> BootstrapResult:
    """
    Bootstrap confidence intervals for J-statistic and CC metrics with two-world handling.

    Backward compatibility:
      - If you call this exactly like the old function, you still get:
          stratified iid bootstrap, percentile/basic/"bca"(now real BCa),
          hypothesis_tests populated, block_size returned (estimated if None).

    Enterprise behavior:
      - Chooses a defensible resampling scheme (auto):
          pair > cluster > block (only if ordered) > iid
      - Tracks failure counts and returns valid flag + NaN CIs if invalid.

    Args:
      results: list of AttackResult
      j_individual: dict rail_name -> J for individual rails
      B, alpha, method: bootstrap settings
      block_size: used only if resample=block (or auto chooses block) and ordering is valid
      cluster_ids/cluster_key: provides clustering for cluster bootstrap
      pair_key: identifies matched pairs across worlds (e.g., attack_id)
      resample: "auto" recommended
      ordered: True/False if you know ordering is meaningful; if None, we try to infer via timestamp monotonicity
      min_valid_frac: validity threshold
      n_permutations: optional permutation test count (0 disables)
      tier: "auto" or explicit label
      return_bootstrap_samples: store bootstrap J samples in result (can be memory heavy)

    Returns:
      BootstrapResult (audit-grade)
    """
    if not results:
        raise ValueError("bootstrap_ci_with_cc: empty results")

    n_total = len(results)
    if n_total < 50:
        warnings.warn(
            f"bootstrap_ci_with_cc: small sample size (n={n_total}). Inference may be unstable.",
            stacklevel=2,
        )

    # --------- Extract arrays / metadata ---------
    world = np.array([_get_world_bit(r) for r in results], dtype=int)
    y = np.array([_get_success(r) for r in results], dtype=int)

    # order metric (only for block inference)
    inferred_ordered = _infer_ordering(results)
    ordered_final = inferred_ordered if ordered is None else bool(ordered)

    # cluster ids
    cluster_vec: np.ndarray | None = None
    cluster_key_used: str | None = None
    if cluster_ids is not None:
        cluster_vec = np.asarray(cluster_ids)
        if cluster_vec.shape[0] != n_total:
            raise ValueError("bootstrap_ci_with_cc: cluster_ids length must match results length")
        cluster_key_used = "cluster_ids"
    elif cluster_key is not None:
        cluster_vec = np.array(
            [_safe_to_str(getattr(r, cluster_key, None)) for r in results], dtype=object
        )
        cluster_key_used = cluster_key

    # pairing ids
    pair_vec: np.ndarray | None = None
    if pair_key is not None:
        pair_vec = np.array(
            [_safe_to_str(getattr(r, pair_key, None)) for r in results], dtype=object
        )

    # stratify
    idx0 = np.where(world == 0)[0]
    idx1 = np.where(world == 1)[0]
    n0 = int(idx0.size)
    n1 = int(idx1.size)
    if n0 == 0 or n1 == 0:
        raise ValueError(f"bootstrap_ci_with_cc: need both worlds; got n0={n0}, n1={n1}")

    # point estimates
    j_emp, p0_emp, p1_emp = compute_j_statistic(results)

    # CC point estimates
    if compute_cc and j_individual:
        cc_metrics = compute_composability_coefficients(j_emp, j_individual, p0_emp)
        cc_max_emp = float(cc_metrics["cc_max"])
        delta_add_emp = float(cc_metrics["delta_add"])
        cc_mult_emp = cc_metrics.get("cc_multiplicative", None)
        j_max_individual = (
            float(cc_metrics["j_max_individual"])
            if np.isfinite(cc_metrics["j_max_individual"])
            else float("nan")
        )
        cc_defined = bool(cc_metrics.get("cc_defined", False))
        cc_reason = str(cc_metrics.get("reason", ""))
    else:
        cc_max_emp = float("nan")
        delta_add_emp = float("nan")
        cc_mult_emp = None
        j_max_individual = float("nan")
        cc_defined = False
        cc_reason = "compute_cc_disabled_or_missing_j_individual"

    # tier label
    tier_label = _infer_tier_label(tier=tier, cluster_vec=cluster_vec, pair_vec=pair_vec)

    # provenance hashes (order-insensitive + order-sensitive)
    input_hash, order_hash = _compute_provenance_hashes(
        results=results,
        config={
            "B": B,
            "alpha": alpha,
            "method": method,
            "compute_cc": compute_cc,
            "block_size": block_size,
            "cluster_key": cluster_key_used,
            "pair_key": pair_key,
            "resample": resample,
            "ordered": ordered_final,
            "tier": tier_label,
        },
    )

    # choose resampling mode
    resample_mode = _choose_resample_mode(
        requested=resample,
        pair_vec=pair_vec,
        cluster_vec=cluster_vec,
        block_size=block_size,
        ordered=ordered_final,
    )

    # block size handling (kept for compatibility)
    block_size_used: int | None = None
    if resample_mode == "block":
        if not ordered_final:
            warnings.warn(
                "Requested/selected block bootstrap but ordered=False; falling back to iid.",
                stacklevel=2,
            )
            resample_mode = "iid"
        else:
            if block_size is None:
                block_size_used = _estimate_block_size(n_total)
            else:
                block_size_used = int(block_size)
            if block_size_used < 1:
                block_size_used = 1

    # dependence diagnostics if clusters provided
    icc0 = icc1 = None
    ncl0 = ncl1 = None
    ess0 = ess1 = None
    if cluster_vec is not None:
        icc0, ncl0, ess0 = _icc_and_ess(y[idx0], cluster_vec[idx0])
        icc1, ncl1, ess1 = _icc_and_ess(y[idx1], cluster_vec[idx1])

    # --------- Bootstrap engine ---------
    ss = np.random.SeedSequence(random_seed)
    rng = np.random.default_rng(ss)

    boot_j: list[float] = []
    boot_cc: list[float] = []
    boot_delta: list[float] = []
    n_failed = 0
    reasons: dict[str, int] = {}

    # Precompute structures for faster resampling
    # (pair bootstrap uses pairs as unit)
    pair_struct = None
    if resample_mode == "pair" and pair_vec is not None:
        pair_struct = _build_pairs(idx0, idx1, y, pair_vec)
        if pair_struct is None:
            warnings.warn(
                "pair bootstrap requested/selected but pairs are invalid; falling back to iid.",
                stacklevel=2,
            )
            resample_mode = "iid"

    # cluster structures
    cluster_struct0 = cluster_struct1 = None
    if resample_mode == "cluster" and cluster_vec is not None:
        cluster_struct0 = _build_clusters(idx0, cluster_vec)
        cluster_struct1 = _build_clusters(idx1, cluster_vec)
        if cluster_struct0 is None or cluster_struct1 is None:
            warnings.warn(
                "cluster bootstrap requested/selected but clusters are invalid; falling back to iid.",
                stacklevel=2,
            )
            resample_mode = "iid"

    # ordered indices within each world for block bootstrap
    ordered_idx0 = idx0
    ordered_idx1 = idx1
    if resample_mode == "block":
        # within-world order is the original list order; that is the best we can do
        ordered_idx0 = idx0
        ordered_idx1 = idx1

    for _b in range(B):
        try:
            if resample_mode == "pair":
                assert pair_struct is not None
                j_b, cc_b, d_b = _bootstrap_stat_pair(
                    pair_struct=pair_struct,
                    rng=rng,
                    compute_cc=compute_cc,
                    j_max_individual=j_max_individual,
                )
            else:
                # produce resampled indices for each world
                if resample_mode == "iid":
                    s0 = rng.choice(idx0, size=n0, replace=True)
                    s1 = rng.choice(idx1, size=n1, replace=True)
                elif resample_mode == "block":
                    s0 = _resample_blocks(ordered_idx0, block_size_used or 1, rng)
                    s1 = _resample_blocks(ordered_idx1, block_size_used or 1, rng)
                elif resample_mode == "cluster":
                    assert cluster_struct0 is not None and cluster_struct1 is not None
                    s0 = _resample_clusters(cluster_struct0, rng)
                    s1 = _resample_clusters(cluster_struct1, rng)
                else:
                    raise RuntimeError(f"Unknown resample mode: {resample_mode}")

                # compute J
                p0_b = float(y[s0].mean()) if s0.size else math.nan
                p1_b = float(y[s1].mean()) if s1.size else math.nan
                j_b = float(p0_b - p1_b)

                if compute_cc and j_individual:
                    if np.isfinite(j_max_individual) and j_max_individual > 0:
                        cc_b = float(j_b / j_max_individual)
                    else:
                        cc_b = float("nan")
                    d_b = (
                        float(j_b - j_max_individual)
                        if np.isfinite(j_max_individual)
                        else float("nan")
                    )
                else:
                    cc_b = float("nan")
                    d_b = float("nan")

            boot_j.append(j_b)
            boot_cc.append(cc_b)
            boot_delta.append(d_b)

        except Exception as e:
            n_failed += 1
            k = type(e).__name__
            reasons[k] = reasons.get(k, 0) + 1

    boot_j_arr = np.array(boot_j, dtype=float)
    boot_cc_arr = np.array(boot_cc, dtype=float)
    boot_delta_arr = np.array(boot_delta, dtype=float)

    # valid replicate accounting
    boot_j_arr = boot_j_arr[np.isfinite(boot_j_arr)]
    boot_cc_arr = boot_cc_arr[np.isfinite(boot_cc_arr)]
    boot_delta_arr = boot_delta_arr[np.isfinite(boot_delta_arr)]

    n_valid = int(boot_j_arr.size)
    failure_rate = float(n_failed / max(B, 1))
    valid = n_valid >= int(min_valid_frac * B)

    if not valid:
        warnings.warn(
            f"bootstrap_ci_with_cc: insufficient valid bootstrap replicates "
            f"(valid={n_valid}/{B}, failure_rate={failure_rate:.2%}). Returning NaN CIs.",
            stacklevel=2,
        )

    # --------- CI computation ---------
    if valid:
        ci_j = _compute_ci(
            samples=boot_j_arr,
            theta_hat=j_emp,
            alpha=alpha,
            method=method,
            jackknife_unit=_jackknife_unit(resample_mode),
            jackknife_data=_jackknife_data_for_mode(
                mode=resample_mode,
                y=y,
                idx0=idx0,
                idx1=idx1,
                pair_struct=pair_struct,
                cluster_struct0=cluster_struct0,
                cluster_struct1=cluster_struct1,
            ),
        )

        if compute_cc and j_individual:
            ci_cc = _compute_ci(
                samples=boot_cc_arr,
                theta_hat=cc_max_emp,
                alpha=alpha,
                method=method,
                jackknife_unit=_jackknife_unit(resample_mode),
                jackknife_data=_jackknife_data_for_mode(
                    mode=resample_mode,
                    y=y,
                    idx0=idx0,
                    idx1=idx1,
                    pair_struct=pair_struct,
                    cluster_struct0=cluster_struct0,
                    cluster_struct1=cluster_struct1,
                    stat_kind="cc",
                    j_max_individual=j_max_individual,
                ),
            )
            ci_delta = _compute_ci(
                samples=boot_delta_arr,
                theta_hat=delta_add_emp,
                alpha=alpha,
                method=method,
                jackknife_unit=_jackknife_unit(resample_mode),
                jackknife_data=_jackknife_data_for_mode(
                    mode=resample_mode,
                    y=y,
                    idx0=idx0,
                    idx1=idx1,
                    pair_struct=pair_struct,
                    cluster_struct0=cluster_struct0,
                    cluster_struct1=cluster_struct1,
                    stat_kind="delta",
                    j_max_individual=j_max_individual,
                ),
            )
        else:
            ci_cc = (math.nan, math.nan)
            ci_delta = (math.nan, math.nan)
    else:
        ci_j = (math.nan, math.nan)
        ci_cc = (math.nan, math.nan)
        ci_delta = (math.nan, math.nan)

    # diagnostics
    convergence = _compute_convergence_diagnostic(boot_j_arr) if valid else math.nan
    ess_global = _global_ess(resample_mode, n0, n1, ess0, ess1, block_size_used, ordered_final)

    # hypothesis tests (defensible by default; permutation optional)
    tests = _run_hypothesis_tests_enterprise(
        y0=y[idx0],
        y1=y[idx1],
        j_emp=j_emp,
        cc_emp=cc_max_emp,
        cc_defined=cc_defined,
        pair_struct=pair_struct,
        rng=np.random.default_rng(ss.spawn(1)[0]),
        n_permutations=int(n_permutations),
        alpha=float(alpha),
    )

    # regime-interpretation sanity (cc may be NaN)
    # (interpret_cc_results also available as separate function)
    # Keep regime set to the 3 legacy labels; explain undefined in evidence.

    # assemble result
    return BootstrapResult(
        j_statistic=float(j_emp),
        p0=float(p0_emp),
        p1=float(p1_emp),
        cc_max=float(cc_max_emp) if np.isfinite(cc_max_emp) else float("nan"),
        delta_add=float(delta_add_emp) if np.isfinite(delta_add_emp) else float("nan"),
        cc_multiplicative=float(cc_mult_emp)
        if isinstance(cc_mult_emp, (int, float)) and np.isfinite(cc_mult_emp)
        else None,
        ci_j=ci_j,
        ci_cc_max=ci_cc,
        ci_delta_add=ci_delta,
        ci_width=float(ci_j[1] - ci_j[0])
        if np.isfinite(ci_j[0]) and np.isfinite(ci_j[1])
        else math.nan,
        bootstrap_samples=(boot_j_arr.copy() if (return_bootstrap_samples and valid) else None),
        n_sessions=int(n_total),
        n_bootstrap=int(B),
        convergence_diagnostic=float(convergence),
        effective_sample_size=float(ess_global) if np.isfinite(ess_global) else math.nan,
        n_valid=int(n_valid),
        n_failed=int(n_failed),
        failure_rate=float(failure_rate),
        failure_reasons=dict(reasons),
        valid=bool(valid),
        resample=resample_mode,
        ordered=ordered_final,
        block_size=block_size_used
        if resample_mode == "block"
        else block_size,  # preserve caller's view
        cluster_key=cluster_key_used,
        n_clusters_world0=ncl0,
        n_clusters_world1=ncl1,
        icc_world0=icc0,
        icc_world1=icc1,
        ess_world0=ess0,
        ess_world1=ess1,
        method=str(method),
        tier=str(tier_label),
        input_hash=str(input_hash),
        order_hash=str(order_hash),
        hypothesis_tests={
            **tests,
            "cc_defined": cc_defined,
            "cc_undefined_reason": cc_reason,
        },
    )


def dkw_confidence_bound(n: int, alpha: float = 0.05) -> float:
    """Dvoretzky-Kiefer-Wolfowitz uniform confidence band halfwidth."""
    return math.sqrt(math.log(2 / alpha) / (2 * max(n, 1)))


def analytical_j_ci(
    results: AttackResults, alpha: float = 0.05
) -> tuple[float, ConfidenceInterval]:
    """
    Analytical CI for J via normal approximation (unpaired, iid-ish).

    Enterprise fix:
      - NEVER clamps J CI to [0,1]. J lives in [-1,1] naturally.
    """
    j, p0, p1 = compute_j_statistic(results)
    world = np.array([_get_world_bit(r) for r in results], dtype=int)
    n0 = int((world == 0).sum())
    n1 = int((world == 1).sum())

    # avoid div-by-zero
    se_p0 = math.sqrt(p0 * (1 - p0) / n0) if n0 > 0 else float("nan")
    se_p1 = math.sqrt(p1 * (1 - p1) / n1) if n1 > 0 else float("nan")
    se_j = (
        math.sqrt(se_p0**2 + se_p1**2)
        if np.isfinite(se_p0) and np.isfinite(se_p1)
        else float("nan")
    )

    z = float(scipy_stats.norm.ppf(1 - alpha / 2))
    ci = (float(j - z * se_j), float(j + z * se_j)) if np.isfinite(se_j) else (math.nan, math.nan)
    return float(j), ci


def sanity_check_j_statistic(results: AttackResults) -> dict[str, bool]:
    """
    Sanity checks for J-statistic validity.

    Enterprise fix:
      - J range is [-1,1], not [0,1].
      - "non_trivial" check is replaced by "not_extreme" with symmetric bounds.
    """
    if not results:
        return {"has_results": False}

    j_stat, p0, p1 = compute_j_statistic(results)
    world = np.array([_get_world_bit(r) for r in results], dtype=int)
    n0 = int((world == 0).sum())
    n1 = int((world == 1).sum())
    n = len(results)

    checks = {
        "has_results": n > 0,
        "both_worlds_present": n0 > 0 and n1 > 0,
        "j_in_range": (-1.0 <= j_stat <= 1.0),
        "p0_valid": (0.0 <= p0 <= 1.0),
        "p1_valid": (0.0 <= p1 <= 1.0),
        "j_consistent": abs(j_stat - (p0 - p1)) < 1e-12,
        "sufficient_samples": n >= 100,
        "balanced_worlds": 0.4 < (n0 / n) < 0.6,
        "not_extreme": abs(j_stat) < 0.99,
        "sufficient_variation": (0.0 < p0 < 1.0) and (0.0 < p1 < 1.0),
    }
    return checks


def interpret_cc_results(
    result: BootstrapResult,
    threshold_constructive: float = 1.05,
    threshold_destructive: float = 0.95,
) -> CCInterpretation:
    """
    Interpret CC results with corrected regime directionality.

    Regimes (legacy 3-label contract):
      - constructive: cc_max > threshold_constructive
      - destructive : cc_max < threshold_destructive
      - independent : otherwise

    If cc_max is NaN (e.g., best individual J <= 0), we return "independent" with
    confidence=0 and include reason in evidence.
    """
    cc = result.cc_max
    ci = result.ci_cc_max

    evidence = {
        "cc_max": cc if np.isfinite(cc) else None,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "ci_width": (ci[1] - ci[0]) if np.isfinite(ci[0]) and np.isfinite(ci[1]) else None,
        "delta_add": result.delta_add,
        "j_statistic": result.j_statistic,
        "n_sessions": result.n_sessions,
        "convergence": result.convergence_diagnostic,
        "valid": result.valid,
        "resample": result.resample,
        "tier": result.tier,
        "cc_defined": bool(result.hypothesis_tests.get("cc_defined", np.isfinite(cc))),
        "cc_undefined_reason": result.hypothesis_tests.get("cc_undefined_reason"),
    }

    if not np.isfinite(cc):
        return CCInterpretation(
            regime="independent",
            confidence=0.0,
            recommendation="CC undefined (best individual J <= 0 or missing). Inspect J and individual rails.",
            evidence=evidence,
        )

    # Handle CI degeneracy safely
    ci_width = (ci[1] - ci[0]) if np.isfinite(ci[0]) and np.isfinite(ci[1]) else float("inf")
    if ci_width <= 0 or not np.isfinite(ci_width):
        ci_width = float("inf")

    if cc > threshold_constructive:
        regime = "constructive"
        confidence = (
            1.0
            if (np.isfinite(ci[0]) and ci[0] > threshold_constructive)
            else max(0.0, float((cc - threshold_constructive) / ci_width))
        )
        recommendation = "Deploy composition - synergistic protection detected"
    elif cc < threshold_destructive:
        regime = "destructive"
        confidence = (
            1.0
            if (np.isfinite(ci[1]) and ci[1] < threshold_destructive)
            else max(0.0, float((threshold_destructive - cc) / ci_width))
        )
        recommendation = "Avoid composition - interference detected"
    else:
        regime = "independent"
        # confidence is "CI contained inside [destructive, constructive]" if possible
        if np.isfinite(ci[0]) and np.isfinite(ci[1]):
            confidence = (
                1.0 if (ci[0] >= threshold_destructive and ci[1] <= threshold_constructive) else 0.5
            )
        else:
            confidence = 0.5
        recommendation = "Use single best guardrail - no interaction benefit"

    return CCInterpretation(
        regime=regime,
        confidence=float(confidence),
        recommendation=recommendation,
        evidence=evidence,
    )


# =============================================================================
# Internals: coercion, provenance, resampling, CI methods
# =============================================================================

_ALLOWED_PROVENANCE_FIELDS = (
    "attack_id",
    "world_bit",
    "success",
    "guardrails_applied",
    "transcript_hash",
    "rng_seed",
    "timestamp",
    "session_id",
    "cluster_id",
)


def _get_world_bit(r: Any) -> int:
    w = getattr(r, "world_bit", None)
    if w is None:
        raise ValueError("AttackResult missing required attribute: world_bit")
    try:
        wi = int(w)
    except Exception as e:
        raise ValueError(f"Invalid world_bit={w!r}") from e
    if wi not in (0, 1):
        raise ValueError(f"world_bit must be 0 or 1; got {wi}")
    return wi


def _get_success(r: Any) -> int:
    s = getattr(r, "success", None)
    if s is None:
        raise ValueError("AttackResult missing required attribute: success")
    # accept bool, int-like, numpy scalars
    try:
        si = int(s)
    except Exception as e:
        raise ValueError(f"Invalid success={s!r}") from e
    if si not in (0, 1):
        raise ValueError(f"success must be binary (0/1); got {si}")
    return si


def _coerce_binary_array(arr: np.ndarray, name: str) -> np.ndarray:
    try:
        out = arr.astype(int)
    except Exception as e:
        raise ValueError(f"{name}: cannot coerce to int") from e
    if out.size and not np.isin(out, [0, 1]).all():
        bad = out[~np.isin(out, [0, 1])][:5]
        raise ValueError(f"{name}: non-binary values found, e.g. {bad!r}")
    return out


def _safe_to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, sort_keys=True, default=str)
    except Exception:
        return str(x)


def _infer_ordering(results: AttackResults) -> bool:
    """
    Infer whether ordering is defensible for block bootstrap.
    Conservative: only returns True if timestamps exist and are monotone non-decreasing.
    """
    ts = []
    for r in results:
        t = getattr(r, "timestamp", None)
        if t is None:
            return False
        try:
            ts.append(float(t))
        except Exception:
            return False
    return all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


def _estimate_block_size(n: int) -> int:
    # rule of thumb: n^(1/3)
    return max(1, round(n ** (1.0 / 3.0)))


def _choose_resample_mode(
    requested: str,
    pair_vec: np.ndarray | None,
    cluster_vec: np.ndarray | None,
    block_size: int | None,
    ordered: bool,
) -> str:
    req = (requested or "auto").lower()
    if req in ("iid", "block", "cluster", "pair"):
        if req == "block" and not ordered:
            warnings.warn(
                "block resampling requested but ordered=False; will fall back to iid.", stacklevel=2
            )
            return "iid"
        if req == "cluster" and cluster_vec is None:
            warnings.warn(
                "cluster resampling requested but no cluster labels provided; falling back to iid.",
                stacklevel=2,
            )
            return "iid"
        if req == "pair" and pair_vec is None:
            warnings.warn(
                "pair resampling requested but no pair_key provided; falling back to iid.",
                stacklevel=2,
            )
            return "iid"
        return req

    # auto policy: pair > cluster > block (if ordered and block_size>1 or specified) > iid
    if pair_vec is not None:
        return "pair"
    if cluster_vec is not None:
        return "cluster"
    if ordered and (block_size is None or (block_size is not None and int(block_size) > 1)):
        return "block"
    return "iid"


def _infer_tier_label(
    tier: str, cluster_vec: np.ndarray | None, pair_vec: np.ndarray | None
) -> str:
    if tier and tier != "auto":
        return str(tier)
    if pair_vec is not None or cluster_vec is not None:
        return "Tier 1"
    return "Tier 0"


def _compute_provenance_hashes(results: AttackResults, config: dict[str, Any]) -> tuple[str, str]:
    """
    Produce both:
      - input_hash (order-insensitive multiset + config)
      - order_hash (order-sensitive + config)
    """
    rows = []
    for i, r in enumerate(results):
        row = {"_i": i}
        for k in _ALLOWED_PROVENANCE_FIELDS:
            row[k] = _safe_to_str(getattr(r, k, None))
        rows.append(row)

    # order-sensitive
    order_payload = {"rows": rows, "config": config}
    order_bytes = json.dumps(
        order_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    order_hash = hashlib.sha256(order_bytes).hexdigest()[:16]

    # order-insensitive (multiset): sort rows by stable tuple (attack_id, world_bit, timestamp, index)
    def sort_key(d: dict[str, Any]) -> tuple[str, str, str, int]:
        return (
            d.get("attack_id", ""),
            d.get("world_bit", ""),
            d.get("timestamp", ""),
            int(d["_i"]),
        )

    rows_sorted = sorted(rows, key=sort_key)
    multiset_payload = {"rows": rows_sorted, "config": config}
    multiset_bytes = json.dumps(
        multiset_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    input_hash = hashlib.sha256(multiset_bytes).hexdigest()[:16]

    return input_hash, order_hash


def _resample_blocks(indices: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving block bootstrap with wrap-around. Returns an array of length len(indices).
    """
    n = int(indices.size)
    if n == 0:
        return indices.copy()
    bs = max(1, int(block_size))
    n_blocks = math.ceil(n / bs)
    starts = rng.integers(0, n, size=n_blocks)
    out = []
    for s in starts:
        # wrap block
        for k in range(bs):
            out.append(indices[(s + k) % n])
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return np.array(out, dtype=int)


def _build_clusters(world_indices: np.ndarray, cluster_vec: np.ndarray) -> list[np.ndarray] | None:
    """
    Build list of arrays, each containing indices belonging to a cluster.
    Returns None if clusters degenerate.
    """
    if world_indices.size == 0:
        return None
    labels = cluster_vec[world_indices]
    uniq = np.unique(labels)
    if uniq.size < 2:
        return None
    clusters: list[np.ndarray] = []
    for c in uniq:
        members = world_indices[labels == c]
        if members.size > 0:
            clusters.append(members.astype(int))
    if len(clusters) < 2:
        return None
    return clusters


def _resample_clusters(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """
    Resample clusters with replacement, include all members of selected clusters.
    Number of clusters drawn equals number of unique clusters.
    Sample size is random (audit-visible).
    """
    k = len(clusters)
    picks = rng.integers(0, k, size=k)
    out = np.concatenate([clusters[i] for i in picks], axis=0)
    return out.astype(int)


def _build_pairs(
    idx0: np.ndarray, idx1: np.ndarray, y: np.ndarray, pair_vec: np.ndarray
) -> np.ndarray | None:
    """
    Build array of shape (n_pairs, 2): [y0, y1] aligned by pair_id.
    Requires complete pairs (one obs per world per pair).
    """
    # map pair_id -> (y0?, y1?)
    m: dict[str, list[int | None]] = {}
    for i in idx0:
        pid = pair_vec[i]
        if pid == "":
            continue
        if pid not in m:
            m[pid] = [None, None]
        m[pid][0] = int(y[i])
    for i in idx1:
        pid = pair_vec[i]
        if pid == "":
            continue
        if pid not in m:
            m[pid] = [None, None]
        m[pid][1] = int(y[i])

    pairs = []
    for pid, (a, b) in m.items():
        if a is None or b is None:
            continue
        pairs.append((a, b))
    if len(pairs) < 2:
        return None
    return np.array(pairs, dtype=int)


def _bootstrap_stat_pair(
    pair_struct: np.ndarray,
    rng: np.random.Generator,
    compute_cc: bool,
    j_max_individual: float,
) -> tuple[float, float, float]:
    """
    Bootstrap over pairs (rows), resampling rows with replacement.
    """
    n_pairs = int(pair_struct.shape[0])
    pick = rng.integers(0, n_pairs, size=n_pairs)
    s = pair_struct[pick, :]
    p0 = float(s[:, 0].mean())
    p1 = float(s[:, 1].mean())
    j = float(p0 - p1)
    if compute_cc and np.isfinite(j_max_individual) and j_max_individual > 0:
        cc = float(j / j_max_individual)
        d = float(j - j_max_individual)
    else:
        cc = float("nan")
        d = float("nan")
    return j, cc, d


def _jackknife_unit(mode: str) -> str:
    # used to choose jackknife scheme
    if mode == "pair":
        return "pair"
    if mode == "cluster":
        return "cluster"
    return "observation"


def _jackknife_data_for_mode(
    mode: str,
    y: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    pair_struct: np.ndarray | None,
    cluster_struct0: list[np.ndarray] | None,
    cluster_struct1: list[np.ndarray] | None,
    stat_kind: str = "j",
    j_max_individual: float | None = None,
) -> dict[str, Any]:
    """
    Provide enough information for jackknife statistic recomputation.
    stat_kind: "j" | "cc" | "delta"
    """
    return {
        "mode": mode,
        "y": y,
        "idx0": idx0,
        "idx1": idx1,
        "pair_struct": pair_struct,
        "cluster_struct0": cluster_struct0,
        "cluster_struct1": cluster_struct1,
        "stat_kind": stat_kind,
        "j_max_individual": j_max_individual,
    }


def _compute_ci(
    samples: np.ndarray,
    theta_hat: float,
    alpha: float,
    method: str,
    jackknife_unit: str,
    jackknife_data: dict[str, Any],
) -> ConfidenceInterval:
    m = (method or "percentile").lower()
    if samples.size == 0 or not np.isfinite(theta_hat):
        return (math.nan, math.nan)

    if m == "percentile":
        return _percentile_ci(samples, alpha)
    if m == "basic":
        return _basic_ci(samples, theta_hat, alpha)
    if m == "bca":
        return _bca_ci_full(samples, theta_hat, alpha, jackknife_unit, jackknife_data)

    raise ValueError(f"bootstrap_ci_with_cc: unknown CI method={method!r}")


def _percentile_ci(samples: np.ndarray, alpha: float) -> ConfidenceInterval:
    if samples.size == 0:
        return (math.nan, math.nan)
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return (lo, hi)


def _basic_ci(samples: np.ndarray, theta_hat: float, alpha: float) -> ConfidenceInterval:
    if samples.size == 0 or not np.isfinite(theta_hat):
        return (math.nan, math.nan)
    p_lo, p_hi = _percentile_ci(samples, alpha)
    return (float(2 * theta_hat - p_hi), float(2 * theta_hat - p_lo))


def _bca_ci_full(
    samples: np.ndarray,
    theta_hat: float,
    alpha: float,
    jackknife_unit: str,
    jackknife_data: dict[str, Any],
) -> ConfidenceInterval:
    """
    Full BCa interval:
      z0 from bootstrap bias,
      a from jackknife acceleration on the resampling unit.

    Notes:
      - BCa on cluster bootstrap uses leave-one-cluster-out jackknife.
      - BCa on pair bootstrap uses leave-one-pair-out jackknife.
      - If jackknife is degenerate (too few units or zero variance), we fall back
        to bias-corrected percentile (a=0) and warn.
    """
    if samples.size == 0 or not np.isfinite(theta_hat):
        return (math.nan, math.nan)

    # bias correction
    prop = float(np.mean(samples < theta_hat))
    prop = min(max(prop, 1e-6), 1 - 1e-6)  # avoid inf
    z0 = float(scipy_stats.norm.ppf(prop))

    # acceleration via jackknife
    jk = _jackknife_estimates(jackknife_unit, jackknife_data)
    a = 0.0
    if jk is None or jk.size < 3 or not np.isfinite(jk).all():
        warnings.warn(
            "BCa acceleration unavailable (degenerate jackknife); using a=0 fallback.", stacklevel=2
        )
    else:
        jk_mean = float(np.mean(jk))
        num = float(np.sum((jk_mean - jk) ** 3))
        den = float(6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5))
        if den <= 0 or not np.isfinite(num) or not np.isfinite(den):
            warnings.warn("BCa acceleration ill-conditioned; using a=0 fallback.", stacklevel=2)
        else:
            a = num / den

    z_lo = float(scipy_stats.norm.ppf(alpha / 2.0))
    z_hi = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))

    # adjusted alpha levels
    a1 = _bca_alpha(z0, a, z_lo)
    a2 = _bca_alpha(z0, a, z_hi)

    # clamp and ensure order
    a1 = float(min(max(a1, 0.0), 1.0))
    a2 = float(min(max(a2, 0.0), 1.0))
    if a2 < a1:
        a1, a2 = a2, a1

    lo = float(np.quantile(samples, a1))
    hi = float(np.quantile(samples, a2))
    return (lo, hi)


def _bca_alpha(z0: float, a: float, z_alpha: float) -> float:
    denom = 1.0 - a * (z0 + z_alpha)
    if denom == 0:
        return float(scipy_stats.norm.cdf(z0 + z0 + z_alpha))
    adj = z0 + (z0 + z_alpha) / denom
    return float(scipy_stats.norm.cdf(adj))


def _jackknife_estimates(jackknife_unit: str, data: dict[str, Any]) -> np.ndarray | None:
    """
    Compute jackknife leave-one-unit-out estimates of the requested statistic.
    """
    data["mode"]
    stat_kind = data.get("stat_kind", "j")
    y = data["y"]
    idx0 = data["idx0"]
    idx1 = data["idx1"]
    j_max = data.get("j_max_individual")

    def stat_from_masks(m0: np.ndarray, m1: np.ndarray) -> float:
        if m0.size == 0 or m1.size == 0:
            return float("nan")
        p0 = float(y[m0].mean())
        p1 = float(y[m1].mean())
        j = float(p0 - p1)
        if stat_kind == "j":
            return j
        if stat_kind == "cc":
            if j_max is None or not np.isfinite(j_max) or j_max <= 0:
                return float("nan")
            return float(j / float(j_max))
        if stat_kind == "delta":
            if j_max is None or not np.isfinite(j_max):
                return float("nan")
            return float(j - float(j_max))
        return float("nan")

    if jackknife_unit == "pair":
        pair_struct = data.get("pair_struct")
        if pair_struct is None or pair_struct.shape[0] < 3:
            return None
        n_pairs = int(pair_struct.shape[0])
        out = np.empty(n_pairs, dtype=float)
        # leave one pair out
        for i in range(n_pairs):
            s = np.delete(pair_struct, i, axis=0)
            if s.shape[0] == 0:
                out[i] = float("nan")
                continue
            p0 = float(s[:, 0].mean())
            p1 = float(s[:, 1].mean())
            j = float(p0 - p1)
            if stat_kind == "j":
                out[i] = j
            elif stat_kind == "cc":
                out[i] = (
                    float(j / j_max)
                    if (j_max is not None and np.isfinite(j_max) and j_max > 0)
                    else float("nan")
                )
            elif stat_kind == "delta":
                out[i] = (
                    float(j - j_max) if (j_max is not None and np.isfinite(j_max)) else float("nan")
                )
            else:
                out[i] = float("nan")
        return out

    if jackknife_unit == "cluster":
        c0 = data.get("cluster_struct0")
        c1 = data.get("cluster_struct1")
        if not c0 or not c1:
            return None
        # leave-one-cluster-out jackknife: remove cluster k from world 0 and cluster k from world 1 separately
        # We do *two* jackknives and combine by concatenation (conservative; many units).
        outs = []
        for clusters, _widx_other in ((c0, idx1), (c1, idx0)):
            k = len(clusters)
            if k < 2:
                continue
            for i in range(k):
                removed = clusters[i]
                if clusters is c0:
                    m0 = np.setdiff1d(idx0, removed, assume_unique=False)
                    m1 = idx1
                else:
                    m0 = idx0
                    m1 = np.setdiff1d(idx1, removed, assume_unique=False)
                outs.append(stat_from_masks(m0, m1))
        if len(outs) < 3:
            return None
        return np.array(outs, dtype=float)

    # observation jackknife (default)
    n = int(idx0.size + idx1.size)
    if n < 3:
        return None

    out = np.empty(n, dtype=float)
    # jackknife over original list indices (keeps two-world structure but allows imbalance)
    all_idx = np.concatenate([idx0, idx1], axis=0)
    for t, drop in enumerate(all_idx):
        # remove one observation from its world
        if drop in idx0:
            m0 = idx0[idx0 != drop]
            m1 = idx1
        else:
            m0 = idx0
            m1 = idx1[idx1 != drop]
        out[t] = stat_from_masks(m0, m1)
    return out


def _compute_convergence_diagnostic(samples: np.ndarray) -> float:
    """Coefficient of variation (std / |mean|) for bootstrap statistic."""
    if samples.size == 0:
        return float("inf")
    mu = float(np.mean(samples))
    if abs(mu) < 1e-12:
        return float("inf")
    return float(np.std(samples, ddof=1) / abs(mu))


def _icc_and_ess(
    y: np.ndarray, clusters: np.ndarray
) -> tuple[float | None, int | None, float | None]:
    """
    One-way random-effects ICC (ANOVA-style) + Kish ESS for clustered samples.

    Returns:
      (icc, n_clusters, ess)

    Robust edge handling:
      - If <2 clusters or too few df, returns (0.0, k, Kish ESS) where possible.
    """
    if y.size == 0:
        return None, None, None

    # build cluster groups
    uniq = np.unique(clusters)
    k = int(uniq.size)
    if k < 2:
        # no clustering information
        return 0.0, k, float(y.size)

    # cluster means and sizes
    means = []
    sizes = []
    ss_within = 0.0
    for c in uniq:
        yc = y[clusters == c]
        if yc.size == 0:
            continue
        sizes.append(int(yc.size))
        mc = float(yc.mean())
        means.append(mc)
        ss_within += float(np.sum((yc - mc) ** 2))

    n = int(sum(sizes))
    if n <= k or n < 3:
        return 0.0, k, float(n)

    grand = float(np.average(means, weights=sizes))
    ss_between = float(np.sum([sz * (m - grand) ** 2 for sz, m in zip(sizes, means, strict=False)]))

    dfb = k - 1
    dfw = n - k
    msb = ss_between / dfb if dfb > 0 else 0.0
    msw = ss_within / dfw if dfw > 0 else 0.0

    # effective cluster size for unequal clusters: (sum n_i^2)/n
    sum_sq = float(np.sum(np.array(sizes, dtype=float) ** 2))
    m_eff = (sum_sq / n) if n > 0 else 1.0

    denom = msb + (m_eff - 1.0) * msw
    icc = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))

    # Kish ESS for cluster sampling: (sum w)^2 / sum w^2 with w=cluster size
    ess = float((n**2) / sum_sq) if sum_sq > 0 else float(n)
    return float(icc), k, ess


def _global_ess(
    mode: str,
    n0: int,
    n1: int,
    ess0: float | None,
    ess1: float | None,
    block_size: int | None,
    ordered: bool,
) -> float:
    """
    Provide a conservative single ESS diagnostic.
    - iid/pair: harmonic mean-ish for two-group comparison
    - cluster: based on world-specific Kish ESS if available
    - block: returns n / block_size as a crude upper bound on independent blocks
    """
    if mode == "cluster" and ess0 is not None and ess1 is not None:
        # effective n for difference of means is dominated by smaller effective group
        return float(1.0 / (1.0 / max(ess0, 1e-9) + 1.0 / max(ess1, 1e-9)))
    if mode == "block" and ordered and block_size and block_size > 1:
        return float((n0 + n1) / block_size)
    # iid/pair fallback
    return float(1.0 / (1.0 / max(n0, 1e-9) + 1.0 / max(n1, 1e-9)))


def _run_hypothesis_tests_enterprise(
    y0: np.ndarray,
    y1: np.ndarray,
    j_emp: float,
    cc_emp: float,
    cc_defined: bool,
    pair_struct: np.ndarray | None,
    rng: np.random.Generator,
    n_permutations: int,
    alpha: float,
) -> dict[str, Any]:
    """
    Produce coherent tests (p-values + notes). Avoid misleading bootstrap p-values.

    Unpaired:
      - Fisher exact (2x2)
      - two-proportion z-test
      - optional permutation test (labelled with assumptions)

    Paired:
      - McNemar exact
      - optional paired-swap permutation
    """
    tests: dict[str, Any] = {}

    # contingency table
    s0 = int(y0.sum())
    f0 = int(y0.size - s0)
    s1 = int(y1.sum())
    f1 = int(y1.size - s1)

    # Fisher exact (two-sided)
    try:
        _, p_fisher = scipy_stats.fisher_exact([[s0, f0], [s1, f1]], alternative="two-sided")
        tests["fisher_exact"] = {
            "p_value": float(p_fisher),
            "note": "Unpaired 2x2 Fisher exact, two-sided.",
        }
    except Exception as e:
        tests["fisher_exact"] = {"p_value": None, "error": type(e).__name__}

    # two-proportion z-test (asymptotic)
    try:
        p_pool = (s0 + s1) / max((y0.size + y1.size), 1)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / max(y0.size, 1) + 1 / max(y1.size, 1)))
        z = (float(y0.mean()) - float(y1.mean())) / se if se > 0 else float("inf")
        p_z = 2 * (1 - float(scipy_stats.norm.cdf(abs(z))))
        tests["z_test_diff_proportions"] = {
            "z": float(z),
            "p_value": float(p_z),
            "note": "Asymptotic z-test for p0-p1.",
        }
    except Exception as e:
        tests["z_test_diff_proportions"] = {"p_value": None, "error": type(e).__name__}

    # Paired McNemar if available
    if pair_struct is not None:
        try:
            a = pair_struct[:, 0]  # y0
            b = pair_struct[:, 1]  # y1
            # discordant counts
            n01 = int(np.sum((a == 0) & (b == 1)))
            n10 = int(np.sum((a == 1) & (b == 0)))
            # exact binomial on discordant
            n_disc = n01 + n10
            if n_disc > 0:
                # two-sided exact p-value
                p_mcn = 2 * min(
                    scipy_stats.binom.cdf(min(n01, n10), n_disc, 0.5),
                    1 - scipy_stats.binom.cdf(max(n01, n10) - 1, n_disc, 0.5),
                )
                p_mcn = min(1.0, float(p_mcn))
            else:
                p_mcn = 1.0
            tests["mcnemar_exact"] = {
                "n01": n01,
                "n10": n10,
                "p_value": float(p_mcn),
                "note": "Paired McNemar exact (discordant pairs).",
            }
        except Exception as e:
            tests["mcnemar_exact"] = {"p_value": None, "error": type(e).__name__}

    # Optional permutation p-values
    if n_permutations and n_permutations > 0:
        try:
            if pair_struct is not None:
                # paired swap permutation: swap within pair with prob 1/2
                obs = float(j_emp)
                more = 0
                for _ in range(int(n_permutations)):
                    swap = rng.integers(0, 2, size=pair_struct.shape[0]).astype(bool)
                    a = pair_struct[:, 0].copy()
                    b = pair_struct[:, 1].copy()
                    a[swap], b[swap] = b[swap], a[swap]
                    j = float(a.mean() - b.mean())
                    if abs(j) >= abs(obs):
                        more += 1
                p_perm = float((more + 1) / (n_permutations + 1))
                tests["permutation_paired_swap"] = {
                    "p_value": p_perm,
                    "note": "Valid under paired exchangeability; safest permutation mode.",
                }
            else:
                # label permutation: pool and reassign
                obs = float(j_emp)
                pooled = np.concatenate([y0, y1], axis=0)
                n0 = y0.size
                more = 0
                for _ in range(int(n_permutations)):
                    perm = rng.permutation(pooled.size)
                    a = pooled[perm[:n0]]
                    b = pooled[perm[n0:]]
                    j = float(a.mean() - b.mean())
                    if abs(j) >= abs(obs):
                        more += 1
                p_perm = float((more + 1) / (n_permutations + 1))
                tests["permutation_label_swap"] = {
                    "p_value": p_perm,
                    "note": "Valid only if world labels are exchangeable under the null.",
                }
        except Exception as e:
            tests["permutation"] = {"p_value": None, "error": type(e).__name__}

    # Optional CC “test” note (not a formal hypothesis test unless you define a null model for CC)
    if cc_defined and np.isfinite(cc_emp):
        tests["cc_note"] = {
            "cc_max": float(cc_emp),
            "note": "CC is a derived ratio; formal testing requires a specified null model for guardrail interactions.",
        }

    tests["alpha"] = float(alpha)
    return tests


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BootstrapResult",
    "CCInterpretation",
    "JBootstrapCI",
    "analytical_j_ci",
    "bootstrap_ci_j_statistic",
    "bootstrap_ci_with_cc",
    "compute_composability_coefficients",
    "compute_j_statistic",
    "dkw_confidence_bound",
    "interpret_cc_results",
    "sanity_check_j_statistic",
]
