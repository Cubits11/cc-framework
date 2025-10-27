# src/cc/analysis/alternative_metrics.py
"""
Alternative metrics for destructive interference detection (Comment 2).

This module implements *four complementary* approaches for evaluating whether a
composition of guardrails (e.g., AND/OR stacking) improves or degrades overall
detection performance relative to an **independence baseline**.

It is designed to run light-weight (NumPy / Pandas / scikit-learn only) and to
plug directly into the cc-framework results pipeline (Week 6–7 JSONL → fixed
metrics). It does **not** require R or heavy ML libraries.

-------------------------------------------------------------------------------
WHAT THIS FILE PROVIDES
-------------------------------------------------------------------------------

1) Euclidean Distance Index (to ROC "perfection" corner)
   - Geometry-based:  d = sqrt((1 - TPR)^2 + (FPR)^2), lower is better
   - Compares observed composition vs. *true* independence baseline point
     (TPR_indep, FPR_indep), not a proxy.

2) Cost-Weighted Error
   - Decision/ops-based:  cost = c_FP * FPR + c_FN * (1 - TPR), lower is better
   - Compares observed composition vs. independence baseline cost.

3) ΔJ against Independence (Youden’s J)
   - Statistics-based:  ΔJ = J_obs − J_indep with a practical significance band
   - 3-way classification: constructive | destructive | independent.

4) FH Envelope Percentile
   - Information-theoretic / model-free: position of J_obs within the
     Fréchet–Hoeffding envelope [FH_lower, FH_upper]; returns y in [0, 1]
   - 3-way classification via a neutral band centered at 0.5.

All four yield per-configuration classifications that you can compare
(agreement rates, κ scores) and analyze for disagreements/failure modes.

-------------------------------------------------------------------------------
CRITICAL REQUIREMENT (fixes the earlier bug)
-------------------------------------------------------------------------------

For Euclidean and Cost-Weighted comparisons, you **must** supply the *true*
independence baseline operating point **(tpr_indep, fpr_indep)**, not an
approximation. We verify consistency via:
    assert abs((tpr_indep - fpr_indep) - j_indep) < 1e-6

Where to get (TPR_indep, FPR_indep):
- Use the independence composition rules computed in your stats/estimation step
  (already present in cc-framework), e.g., for two rails:
    AND: TPR_indep = TPR_A * TPR_B; FPR_indep = FPR_A * FPR_B
    OR : TPR_indep = TPR_A + TPR_B - TPR_A * TPR_B
  (Generalization exists for >2 rails; the framework already computes these.)

-------------------------------------------------------------------------------
WHEN YOUDEN’S J OUTPERFORMS ALTERNATIVES (for the paper)
-------------------------------------------------------------------------------

- Deployment typically constrains FPR to a narrow policy band (e.g., ≤ 0.06).
  Euclidean Distance treats TPR and FPR symmetrically over the full ROC
  geometry, which can over-reward TPR gains that slightly violate policy FPR.
  J = TPR − FPR is a *linear* trade-off directly interpretable at a single
  operating point and respects such policy windows more transparently.
- Cost-Weighted Error needs accurate (c_FP, c_FN). If these are unknown or
  contentious across stakeholders, J provides a *domain-agnostic* summary.
- FH percentile is model-free but lacks effect-size (magnitude) semantics; ΔJ
  supplies both direction and magnitude relative to a testable baseline.

-------------------------------------------------------------------------------
INPUT SCHEMA (config_results)
-------------------------------------------------------------------------------

compare_all_metrics expects a dict:
    {
      "<config_name>": {
        "tpr": float,        # observed composition
        "fpr": float,        # observed composition
        "j_obs": float,      # observed J (== tpr - fpr)

        "tpr_indep": float,  # independence baseline (TRUE point)
        "fpr_indep": float,  # independence baseline (TRUE point)
        "j_indep": float,    # j_indep == tpr_indep - fpr_indep (verified)

        "fh_lower": float,   # FH lower bound for J
        "fh_upper": float,   # FH upper bound for J

        # Optional counts for future CI bootstraps (not used here):
        # "tp": int, "fp": int, "fn": int, "tn": int
      },
      ...
    }

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

A pandas.DataFrame with per-config values and classifications for each metric,
plus consensus diagnostics (majority label/size, unanimity flag).

-------------------------------------------------------------------------------
EXAMPLE
-------------------------------------------------------------------------------

>>> data = {
...   "demo": {
...     "tpr": 0.90, "fpr": 0.08, "j_obs": 0.82,
...     "tpr_indep": 0.87, "fpr_indep": 0.10, "j_indep": 0.77,
...     "fh_lower": 0.70, "fh_upper": 0.90
...   }
... }
>>> df = compare_all_metrics(data)
>>> df.loc[0, ["youden_delta_j", "youden_class"]].to_dict()
{'youden_delta_j': 0.050000000000000044, 'youden_class': 'constructive'}

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """
    Result from a single interference-detection metric.

    Attributes
    ----------
    metric_name : str
        Name of the metric.
    score : float
        Observed score at the composition operating point (e.g., distance or cost).
    baseline_score : Optional[float]
        Baseline score at the independence operating point, if applicable.
    delta : Optional[float]
        score - baseline_score when applicable; else None.
    classification : str
        One of {'constructive', 'destructive', 'independent'}.
    confidence_interval : Optional[Tuple[float, float]]
        Reserved for future CI reporting (requires counts, not just rates).
    notes : Optional[str]
        Free-form note for flags (e.g., “FH clip”, “out-of-envelope”).
    """
    metric_name: str
    score: float
    baseline_score: Optional[float]
    delta: Optional[float]
    classification: str
    confidence_interval: Optional[Tuple[float, float]] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Primitive scorers
# ---------------------------------------------------------------------------

def euclidean_distance_index(tpr: float, fpr: float) -> float:
    """
    Euclidean distance to ROC perfection corner (FPR=0, TPR=1). Lower is better.

    Parameters
    ----------
    tpr : float
        True Positive Rate ∈ [0, 1].
    fpr : float
        False Positive Rate ∈ [0, 1].

    Returns
    -------
    float
        Distance d = sqrt((1 - TPR)^2 + (FPR)^2).
    """
    tpr = float(tpr); fpr = float(fpr)
    return float(np.sqrt((1.0 - tpr) ** 2 + (fpr ** 2)))


def cost_weighted_error(
    tpr: float,
    fpr: float,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> float:
    """
    Expected cost: cost = c_FP * FPR + c_FN * (1 - TPR). Lower is better.

    Parameters
    ----------
    tpr : float
        True Positive Rate ∈ [0, 1].
    fpr : float
        False Positive Rate ∈ [0, 1].
    cost_fp : float, default=1.0
        Cost per false positive.
    cost_fn : float, default=1.0
        Cost per false negative.

    Returns
    -------
    float
        Expected cost in [0, cost_fp + cost_fn].
    """
    tpr = float(tpr); fpr = float(fpr)
    return float(cost_fp * fpr + cost_fn * (1.0 - tpr))


def independence_delta_j(
    j_observed: float,
    j_independent: float,
) -> float:
    """
    Compute ΔJ = J_obs − J_indep (no classification here).

    Parameters
    ----------
    j_observed : float
        Observed Youden’s J at the composition operating point.
    j_independent : float
        Independence-baseline Youden’s J.

    Returns
    -------
    float
        ΔJ.
    """
    return float(j_observed) - float(j_independent)


def fh_envelope_percentile(
    j_observed: float,
    fh_lower: float,
    fh_upper: float,
    *,
    clip_outside: bool = True,
) -> Tuple[float, Optional[str]]:
    """
    Position of J_obs within FH envelope: y = (J_obs − FH_lo) / (FH_hi − FH_lo).

    Handles degenerate/invalid envelopes and out-of-envelope observations.

    Parameters
    ----------
    j_observed : float
        Observed J.
    fh_lower : float
        Lower FH bound for J.
    fh_upper : float
        Upper FH bound for J.
    clip_outside : bool, default=True
        If True, clip y into [0, 1] and return a note if clipping occurred.

    Returns
    -------
    (float, Optional[str])
        y ∈ [0, 1] (possibly clipped) and an optional note ("fh_degenerate",
        "fh_clip_low", "fh_clip_high").
    """
    j_observed = float(j_observed)
    fh_lower = float(fh_lower)
    fh_upper = float(fh_upper)

    if not np.isfinite(fh_lower) or not np.isfinite(fh_upper):
        return 0.5, "fh_degenerate"

    if fh_upper < fh_lower:
        # Swap if mis-ordered
        fh_lower, fh_upper = fh_upper, fh_lower

    if np.isclose(fh_upper, fh_lower):
        return 0.5, "fh_degenerate"

    y = (j_observed - fh_lower) / (fh_upper - fh_lower)
    note: Optional[str] = None
    if y < 0.0:
        if clip_outside:
            y, note = 0.0, "fh_clip_low"
        else:
            note = "fh_below"
    elif y > 1.0:
        if clip_outside:
            y, note = 1.0, "fh_clip_high"
        else:
            note = "fh_above"
    return float(y), note


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _classify_delta(
    obs_value: float,
    baseline_value: float,
    *,
    threshold: float,
    lower_is_better: bool,
) -> str:
    """
    Generic 3-way classifier from a comparison of observed vs. baseline.

    If lower_is_better:
        obs < base − thr → "constructive"
        obs > base + thr → "destructive"
        else             → "independent"

    If higher_is_better:
        obs > base + thr → "constructive"
        obs < base − thr → "destructive"
        else             → "independent"
    """
    obs = float(obs_value); base = float(baseline_value); thr = float(threshold)
    if lower_is_better:
        if obs < base - thr:
            return "constructive"
        if obs > base + thr:
            return "destructive"
        return "independent"
    else:
        if obs > base + thr:
            return "constructive"
        if obs < base - thr:
            return "destructive"
        return "independent"


def _apply_multiple_comparison_correction(
    *,
    threshold_j: float,
    threshold_euclid: float,
    threshold_cost: float,
    fh_bound: float,
    method_count: int,
    correction: str = "none",
) -> Tuple[float, float, float, float]:
    """
    Apply a simple Bonferroni-style tightening of decision thresholds.

    Notes
    -----
    - For ΔJ / Euclidean / Cost thresholds: we *shrink* the allowed deviation
      by dividing by the number of methods (stricter classification).
    - For FH percentile, we *widen* the neutrality band by increasing the
      distance from 0.5 to the decision bound.

    Parameters
    ----------
    threshold_j, threshold_euclid, threshold_cost : float
        Base thresholds for decision deltas.
    fh_bound : float
        Base FH lower/higher decision bound (e.g., 0.4/0.6 → bound=0.4).
    method_count : int
        How many parallel alternatives we're judging (usually 3 here, excluding ΔJ
        if you treat ΔJ as the baseline; or 4 to include it).
    correction : {"none","bonferroni"}
        If "bonferroni", thresholds are tightened.

    Returns
    -------
    (thr_j, thr_e, thr_c, fh_bound_corrected)
    """
    if correction.lower() != "bonferroni" or method_count <= 1:
        return threshold_j, threshold_euclid, threshold_cost, fh_bound

    factor = float(method_count)
    thr_j = threshold_j / factor
    thr_e = threshold_euclid / factor
    thr_c = threshold_cost / factor

    # FH: widen neutrality band around 0.5
    # base band half-width = (0.5 - fh_bound). Make it factor times wider.
    half_width = max(0.0, 0.5 - fh_bound)
    widened = min(0.5, half_width * factor)
    fh_bound_corrected = 0.5 - widened  # bound decreases toward 0.0
    return thr_j, thr_e, thr_c, fh_bound_corrected


def _validate_record(cfg: str, rec: Dict[str, float]) -> None:
    """
    Sanity checks for a single configuration record.
    Raises ValueError with a helpful message on inconsistencies.
    """
    required = ["tpr", "fpr", "j_obs", "tpr_indep", "fpr_indep", "j_indep", "fh_lower", "fh_upper"]
    missing = [k for k in required if k not in rec]
    if missing:
        raise ValueError(f"[{cfg}] missing fields: {missing}")

    tpr_ind = float(rec["tpr_indep"])
    fpr_ind = float(rec["fpr_indep"])
    j_ind = float(rec["j_indep"])
    if abs((tpr_ind - fpr_ind) - j_ind) > 1e-6:
        raise ValueError(
            f"[{cfg}] j_indep inconsistent with (tpr_indep - fpr_indep): "
            f"{j_ind} vs {tpr_ind - fpr_ind}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_all_metrics(
    config_results: Dict[str, Dict],
    *,
    # thresholds (tune if needed; ΔJ=0.05 aligns with α≈0.05 practical band)
    threshold_j: float = 0.05,
    threshold_euclid: float = 0.02,
    threshold_cost: float = 0.02,
    fh_neutral_bound: float = 0.40,  # ⇒ neutral band [0.40, 0.60]
    # cost weights (only used by cost-weighted error)
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    # correction across multiple methods
    multiple_comparison_correction: str = "none",  # or "bonferroni"
    include_delta_j_as_method: bool = False,
) -> pd.DataFrame:
    """
    Run all metrics on a mapping of configurations and compare classifications.

    Parameters
    ----------
    config_results : Dict[str, Dict]
        See module docstring for required schema.
    threshold_j : float, default=0.05
        Practical significance threshold for ΔJ classification.
    threshold_euclid : float, default=0.02
        Minimal Euclidean distance improvement needed to call "constructive"
        (and worsening to call "destructive").
    threshold_cost : float, default=0.02
        Minimal cost difference needed to call "constructive"/"destructive".
    fh_neutral_bound : float, default=0.40
        Percentile bound for FH classification; neutral band is
        [fh_neutral_bound, 1 - fh_neutral_bound]. With default 0.40, neutral is
        [0.40, 0.60]; outside that → constructive/destructive.
    cost_fp, cost_fn : float, default=1.0
        Cost weights for cost-weighted error metric.
    multiple_comparison_correction : {"none","bonferroni"}, default="none"
        If "bonferroni", tightens thresholds across alternatives.
    include_delta_j_as_method : bool, default=False
        If True, include ΔJ in the multiple comparison count.

    Returns
    -------
    pandas.DataFrame
        Per-configuration scores and classifications for all metrics, plus
        consensus diagnostics.
    """
    if not config_results:
        return pd.DataFrame(
            columns=[
                "config_name",
                "tpr", "fpr", "j_obs",
                "tpr_indep", "fpr_indep", "j_indep",
                "euclidean_dist", "euclidean_dist_indep", "euclidean_delta", "euclidean_class",
                "cost_value", "cost_indep", "cost_delta", "cost_class",
                "fh_percentile", "fh_note", "fh_class",
                "youden_delta_j", "youden_class",
                "majority_size", "majority_label", "unanimous",
            ]
        )

    # How many alternative methods? (Euclid, Cost, FH) = 3
    # Optionally include ΔJ itself in the correction count.
    method_count = 3 + (1 if include_delta_j_as_method else 0)
    thr_j, thr_e, thr_c, fh_bound = _apply_multiple_comparison_correction(
        threshold_j=threshold_j,
        threshold_euclid=threshold_euclid,
        threshold_cost=threshold_cost,
        fh_bound=fh_neutral_bound,
        method_count=method_count,
        correction=multiple_comparison_correction,
    )
    fh_lo_bound = fh_bound
    fh_hi_bound = 1.0 - fh_bound

    rows: List[Dict[str, object]] = []

    for cfg_name, rec in config_results.items():
        _validate_record(cfg_name, rec)

        tpr = float(rec["tpr"])
        fpr = float(rec["fpr"])
        j_obs = float(rec["j_obs"])
        tpr_indep = float(rec["tpr_indep"])
        fpr_indep = float(rec["fpr_indep"])
        j_indep = float(rec["j_indep"])
        fh_lower = float(rec["fh_lower"])
        fh_upper = float(rec["fh_upper"])

        # ΔJ (higher is better)
        delta_j = independence_delta_j(j_obs, j_indep)
        youden_class = (
            "constructive" if delta_j > thr_j
            else "destructive" if delta_j < -thr_j
            else "independent"
        )

        # Euclidean (lower is better) with TRUE independence baseline
        e_obs = euclidean_distance_index(tpr, fpr)
        e_base = euclidean_distance_index(tpr_indep, fpr_indep)
        e_delta = e_obs - e_base
        euclidean_class = _classify_delta(
            obs_value=e_obs, baseline_value=e_base,
            threshold=thr_e, lower_is_better=True,
        )

        # Cost-weighted (lower is better) with TRUE independence baseline
        c_obs = cost_weighted_error(tpr, fpr, cost_fp, cost_fn)
        c_base = cost_weighted_error(tpr_indep, fpr_indep, cost_fp, cost_fn)
        c_delta = c_obs - c_base
        cost_class = _classify_delta(
            obs_value=c_obs, baseline_value=c_base,
            threshold=thr_c, lower_is_better=True,
        )

        # FH percentile (constructive if clearly upper-side; destructive if lower-side)
        fh_pct, fh_note = fh_envelope_percentile(j_obs, fh_lower, fh_upper, clip_outside=True)
        if fh_pct > fh_hi_bound:
            fh_class = "constructive"
        elif fh_pct < fh_lo_bound:
            fh_class = "destructive"
        else:
            fh_class = "independent"

        # Consensus
        labels = [youden_class, euclidean_class, cost_class, fh_class]
        label_counts = pd.Series(labels).value_counts()
        majority_label = str(label_counts.idxmax())
        majority_size = int(label_counts.max())
        unanimous = bool(majority_size == len(labels))

        rows.append({
            "config_name": cfg_name,
            "tpr": tpr, "fpr": fpr, "j_obs": j_obs,
            "tpr_indep": tpr_indep, "fpr_indep": fpr_indep, "j_indep": j_indep,

            "euclidean_dist": e_obs,
            "euclidean_dist_indep": e_base,
            "euclidean_delta": e_delta,
            "euclidean_class": euclidean_class,

            "cost_value": c_obs,
            "cost_indep": c_base,
            "cost_delta": c_delta,
            "cost_class": cost_class,

            "fh_percentile": fh_pct,
            "fh_note": fh_note,
            "fh_class": fh_class,

            "youden_delta_j": delta_j,
            "youden_class": youden_class,

            "majority_size": majority_size,
            "majority_label": majority_label,
            "unanimous": unanimous,
        })

    return pd.DataFrame(rows)


def cohen_kappa(method1_classes: Iterable[str], method2_classes: Iterable[str]) -> float:
    """
    Cohen’s κ agreement between two classification arrays.

    Parameters
    ----------
    method1_classes : Iterable[str]
        e.g., ["constructive", "independent", ...]
    method2_classes : Iterable[str]
        Same length, same label set.

    Returns
    -------
    float
        κ in [−1, 1].  1.0 = perfect agreement; 0.0 = chance-level.
    """
    return float(cohen_kappa_score(list(method1_classes), list(method2_classes)))


def kappa_matrix(df: pd.DataFrame, class_cols: List[str]) -> pd.DataFrame:
    """
    Pairwise κ matrix over multiple method classification columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of compare_all_metrics.
    class_cols : List[str]
        Column names containing categorical labels.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix of κ scores (diagonal = 1.0).
    """
    cols = list(class_cols)
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if j <= i:
                continue
            k = cohen_kappa(df[ci].astype(str), df[cj].astype(str))
            out.loc[ci, cj] = k
            out.loc[cj, ci] = k
    return out


def analyze_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only configurations with non-unanimous classifications, along with
    a compact “pattern” column to help with failure analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of compare_all_metrics.

    Returns
    -------
    pandas.DataFrame
        Subset where unanimous == False, with a 'pattern' column like:
        "ΔJ:C | E:D | C:I | FH:I"
    """
    if df.empty:
        return df.copy()

    sub = df.loc[~df["unanimous"]].copy()
    if sub.empty:
        return sub

    patterns = []
    for _, r in sub.iterrows():
        parts = [
            f"ΔJ:{r['youden_class'][0].upper()}",
            f"E:{r['euclidean_class'][0].upper()}",
            f"C:{r['cost_class'][0].upper()}",
            f"FH:{r['fh_class'][0].upper()}",
        ]
        patterns.append(" | ".join(parts))
    sub["pattern"] = patterns
    return sub


# ---------------------------------------------------------------------------
# (Optional) Tiny demo when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = {
        "cfg_demo": {
            "tpr": 0.90, "fpr": 0.08, "j_obs": 0.82,
            "tpr_indep": 0.87, "fpr_indep": 0.10, "j_indep": 0.77,
            "fh_lower": 0.70, "fh_upper": 0.90,
        }
    }
    df = compare_all_metrics(
        demo,
        threshold_j=0.05,
        threshold_euclid=0.02,
        threshold_cost=0.02,
        fh_neutral_bound=0.40,
        multiple_comparison_correction="none",
    )
    print(df.to_string(index=False))

    kappas = kappa_matrix(df, ["youden_class", "euclidean_class", "cost_class", "fh_class"])
    print("\nκ matrix:\n", kappas)
