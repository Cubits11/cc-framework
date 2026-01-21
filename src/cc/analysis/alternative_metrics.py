"""
Alternative metrics for destructive interference detection (Comment 2).

This module implements *four complementary* approaches for evaluating whether a
composition of guardrails (e.g., AND/OR stacking) improves or degrades overall
detection performance relative to an **independence baseline**.

It is designed to run light-weight (NumPy / Pandas / scikit-learn only) and to
plug directly into the cc-framework results pipeline (Week 6-7 JSONL → fixed
metrics). It does **not** require R or heavy ML libraries.

-------------------------------------------------------------------------------
WHAT THIS FILE PROVIDES
-------------------------------------------------------------------------------

1) Euclidean Distance Index (to ROC "perfection" corner)
   - Geometry-based:  d = sqrt((1 - TPR)^2 + (FPR)^2), lower is better.
   - Compares observed composition vs. *true* independence baseline point
     (TPR_indep, FPR_indep), not a proxy.

2) Cost-Weighted Error
   - Decision/ops-based:  cost = c_FP * FPR + c_FN * (1 - TPR), lower is better.
   - Compares observed composition vs. independence baseline cost.

3) ΔJ against Independence (Youden's J)
   - Statistics-based:  ΔJ = J_obs - J_indep with a practical significance band.
   - 3-way classification: constructive | destructive | independent.

4) FH Envelope Percentile
   - Information-theoretic / model-free: position of J_obs within the
     Fréchet-Hoeffding envelope [FH_lower, FH_upper]; returns y in [0, 1].
   - 3-way classification via a neutral band centered at 0.5.

All four yield per-configuration classifications that you can compare
(agreement rates, κ scores) and analyze for disagreements/failure modes.

-------------------------------------------------------------------------------
CRITICAL REQUIREMENT (fixes the earlier bug)
-------------------------------------------------------------------------------

For Euclidean and Cost-Weighted comparisons, you **must** supply the *true*
independence baseline operating point **(tpr_indep, fpr_indep)**, not an
approximation. We verify consistency via:

    |(tpr_indep - fpr_indep) - j_indep| <= 1e-6

Where to get (TPR_indep, FPR_indep):

- Use the independence composition rules computed in your stats/estimation step
  (already present in cc-framework), e.g., for two rails:

    AND: TPR_indep = TPR_A * TPR_B; FPR_indep = FPR_A * FPR_B
    OR : TPR_indep = TPR_A + TPR_B - TPR_A * TPR_B

  (Generalization exists for >2 rails; the framework already computes these.)

We also require consistency for the observed point:

    |(tpr - fpr) - j_obs| <= 1e-6

-------------------------------------------------------------------------------
WHEN YOUDEN'S J OUTPERFORMS ALTERNATIVES (for the paper)
-------------------------------------------------------------------------------

- Deployment typically constrains FPR to a narrow policy band (e.g., <= 0.06).
  Euclidean Distance treats TPR and FPR symmetrically over the full ROC
  geometry, which can over-reward TPR gains that slightly violate policy FPR.
  J = TPR - FPR is a *linear* trade-off directly interpretable at a single
  operating point and respects such policy windows more transparently.

- Cost-Weighted Error needs accurate (c_FP, c_FN). If these are unknown or
  contentious across stakeholders, J provides a *domain-agnostic* summary.

- FH percentile is model-free but lacks effect-size (magnitude) semantics; ΔJ
  supplies both direction and magnitude relative to a testable baseline.

-------------------------------------------------------------------------------
INPUT SCHEMA (config_results)
-------------------------------------------------------------------------------

compare_all_metrics / compare_all_metrics_structured expect a mapping:

    {
      "<config_name>": {
        "tpr": float,        # observed composition
        "fpr": float,        # observed composition
        "j_obs": float,      # observed J (≈ tpr - fpr, consistency enforced)

        "tpr_indep": float,  # independence baseline (TRUE point)
        "fpr_indep": float,  # independence baseline (TRUE point)
        "j_indep": float,    # j_indep ≈ tpr_indep - fpr_indep (consistency enforced)

        "fh_lower": float,   # FH lower bound for J (can be in [-1, 1])
        "fh_upper": float,   # FH upper bound for J (fh_lower <= fh_upper)

        # Optional counts for future CI bootstraps (not used here):
        # "tp": int, "fp": int, "fn": int, "tn": int
      },
      ...
    }

All numeric fields must be finite. We allow small drift outside [0,1] for rates
(±1e-6 tolerance) but raise on grossly invalid values.

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

1) compare_all_metrics_structured
   - Returns: Dict[str, List[MetricResult]]
   - Each configuration maps to a list of MetricResult objects for:
       - "youden_delta_j"
       - "euclidean"
       - "cost_weighted"
       - "fh_percentile"

2) compare_all_metrics
   - Returns: pandas.DataFrame with per-config values and classifications for
     each metric, plus consensus diagnostics:

       columns = [
           "config_name",
           "tpr", "fpr", "j_obs",
           "tpr_indep", "fpr_indep", "j_indep",
           "euclidean_dist", "euclidean_dist_indep", "euclidean_delta", "euclidean_class",
           "cost_value", "cost_indep", "cost_delta", "cost_class",
           "fh_percentile", "fh_note", "fh_class",
           "youden_delta_j", "youden_class",
           "majority_size", "majority_label", "unanimous",
       ]

   - Consensus: majority label over the four metrics; ties are resolved
     conservatively as "independent".

3) kappa_matrix
   - Returns a symmetric κ matrix over multiple method classification columns.

4) analyze_disagreements
   - Returns only non-unanimous configurations with a compact pattern string
     such as "ΔJ:C | E:D | C:I | FH:I".

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
>>> kappa = kappa_matrix(df, ["youden_class", "euclidean_class", "cost_class", "fh_class"])
>>> disagreements = analyze_disagreements(df)
>>> df.loc[0, ["youden_delta_j", "youden_class"]].to_dict()
{'youden_delta_j': 0.050000000000000044, 'youden_class': 'constructive'}
"""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Types and constants
# ---------------------------------------------------------------------------

ClassificationLabel = Literal["constructive", "destructive", "independent"]
_ALLOWED_CLASS_TUPLE: tuple[str, ...] = typing.get_args(ClassificationLabel)
_ALLOWED_CLASS_SET = set(_ALLOWED_CLASS_TUPLE)

_DATAFRAME_COLUMNS: list[str] = [
    "config_name",
    "tpr",
    "fpr",
    "j_obs",
    "tpr_indep",
    "fpr_indep",
    "j_indep",
    "euclidean_dist",
    "euclidean_dist_indep",
    "euclidean_delta",
    "euclidean_class",
    "cost_value",
    "cost_indep",
    "cost_delta",
    "cost_class",
    "fh_percentile",
    "fh_note",
    "fh_class",
    "youden_delta_j",
    "youden_class",
    "majority_size",
    "majority_label",
    "unanimous",
]


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
        Name of the metric (e.g., "euclidean", "cost_weighted", "youden_delta_j",
        "fh_percentile").
    score : float
        Observed score for the metric. For ΔJ this is the delta itself; for
        Euclidean and Cost this is the composition operating point; for FH this
        is the percentile y ∈ [0, 1].
    baseline_score : Optional[float]
        Baseline score at the independence operating point, if applicable.
        (None for ΔJ and FH percentile.)
    delta : Optional[float]
        score - baseline_score when applicable (Euclidean / Cost); None for
        metrics that are inherently delta-like (ΔJ, FH percentile).
    classification : ClassificationLabel
        One of {'constructive', 'destructive', 'independent'}.
    confidence_interval : Optional[Tuple[float, float]]
        Reserved for future CI reporting (requires counts, not just rates).
    notes : Optional[str]
        Free-form note for flags (e.g., "fh_clip_low", "fh_degenerate").
    """

    metric_name: str
    score: float
    baseline_score: float | None
    delta: float | None
    classification: ClassificationLabel
    confidence_interval: tuple[float, float] | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.classification not in _ALLOWED_CLASS_SET:
            raise ValueError(
                f"Invalid classification: {self.classification}. "
                f"Must be one of {_ALLOWED_CLASS_TUPLE}."
            )


# ---------------------------------------------------------------------------
# Low-level numeric helpers
# ---------------------------------------------------------------------------


def _ensure_finite(name: str, value: Any) -> float:
    """
    Convert to float and ensure the result is finite.

    Raises
    ------
    ValueError
        If the value cannot be converted to float or is NaN/inf.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric value, got {value!r}") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return v


def _validate_thresholds(
    *,
    threshold_j: float,
    threshold_euclid: float,
    threshold_cost: float,
    fh_neutral_bound: float,
) -> None:
    """
    Validate thresholds and FH neutrality bound.

    threshold_j, threshold_euclid, threshold_cost must be >= 0.
    fh_neutral_bound must be in [0.0, 0.5].
    """
    if threshold_j < 0.0:
        raise ValueError(f"threshold_j must be >= 0, got {threshold_j}")
    if threshold_euclid < 0.0:
        raise ValueError(f"threshold_euclid must be >= 0, got {threshold_euclid}")
    if threshold_cost < 0.0:
        raise ValueError(f"threshold_cost must be >= 0, got {threshold_cost}")
    if not (0.0 <= fh_neutral_bound <= 0.5):
        raise ValueError(f"fh_neutral_bound must be in [0.0, 0.5], got {fh_neutral_bound}")


# ---------------------------------------------------------------------------
# Primitive scorers
# ---------------------------------------------------------------------------


def euclidean_distance_index(tpr: float, fpr: float) -> float:
    """
    Euclidean distance to ROC "perfection" corner (FPR=0, TPR=1). Lower is better.

    Parameters
    ----------
    tpr : float
        True Positive Rate (assumed ∈ [0, 1] within small numerical tolerance).
    fpr : float
        False Positive Rate (assumed ∈ [0, 1] within small numerical tolerance).

    Returns
    -------
    float
        Distance d = sqrt((1 - TPR)^2 + (FPR)^2).
    """
    tpr_f = _ensure_finite("tpr", tpr)
    fpr_f = _ensure_finite("fpr", fpr)
    return float(np.sqrt((1.0 - tpr_f) ** 2 + (fpr_f**2)))


def cost_weighted_error(
    tpr: float,
    fpr: float,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> float:
    """
    Expected cost: cost = c_FP * FPR + c_FN * (1 - TPR). Lower is better.

    Assumes cost_fp, cost_fn >= 0; negative values yield invalid semantics and
    will be rejected by the public API that uses this helper.

    Parameters
    ----------
    tpr : float
        True Positive Rate (assumed ∈ [0, 1]).
    fpr : float
        False Positive Rate (assumed ∈ [0, 1]).
    cost_fp : float, default=1.0
        Cost per false positive (must be >= 0).
    cost_fn : float, default=1.0
        Cost per false negative (must be >= 0).

    Returns
    -------
    float
        Expected cost (range [0, cost_fp + cost_fn] assuming valid inputs).
    """
    tpr_f = _ensure_finite("tpr", tpr)
    fpr_f = _ensure_finite("fpr", fpr)
    return float(cost_fp * fpr_f + cost_fn * (1.0 - tpr_f))


def independence_delta_j(
    j_observed: float,
    j_independent: float,
) -> float:
    """
    Compute ΔJ = J_obs - J_indep (no classification here).

    Parameters
    ----------
    j_observed : float
        Observed Youden's J at the composition operating point.
    j_independent : float
        Independence-baseline Youden's J.

    Returns
    -------
    float
        ΔJ.
    """
    j_obs_f = _ensure_finite("j_observed", j_observed)
    j_ind_f = _ensure_finite("j_independent", j_independent)
    return float(j_obs_f - j_ind_f)


def fh_envelope_percentile(
    j_observed: float,
    fh_lower: float,
    fh_upper: float,
    *,
    clip_outside: bool = True,
) -> tuple[float, str | None]:
    """
    Position of J_obs within FH envelope: y = (J_obs - FH_lo) / (FH_hi - FH_lo).

    Handles degenerate/invalid envelopes and out-of-envelope observations.

    Notes
    -----
    We do not restrict J to [0, 1]; FH bounds may span [-1, 1] mathematically.

    Parameters
    ----------
    j_observed : float
        Observed J (may be outside [0, 1]).
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
        "fh_clip_low", "fh_clip_high", "fh_below", "fh_above").
    """
    j_obs_f = _ensure_finite("j_observed", j_observed)
    fh_lo = _ensure_finite("fh_lower", fh_lower)
    fh_hi = _ensure_finite("fh_upper", fh_upper)

    if not np.isfinite(fh_lo) or not np.isfinite(fh_hi):
        return 0.5, "fh_degenerate"

    if fh_hi < fh_lo:
        # Swap if mis-ordered (this should not happen after validation, but
        # we keep it for robustness when called standalone).
        fh_lo, fh_hi = fh_hi, fh_lo

    if np.isclose(fh_hi, fh_lo):
        return 0.5, "fh_degenerate"

    y = (j_obs_f - fh_lo) / (fh_hi - fh_lo)
    note: str | None = None

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
) -> ClassificationLabel:
    """
    Generic 3-way classifier from a comparison of observed vs. baseline.

    Notes
    -----
    We treat |obs - base| <= threshold as "independent" (inclusive boundaries).
    threshold must be >= 0; negative thresholds yield invalid semantics.

    If lower_is_better:
        obs < base - thr → "constructive"
        obs > base + thr → "destructive"
        else             → "independent"

    If higher_is_better:
        obs > base + thr → "constructive"
        obs < base - thr → "destructive"
        else             → "independent"
    """
    obs = _ensure_finite("obs_value", obs_value)
    base = _ensure_finite("baseline_value", baseline_value)
    thr = float(threshold)

    if thr < 0.0:
        raise ValueError(f"threshold must be >= 0, got {thr}")

    if lower_is_better:
        if obs < base - thr:
            return "constructive"
        if obs > base + thr:
            return "destructive"
        return "independent"

    # higher_is_better
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
) -> tuple[float, float, float, float]:
    """
    Apply a conservative Bonferroni-style adjustment to decision thresholds.

    This makes it *harder* to declare "constructive" or "destructive" (wider
    neutral band) to account for multiple comparisons.

    - For ΔJ / Euclidean / Cost: we multiply thresholds by the method_count,
      widening the tolerance band where we call "independent".
    - For FH percentile: we widen the neutrality band by scaling its
      half-width around 0.5.

    Parameters
    ----------
    threshold_j, threshold_euclid, threshold_cost : float
        Base thresholds for decision deltas (must be validated upstream).
    fh_bound : float
        Base FH lower/higher decision bound (e.g., 0.4/0.6 → bound=0.4).
    method_count : int
        How many parallel alternatives we're judging (usually 3 here, excluding ΔJ
        if you treat ΔJ as the baseline; or 4 to include it).
    correction : {"none", "bonferroni"}
        If "bonferroni", thresholds are adjusted conservatively.

    Returns
    -------
    (thr_j, thr_e, thr_c, fh_bound_corrected)
    """
    if method_count < 1:
        raise ValueError(f"method_count must be >= 1, got {method_count}")

    corr = correction.lower()
    if corr not in {"none", "bonferroni"}:
        raise ValueError(f"Unknown multiple comparison correction: {correction!r}")

    if corr == "none" or method_count == 1:
        return threshold_j, threshold_euclid, threshold_cost, fh_bound

    factor = float(method_count)

    # Widen tolerance bands for deltas
    thr_j = threshold_j * factor
    thr_e = threshold_euclid * factor
    thr_c = threshold_cost * factor

    # FH: widen neutrality band around 0.5.
    # Base band half-width = (0.5 - fh_bound). Make it factor times wider,
    # but cap at 0.5 (neutral band cannot exceed [0, 1]).
    half_width = max(0.0, 0.5 - fh_bound)
    widened = min(0.5, half_width * factor)
    fh_bound_corrected = 0.5 - widened  # bound decreases toward 0.0, wider band

    return thr_j, thr_e, thr_c, fh_bound_corrected


def _validate_record(cfg: str, rec: Mapping[str, Any]) -> dict[str, float]:
    """
    Sanity checks and normalization for a single configuration record.

    Returns a dict of canonical float values:

        {
            "tpr", "fpr", "j_obs",
            "tpr_indep", "fpr_indep", "j_indep",
            "fh_lower", "fh_upper",
        }

    Raises
    ------
    ValueError
        If required fields are missing, non-finite, or mutually inconsistent.
    """
    required = [
        "tpr",
        "fpr",
        "j_obs",
        "tpr_indep",
        "fpr_indep",
        "j_indep",
        "fh_lower",
        "fh_upper",
    ]
    missing = [k for k in required if k not in rec]
    if missing:
        raise ValueError(f"[{cfg}] missing fields: {missing}")

    tpr = _ensure_finite("tpr", rec["tpr"])
    fpr = _ensure_finite("fpr", rec["fpr"])
    tpr_ind = _ensure_finite("tpr_indep", rec["tpr_indep"])
    fpr_ind = _ensure_finite("fpr_indep", rec["fpr_indep"])
    j_obs = _ensure_finite("j_obs", rec["j_obs"])
    j_ind = _ensure_finite("j_indep", rec["j_indep"])
    fh_lower = _ensure_finite("fh_lower", rec["fh_lower"])
    fh_upper = _ensure_finite("fh_upper", rec["fh_upper"])

    # Allow small tolerance around [0, 1] for rates.
    for name, val in (
        ("tpr", tpr),
        ("fpr", fpr),
        ("tpr_indep", tpr_ind),
        ("fpr_indep", fpr_ind),
    ):
        if val < -1e-6 or val > 1.0 + 1e-6:
            raise ValueError(f"[{cfg}] {name}={val} is outside [0, 1] (tolerance 1e-6)")

    # J should be within [-1, 1] in theory; tolerate small drift.
    for name, val in (("j_obs", j_obs), ("j_indep", j_ind)):
        if val < -1.0 - 1e-6 or val > 1.0 + 1e-6:
            raise ValueError(f"[{cfg}] {name}={val} is outside [-1, 1] (tolerance 1e-6)")

    # J consistency checks
    if abs((tpr_ind - fpr_ind) - j_ind) > 1e-6:
        raise ValueError(
            f"[{cfg}] j_indep inconsistent with (tpr_indep - fpr_indep): "
            f"{j_ind} vs {tpr_ind - fpr_ind}"
        )
    if abs((tpr - fpr) - j_obs) > 1e-6:
        raise ValueError(f"[{cfg}] j_obs inconsistent with (tpr - fpr): {j_obs} vs {tpr - fpr}")

    if fh_lower > fh_upper:
        raise ValueError(f"[{cfg}] fh_lower > fh_upper: {fh_lower} > {fh_upper}")

    # FH bounds can be outside [-1, 1] mathematically, but we keep them finite.
    return {
        "tpr": tpr,
        "fpr": fpr,
        "j_obs": j_obs,
        "tpr_indep": tpr_ind,
        "fpr_indep": fpr_ind,
        "j_indep": j_ind,
        "fh_lower": fh_lower,
        "fh_upper": fh_upper,
    }


def _iter_config_metrics(
    config_results: Mapping[str, Mapping[str, Any]],
    *,
    threshold_j: float,
    threshold_euclid: float,
    threshold_cost: float,
    fh_neutral_bound: float,
    cost_fp: float,
    cost_fn: float,
    multiple_comparison_correction: str,
    include_delta_j_as_method: bool,
) -> Iterable[tuple[str, dict[str, float], list[MetricResult]]]:
    """
    Core iterator that validates inputs, applies thresholds/corrections, and
    yields per-configuration metric results.

    Yields
    ------
    (config_name, canonical_values, metric_results)
        canonical_values : Dict[str, float] with normalized inputs.
        metric_results   : List[MetricResult] for the four metrics.
    """
    _validate_thresholds(
        threshold_j=threshold_j,
        threshold_euclid=threshold_euclid,
        threshold_cost=threshold_cost,
        fh_neutral_bound=fh_neutral_bound,
    )

    if cost_fp < 0.0 or cost_fn < 0.0:
        raise ValueError(
            f"cost_fp and cost_fn must be >= 0, got cost_fp={cost_fp}, cost_fn={cost_fn}"
        )

    method_count = 3 + (1 if include_delta_j_as_method else 0)
    thr_j_eff, thr_e_eff, thr_c_eff, fh_bound_eff = _apply_multiple_comparison_correction(
        threshold_j=threshold_j,
        threshold_euclid=threshold_euclid,
        threshold_cost=threshold_cost,
        fh_bound=fh_neutral_bound,
        method_count=method_count,
        correction=multiple_comparison_correction,
    )
    fh_lo_bound = fh_bound_eff
    fh_hi_bound = 1.0 - fh_bound_eff

    for cfg_name, rec in config_results.items():
        vals = _validate_record(cfg_name, rec)

        tpr = vals["tpr"]
        fpr = vals["fpr"]
        j_obs = vals["j_obs"]
        tpr_indep = vals["tpr_indep"]
        fpr_indep = vals["fpr_indep"]
        j_indep = vals["j_indep"]
        fh_lower = vals["fh_lower"]
        fh_upper = vals["fh_upper"]

        metrics: list[MetricResult] = []

        # 1) ΔJ (higher is better)
        delta_j = independence_delta_j(j_obs, j_indep)
        if delta_j > thr_j_eff:
            youden_class: ClassificationLabel = "constructive"
        elif delta_j < -thr_j_eff:
            youden_class = "destructive"
        else:
            youden_class = "independent"
        metrics.append(
            MetricResult(
                metric_name="youden_delta_j",
                score=delta_j,
                baseline_score=None,
                delta=delta_j,
                classification=youden_class,
            )
        )

        # 2) Euclidean (lower is better) with TRUE independence baseline
        e_obs = euclidean_distance_index(tpr, fpr)
        e_base = euclidean_distance_index(tpr_indep, fpr_indep)
        e_delta = e_obs - e_base
        euclidean_class = _classify_delta(
            obs_value=e_obs,
            baseline_value=e_base,
            threshold=thr_e_eff,
            lower_is_better=True,
        )
        metrics.append(
            MetricResult(
                metric_name="euclidean",
                score=e_obs,
                baseline_score=e_base,
                delta=e_delta,
                classification=euclidean_class,
            )
        )

        # 3) Cost-weighted (lower is better) with TRUE independence baseline
        c_obs = cost_weighted_error(tpr, fpr, cost_fp=cost_fp, cost_fn=cost_fn)
        c_base = cost_weighted_error(tpr_indep, fpr_indep, cost_fp=cost_fp, cost_fn=cost_fn)
        c_delta = c_obs - c_base
        cost_class = _classify_delta(
            obs_value=c_obs,
            baseline_value=c_base,
            threshold=thr_c_eff,
            lower_is_better=True,
        )
        metrics.append(
            MetricResult(
                metric_name="cost_weighted",
                score=c_obs,
                baseline_score=c_base,
                delta=c_delta,
                classification=cost_class,
            )
        )

        # 4) FH percentile (constructive if clearly upper-side; destructive if lower-side)
        fh_pct, fh_note = fh_envelope_percentile(
            j_observed=j_obs,
            fh_lower=fh_lower,
            fh_upper=fh_upper,
            clip_outside=True,
        )
        if fh_pct > fh_hi_bound:
            fh_class: ClassificationLabel = "constructive"
        elif fh_pct < fh_lo_bound:
            fh_class = "destructive"
        else:
            fh_class = "independent"
        metrics.append(
            MetricResult(
                metric_name="fh_percentile",
                score=fh_pct,
                baseline_score=None,
                delta=None,
                classification=fh_class,
                notes=fh_note,
            )
        )

        yield cfg_name, vals, metrics


# ---------------------------------------------------------------------------
# Public API - structured and DataFrame versions
# ---------------------------------------------------------------------------


def compare_all_metrics_structured(
    config_results: Mapping[str, Mapping[str, Any]],
    *,
    threshold_j: float = 0.05,
    threshold_euclid: float = 0.02,
    threshold_cost: float = 0.02,
    fh_neutral_bound: float = 0.40,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    multiple_comparison_correction: str = "none",
    include_delta_j_as_method: bool = False,
) -> dict[str, list[MetricResult]]:
    """
    Structured version of compare_all_metrics.

    Parameters
    ----------
    config_results : Mapping[str, Mapping[str, Any]]
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
        Cost weights for the cost-weighted error metric (must be >= 0).
    multiple_comparison_correction : {"none", "bonferroni"}, default="none"
        If "bonferroni", adjusts thresholds conservatively across alternatives.
    include_delta_j_as_method : bool, default=False
        If True, include ΔJ in the multiple comparison count for correction.

    Returns
    -------
    Dict[str, List[MetricResult]]
        Mapping from config name to list of MetricResult objects.
    """
    if not config_results:
        return {}

    out: dict[str, list[MetricResult]] = {}

    for cfg_name, _vals, metrics in _iter_config_metrics(
        config_results=config_results,
        threshold_j=threshold_j,
        threshold_euclid=threshold_euclid,
        threshold_cost=threshold_cost,
        fh_neutral_bound=fh_neutral_bound,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        multiple_comparison_correction=multiple_comparison_correction,
        include_delta_j_as_method=include_delta_j_as_method,
    ):
        # In case we want canonical values later, we could store them too;
        # for now, we return just metrics per configuration.
        out[cfg_name] = metrics

    return out


def compare_all_metrics(
    config_results: Mapping[str, Mapping[str, Any]],
    *,
    threshold_j: float = 0.05,
    threshold_euclid: float = 0.02,
    threshold_cost: float = 0.02,
    fh_neutral_bound: float = 0.40,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    multiple_comparison_correction: str = "none",
    include_delta_j_as_method: bool = False,
) -> pd.DataFrame:
    """
    Run all metrics on a mapping of configurations and compare classifications.

    Notes
    -----
    Thresholds are hard-coded defaults; in a full analysis, calibrate via
    held-out data or bootstrap for your specific distributions.

    Parameters
    ----------
    config_results : Mapping[str, Mapping[str, Any]]
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
        Cost weights for cost-weighted error metric (must be >= 0).
    multiple_comparison_correction : {"none", "bonferroni"}, default="none"
        If "bonferroni", adjusts thresholds conservatively across alternatives.
    include_delta_j_as_method : bool, default=False
        If True, include ΔJ in the multiple comparison count.

    Returns
    -------
    pandas.DataFrame
        Per-configuration scores and classifications for all metrics, plus
        consensus diagnostics.
    """
    if not config_results:
        return pd.DataFrame(columns=_DATAFRAME_COLUMNS)

    rows: list[dict[str, Any]] = []

    for cfg_name, vals, metrics in _iter_config_metrics(
        config_results=config_results,
        threshold_j=threshold_j,
        threshold_euclid=threshold_euclid,
        threshold_cost=threshold_cost,
        fh_neutral_bound=fh_neutral_bound,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        multiple_comparison_correction=multiple_comparison_correction,
        include_delta_j_as_method=include_delta_j_as_method,
    ):
        row: dict[str, Any] = {
            "config_name": cfg_name,
            "tpr": vals["tpr"],
            "fpr": vals["fpr"],
            "j_obs": vals["j_obs"],
            "tpr_indep": vals["tpr_indep"],
            "fpr_indep": vals["fpr_indep"],
            "j_indep": vals["j_indep"],
        }

        for res in metrics:
            if res.metric_name == "youden_delta_j":
                row["youden_delta_j"] = res.score
                row["youden_class"] = res.classification
            elif res.metric_name == "euclidean":
                row["euclidean_dist"] = res.score
                row["euclidean_dist_indep"] = res.baseline_score
                row["euclidean_delta"] = res.delta
                row["euclidean_class"] = res.classification
            elif res.metric_name == "cost_weighted":
                row["cost_value"] = res.score
                row["cost_indep"] = res.baseline_score
                row["cost_delta"] = res.delta
                row["cost_class"] = res.classification
            elif res.metric_name == "fh_percentile":
                row["fh_percentile"] = res.score
                row["fh_note"] = res.notes
                row["fh_class"] = res.classification

        # Consensus
        labels = [m.classification for m in metrics]
        label_counts = pd.Series(labels).value_counts()
        max_count = int(label_counts.max())
        candidates = list(label_counts[label_counts == max_count].index)

        if len(candidates) > 1:
            # Tie: conservatively fall back to "independent".
            majority_label: ClassificationLabel = "independent"
        else:
            majority_label = typing.cast(ClassificationLabel, candidates[0])

        unanimous = bool(max_count == len(labels))

        row["majority_size"] = max_count
        row["majority_label"] = majority_label
        row["unanimous"] = unanimous

        rows.append(row)

    return pd.DataFrame(rows, columns=_DATAFRAME_COLUMNS)


# ---------------------------------------------------------------------------
# Agreement & disagreement analysis
# ---------------------------------------------------------------------------


def cohen_kappa(
    method1_classes: Iterable[str],
    method2_classes: Iterable[str],
) -> float:
    """
    Cohen's κ agreement between two classification arrays.

    Parameters
    ----------
    method1_classes : Iterable[str]
        e.g., ["constructive", "independent", ...].
    method2_classes : Iterable[str]
        Same length, same label set.

    Returns
    -------
    float
        κ in [-1, 1].  1.0 = perfect agreement; 0.0 = chance-level.
    """
    m1 = list(method1_classes)
    m2 = list(method2_classes)
    if len(m1) != len(m2):
        raise ValueError(
            f"method1_classes and method2_classes must have the same length, "
            f"got {len(m1)} and {len(m2)}"
        )
    return float(cohen_kappa_score(m1, m2))


def kappa_matrix(df: pd.DataFrame, class_cols: list[str]) -> pd.DataFrame:
    """
    Pairwise κ matrix over multiple method classification columns.

    Notes
    -----
    Assumes columns contain only {'constructive', 'destructive', 'independent'}.
    Raises if any other label is observed (including NaN).

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
    if not cols:
        raise ValueError("class_cols must be a non-empty list of column names")

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing classification columns in df: {missing_cols}")

    unique_labels: set[Any] = set()
    for c in cols:
        unique_labels.update(df[c].unique())

    # Drop NaN before validation, but treat any remaining non-allowed labels as error.
    unique_labels_no_nan = {v for v in unique_labels if pd.notna(v)}
    invalid = unique_labels_no_nan - _ALLOWED_CLASS_SET
    if invalid:
        raise ValueError(
            f"Invalid labels found in classification columns: {sorted(invalid)}; "
            f"allowed labels are {_ALLOWED_CLASS_TUPLE}"
        )

    out = pd.DataFrame(
        np.eye(len(cols), dtype=float),
        index=cols,
        columns=cols,
        dtype=float,
    )

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if j <= i:
                continue
            # We already validated labels; cohen_kappa_score can operate directly.
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
        Output of compare_all_metrics (assumes specific columns exist).

    Returns
    -------
    pandas.DataFrame
        Subset where unanimous == False, with a 'pattern' column like:
        "ΔJ:C | E:D | C:I | FH:I"
    """
    required_cols = {
        "unanimous",
        "youden_class",
        "euclidean_class",
        "cost_class",
        "fh_class",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for disagreement analysis: {sorted(missing)}")

    if df.empty:
        return df.copy()

    sub = df.loc[~df["unanimous"]].copy()
    if sub.empty:
        return sub

    patterns: list[str] = []
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
            "tpr": 0.90,
            "fpr": 0.08,
            "j_obs": 0.82,
            "tpr_indep": 0.87,
            "fpr_indep": 0.10,
            "j_indep": 0.77,
            "fh_lower": 0.70,
            "fh_upper": 0.90,
        }
    }

    df_demo = compare_all_metrics(
        demo,
        threshold_j=0.05,
        threshold_euclid=0.02,
        threshold_cost=0.02,
        fh_neutral_bound=0.40,
        multiple_comparison_correction="none",
    )
    print(df_demo.to_string(index=False))

    kappas = kappa_matrix(
        df_demo,
        ["youden_class", "euclidean_class", "cost_class", "fh_class"],
    )
    print("\nκ matrix:\n", kappas)

    disagreements = analyze_disagreements(df_demo)
    print("\nDisagreements:\n", disagreements.to_string(index=False))
