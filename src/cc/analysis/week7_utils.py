"""Week 7 statistical helpers for the CC framework.

The goal of this module is to factor out the mathematical and statistical
operations that are shared by the Week 7 execution scripts.  It provides
numerically stable implementations of

* Wilson score confidence intervals for binomial proportions,
* Fréchet–Hoeffding envelopes for OR/AND guardrail compositions,
* Independence baselines for multi-rail systems, and
* BCa bootstrap intervals for derived statistics (ΔJ, J, etc.).

The routines are deliberately self contained so they can be exercised in
unit tests without having to execute the full experiment pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Binomial confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WilsonInterval:
    """Wilson score interval for a binomial proportion."""

    lower: float
    upper: float
    center: float
    width: float


def wilson_interval(successes: int, trials: int, alpha: float = 0.05) -> WilsonInterval:
    """Return the Wilson score confidence interval for a binomial proportion.

    The implementation follows the standard closed-form solution.  Values are
    clamped to ``[0, 1]`` to avoid returning invalid probabilities for edge
    cases (e.g., zero counts or saturated success rates).
    """

    if trials <= 0:
        raise ValueError("Wilson interval undefined for non-positive trial count")
    if successes < 0 or successes > trials:
        raise ValueError("Wilson interval successes must lie in [0, n]")

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = successes / trials
    denom = 1.0 + (z**2) / trials
    center = (phat + (z**2) / (2.0 * trials)) / denom
    half_width = z * math.sqrt((phat * (1.0 - phat) / trials) + (z**2) / (4.0 * trials**2)) / denom

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return WilsonInterval(lower=lower, upper=upper, center=center, width=upper - lower)


# ---------------------------------------------------------------------------
# Independence baselines
# ---------------------------------------------------------------------------


def stable_prod_one_minus(values: Sequence[float]) -> float:
    """Compute ``∏ (1 - x)`` with log-space stabilisation.

    This helper avoids underflow when many terms are close to one.  Inputs are
    clamped to ``[0, 1]``.  The return value is guaranteed to live in the same
    interval.
    """

    total = 0.0
    for x in values:
        if math.isnan(x) or math.isinf(x):
            raise ValueError("Probability values must be finite numbers")
        if x <= 0.0:
            total += 0.0
        elif x >= 1.0:
            return 0.0
        else:
            total += math.log1p(-float(x))
    return float(min(1.0, max(0.0, math.exp(total))))


def independence_or(tprs: Sequence[float], fprs: Sequence[float]) -> Dict[str, float]:
    """Return the independence baseline for an OR composition."""

    if len(tprs) != len(fprs):
        raise ValueError("TPR and FPR arrays must have equal length")
    if not tprs:
        raise ValueError("Need at least one rail to compute independence")

    miss_prod = stable_prod_one_minus(tprs)
    alarm_prod = stable_prod_one_minus(fprs)
    tpr_pi = 1.0 - miss_prod
    fpr_pi = 1.0 - alarm_prod
    return {
        "tpr": float(min(1.0, max(0.0, tpr_pi))),
        "fpr": float(min(1.0, max(0.0, fpr_pi))),
        "j": float(tpr_pi - fpr_pi),
    }


def independence_and(tprs: Sequence[float], fprs: Sequence[float]) -> Dict[str, float]:
    """Return the independence baseline for an AND composition."""

    if len(tprs) != len(fprs):
        raise ValueError("TPR and FPR arrays must have equal length")
    if not tprs:
        raise ValueError("Need at least one rail to compute independence")

    prod_tpr = 1.0
    prod_fpr = 1.0
    for tpr in tprs:
        if math.isnan(tpr) or math.isinf(tpr):
            raise ValueError("Probability values must be finite numbers")
        prod_tpr *= float(min(1.0, max(0.0, tpr)))
    for fpr in fprs:
        if math.isnan(fpr) or math.isinf(fpr):
            raise ValueError("Probability values must be finite numbers")
        prod_fpr *= float(min(1.0, max(0.0, fpr)))
    return {"tpr": prod_tpr, "fpr": prod_fpr, "j": prod_tpr - prod_fpr}


# ---------------------------------------------------------------------------
# Fréchet–Hoeffding envelopes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FHEnvelope:
    """Container for TPR/FPR envelope and J band."""

    tpr_lower: float
    tpr_upper: float
    fpr_lower: float
    fpr_upper: float
    j_lower: float
    j_upper: float


def fh_bounds_intersection(marginals: Sequence[float]) -> Tuple[float, float]:
    """Bounds for intersection probability with the FH inequalities."""

    if not marginals:
        raise ValueError("At least one marginal required")
    for p in marginals:
        if math.isnan(p) or math.isinf(p):
            raise ValueError("Marginals must be finite numbers")
        if p < 0.0 or p > 1.0:
            raise ValueError("Marginals must lie in [0, 1]")

    k = len(marginals)
    lower = max(0.0, sum(marginals) - (k - 1))
    upper = min(marginals)
    return lower, upper


def fh_bounds_union(marginals: Sequence[float]) -> Tuple[float, float]:
    """Bounds for union probability with the FH inequalities."""

    if not marginals:
        raise ValueError("At least one marginal required")
    for p in marginals:
        if math.isnan(p) or math.isinf(p):
            raise ValueError("Marginals must be finite numbers")
        if p < 0.0 or p > 1.0:
            raise ValueError("Marginals must lie in [0, 1]")

    lower = max(marginals)
    upper = min(1.0, sum(marginals))
    return lower, upper


def fh_envelope(topology: str, tprs: Sequence[float], fprs: Sequence[float]) -> FHEnvelope:
    """Compute the FH envelope for a multi-rail composition.

    Parameters
    ----------
    topology:
        Either ``"serial_or"`` or ``"parallel_and"``.
    tprs / fprs:
        Per-rail true/false positive rates.
    """

    if len(tprs) != len(fprs):
        raise ValueError("TPR and FPR vectors must match in length")
    if not tprs:
        raise ValueError("Need at least one rail to compute FH envelope")

    miss_rates = [1.0 - float(min(1.0, max(0.0, t))) for t in tprs]
    alarm_rates = [float(min(1.0, max(0.0, f))) for f in fprs]

    if topology == "serial_or":
        miss_lower, miss_upper = fh_bounds_intersection(miss_rates)
        alarm_lower, alarm_upper = fh_bounds_union(alarm_rates)
    elif topology == "parallel_and":
        miss_lower, miss_upper = fh_bounds_union(miss_rates)
        alarm_lower, alarm_upper = fh_bounds_intersection(alarm_rates)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    tpr_lower = 1.0 - miss_upper
    tpr_upper = 1.0 - miss_lower
    fpr_lower = alarm_lower
    fpr_upper = alarm_upper

    j_lower = tpr_lower - fpr_upper
    j_upper = tpr_upper - fpr_lower

    # Clamp to numerical range
    tpr_lower = float(min(1.0, max(0.0, tpr_lower)))
    tpr_upper = float(min(1.0, max(0.0, tpr_upper)))
    fpr_lower = float(min(1.0, max(0.0, fpr_lower)))
    fpr_upper = float(min(1.0, max(0.0, fpr_upper)))
    j_lower = float(min(1.0, max(-1.0, j_lower)))
    j_upper = float(min(1.0, max(-1.0, j_upper)))

    return FHEnvelope(
        tpr_lower=tpr_lower,
        tpr_upper=tpr_upper,
        fpr_lower=fpr_lower,
        fpr_upper=fpr_upper,
        j_lower=j_lower,
        j_upper=j_upper,
    )


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BCaInterval:
    """Bias corrected and accelerated confidence interval."""

    lower: float
    upper: float
    width: float


def _jackknife_statistics(
    samples: Sequence[np.ndarray], stat_fn: Callable[[Sequence[np.ndarray]], float]
) -> np.ndarray:
    parts: List[float] = []
    for idx, arr in enumerate(samples):
        if arr.size == 0:
            raise ValueError("Cannot jackknife empty sample")
        for drop in range(arr.size):
            reduced = [a if j != idx else np.delete(a, drop) for j, a in enumerate(samples)]
            parts.append(stat_fn(reduced))
    return np.asarray(parts)


def _phi_inv(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("Probability must lie strictly between 0 and 1")
    return NormalDist().inv_cdf(p)


def bca_bootstrap(
    samples: Sequence[np.ndarray] | np.ndarray,
    stat_fn: Callable[[Sequence[np.ndarray]], float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    rng: int | np.random.Generator | None = None,
) -> BCaInterval:
    """Compute a BCa interval for ``stat_fn`` evaluated on ``samples``.

    Parameters
    ----------
    samples:
        Either a single ``ndarray`` or a sequence of arrays.  Each array is
        resampled independently with replacement.
    stat_fn:
        Callable returning a scalar statistic when fed the resampled arrays.
    alpha:
        Two-sided error rate (default 0.05 ⇒ 95% interval).
    n_bootstrap:
        Number of bootstrap replicates.
    rng:
        Optional random seed or ``Generator`` instance for reproducibility.
    """

    if isinstance(samples, np.ndarray):
        arrays = [np.asarray(samples)]
    else:
        arrays = [np.asarray(a) for a in samples]

    if not arrays:
        raise ValueError("BCa bootstrap requires at least one sample array")
    for arr in arrays:
        if arr.ndim != 1:
            raise ValueError("Bootstrap arrays must be one-dimensional")
        if arr.size == 0:
            raise ValueError("Bootstrap arrays must be non-empty")

    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    theta_hat = stat_fn(arrays)

    boot_stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        resampled = [arr[gen.integers(0, arr.size, arr.size)] for arr in arrays]
        boot_stats[i] = stat_fn(resampled)

    proportion_less = np.mean(boot_stats < theta_hat)
    z0 = _phi_inv(max(min(proportion_less, 1 - 1e-12), 1e-12))

    jack_stats = _jackknife_statistics(arrays, stat_fn)
    jack_bar = float(np.mean(jack_stats))
    num = np.sum((jack_bar - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_bar - jack_stats) ** 2) ** 1.5) + 1e-12
    acceleration = num / den

    z_alpha_low = _phi_inv(alpha / 2.0)
    z_alpha_high = _phi_inv(1.0 - alpha / 2.0)

    def _adjust(z: float) -> float:
        denom = 1.0 - acceleration * (z0 + z)
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        return 0.5 * (1.0 + math.erf((z0 + (z0 + z) / denom) / math.sqrt(2.0)))

    p_low = _adjust(z_alpha_low)
    p_high = _adjust(z_alpha_high)

    p_low = max(min(p_low, 1.0 - 1e-6), 1e-6)
    p_high = max(min(p_high, 1.0 - 1e-6), 1e-6)

    lower = float(np.quantile(boot_stats, p_low))
    upper = float(np.quantile(boot_stats, p_high))
    if lower > upper:
        lower, upper = upper, lower
    return BCaInterval(lower=lower, upper=upper, width=upper - lower)


# ---------------------------------------------------------------------------
# Grouping utilities used by figure + memo writers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointRecord:
    """Minimal schema used by the downstream scripts."""

    topology: str
    rails: Tuple[str, ...]
    thresholds: Mapping[str, float]
    seed: int
    episodes: int
    empirical_tpr: float
    empirical_fpr: float
    empirical_j: float
    delta_j: float
    independence_tpr: float
    independence_fpr: float
    independence_j: float
    fh_tpr_lower: float
    fh_tpr_upper: float
    fh_fpr_lower: float
    fh_fpr_upper: float
    fh_j_lower: float
    fh_j_upper: float
    classification: str
    cc_l: float
    d_lamp: bool
    wilson_world0_width: float
    wilson_world1_width: float
    bca_delta_width: float
    bca_j_width: float

    @property
    def group_key(self) -> str:
        th = ",".join(f"{name}={self.thresholds[name]:.2f}" for name in sorted(self.thresholds))
        rails = "+".join(self.rails)
        return f"{self.topology}:{rails}@{th}"


def aggregate_by_group(points: Sequence[PointRecord]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for p in points:
        entry = grouped.setdefault(
            p.group_key,
            {
                "topology": p.topology,
                "rails": list(p.rails),
                "thresholds": {name: p.thresholds[name] for name in sorted(p.thresholds)},
                "empirical_tpr": [],
                "empirical_fpr": [],
                "empirical_j": [],
                "delta_j": [],
                "independence_tpr": [],
                "independence_fpr": [],
                "independence_j": [],
                "fh_tpr_lower": [],
                "fh_tpr_upper": [],
                "fh_fpr_lower": [],
                "fh_fpr_upper": [],
                "fh_j_lower": [],
                "fh_j_upper": [],
                "classification": [],
                "cc_l": [],
                "d_lamp": [],
                "wilson_world0_width": [],
                "wilson_world1_width": [],
                "bca_delta_width": [],
                "bca_j_width": [],
            },
        )
        for key in [
            "empirical_tpr",
            "empirical_fpr",
            "empirical_j",
            "delta_j",
            "independence_tpr",
            "independence_fpr",
            "independence_j",
            "fh_tpr_lower",
            "fh_tpr_upper",
            "fh_fpr_lower",
            "fh_fpr_upper",
            "fh_j_lower",
            "fh_j_upper",
            "cc_l",
            "wilson_world0_width",
            "wilson_world1_width",
            "bca_delta_width",
            "bca_j_width",
        ]:
            entry[key].append(getattr(p, key))
        entry["classification"].append(p.classification)
        entry["d_lamp"].append(p.d_lamp)
    return grouped


def summarise_group(entry: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "topology": entry["topology"],
        "rails": entry["rails"],
        "thresholds": entry["thresholds"],
    }

    def mean(xs: Sequence[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    def frac(predicate: Sequence[bool]) -> float:
        return float(np.mean(predicate)) if predicate else float("nan")

    for key in [
        "empirical_tpr",
        "empirical_fpr",
        "empirical_j",
        "delta_j",
        "independence_tpr",
        "independence_fpr",
        "independence_j",
        "fh_tpr_lower",
        "fh_tpr_upper",
        "fh_fpr_lower",
        "fh_fpr_upper",
        "fh_j_lower",
        "fh_j_upper",
        "cc_l",
        "wilson_world0_width",
        "wilson_world1_width",
        "bca_delta_width",
        "bca_j_width",
    ]:
        summary[key] = {
            "mean": mean(entry[key]),
            "std": float(np.std(entry[key], ddof=1)) if len(entry[key]) > 1 else 0.0,
        }
    summary["classification"] = {
        "constructive": entry["classification"].count("constructive"),
        "independent": entry["classification"].count("independent"),
        "destructive": entry["classification"].count("destructive"),
    }
    summary["d_lamp_rate"] = frac(entry["d_lamp"])
    return summary


def compute_regime_counts(points: Sequence[PointRecord]) -> Dict[str, int]:
    counts = {"constructive": 0, "independent": 0, "destructive": 0}
    for p in points:
        if p.classification not in counts:
            continue
        counts[p.classification] += 1
    return counts
