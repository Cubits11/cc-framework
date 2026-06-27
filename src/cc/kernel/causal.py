"""Causal estimators for the two-world guardrail protocol.

The primary estimand is the population average treatment effect of adding
guardrail B to a system already running guardrail A:

    tau = E[Y(1) - Y(0)]

where ``Y(0)`` is the outcome under A-only and ``Y(1)`` is the outcome under
the A+B composition. This module focuses on inference when prompts arrive in
correlated batches or attack-strategy clusters.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

AssignmentMode = Literal["cluster", "individual"]


@dataclass(frozen=True)
class ClusteredTwoWorldData:
    """Synthetic or observed two-world outcomes with cluster labels."""

    outcome: np.ndarray
    world: np.ndarray
    cluster: np.ndarray
    true_effect: float | None = None
    true_icc: float | None = None
    assignment_mode: AssignmentMode | None = None


@dataclass(frozen=True)
class ClusterATEEstimate:
    """Cluster-robust two-world ATE estimate."""

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    alpha: float
    n_observations: int
    n_clusters: int
    n_world0: int
    n_world1: int
    bootstrap_reps: int
    bootstrap_valid_reps: int
    icc_estimate: float
    naive_se: float
    naive_ci_lower: float
    naive_ci_upper: float


def _as_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_1d_world(values: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    if arr.ndim != 1:
        raise ValueError("world must be one-dimensional")
    if not np.all(np.isin(arr, [0, 1])):
        raise ValueError("world must contain only 0/1 indicators")
    return arr


def _as_1d_cluster(values: Sequence[object] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("cluster must be one-dimensional")
    return arr


def _validate_lengths(outcome: np.ndarray, world: np.ndarray, cluster: np.ndarray) -> None:
    if not (outcome.size == world.size == cluster.size):
        raise ValueError("outcome, world, and cluster must have the same length")
    if outcome.size < 2:
        raise ValueError("at least two observations are required")
    if not np.any(world == 0) or not np.any(world == 1):
        raise ValueError("both worlds must be observed")


def difference_in_means(outcome: Sequence[float] | np.ndarray, world: Sequence[int] | np.ndarray) -> float:
    """Return ``mean(Y | W=1) - mean(Y | W=0)``."""

    y = _as_1d_float(outcome, "outcome")
    w = _as_1d_world(world)
    if y.size != w.size:
        raise ValueError("outcome and world must have the same length")
    if not np.any(w == 0) or not np.any(w == 1):
        raise ValueError("both worlds must be observed")
    return float(np.mean(y[w == 1]) - np.mean(y[w == 0]))


def naive_standard_error(
    outcome: Sequence[float] | np.ndarray,
    world: Sequence[int] | np.ndarray,
) -> float:
    """Independent-observation standard error for the two-world mean difference."""

    y = _as_1d_float(outcome, "outcome")
    w = _as_1d_world(world)
    if y.size != w.size:
        raise ValueError("outcome and world must have the same length")
    y0 = y[w == 0]
    y1 = y[w == 1]
    if y0.size == 0 or y1.size == 0:
        raise ValueError("both worlds must be observed")
    v0 = float(np.var(y0, ddof=1)) if y0.size > 1 else 0.0
    v1 = float(np.var(y1, ddof=1)) if y1.size > 1 else 0.0
    return float(math.sqrt(max(v0 / max(y0.size, 1) + v1 / max(y1.size, 1), 0.0)))


def empirical_intraclass_correlation(
    outcome: Sequence[float] | np.ndarray,
    cluster: Sequence[object] | np.ndarray,
) -> float:
    """Return a method-of-moments ICC diagnostic for clustered outcomes.

    The protocol uses this only as a diagnostic. Confidence intervals are
    computed by cluster bootstrap, not by applying a design-effect correction.
    """

    y = _as_1d_float(outcome, "outcome")
    c = _as_1d_cluster(cluster)
    if y.size != c.size:
        raise ValueError("outcome and cluster must have the same length")
    unique, inverse = np.unique(c, return_inverse=True)
    g = unique.size
    n = y.size
    if g < 2 or n <= g:
        return 0.0

    cluster_sizes = np.bincount(inverse)
    cluster_sums = np.bincount(inverse, weights=y)
    cluster_means = cluster_sums / np.maximum(cluster_sizes, 1)
    grand_mean = float(np.mean(y))
    ss_between = float(np.sum(cluster_sizes * (cluster_means - grand_mean) ** 2))
    ms_between = ss_between / max(g - 1, 1)

    ss_within = 0.0
    for idx in range(g):
        mask = inverse == idx
        ss_within += float(np.sum((y[mask] - cluster_means[idx]) ** 2))
    ms_within = ss_within / max(n - g, 1)

    m_bar = float(n) / float(g)
    denom = ms_between + (m_bar - 1.0) * ms_within
    if denom <= 0.0:
        return 0.0
    return float(np.clip((ms_between - ms_within) / denom, 0.0, 1.0))


def estimate_clustered_ate(
    outcome: Sequence[float] | np.ndarray,
    world: Sequence[int] | np.ndarray,
    cluster: Sequence[object] | np.ndarray,
    *,
    alpha: float = 0.05,
    bootstrap_reps: int = 2000,
    seed: int | np.random.Generator = 0,
) -> ClusterATEEstimate:
    """Estimate the two-world ATE with a cluster-bootstrap confidence interval.

    The bootstrap resamples whole clusters with replacement. This is appropriate
    when clusters are independent draws and observations within a cluster may be
    arbitrarily correlated.
    """

    y = _as_1d_float(outcome, "outcome")
    w = _as_1d_world(world)
    c = _as_1d_cluster(cluster)
    _validate_lengths(y, w, c)
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if int(bootstrap_reps) < 1:
        raise ValueError("bootstrap_reps must be positive")

    observed = difference_in_means(y, w)
    unique_clusters = np.unique(c)
    n_clusters = int(unique_clusters.size)
    if n_clusters < 2:
        raise ValueError("at least two clusters are required for cluster bootstrap")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(int(seed))
    cluster_indices = [np.flatnonzero(c == label) for label in unique_clusters]
    boot_effects: list[float] = []
    for _ in range(int(bootstrap_reps)):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([cluster_indices[int(j)] for j in sampled])
        wb = w[idx]
        if not np.any(wb == 0) or not np.any(wb == 1):
            continue
        boot_effects.append(difference_in_means(y[idx], wb))

    valid_reps = len(boot_effects)
    if valid_reps < max(30, int(0.2 * bootstrap_reps)):
        raise ValueError(
            "too few valid cluster-bootstrap resamples; ensure clusters cover both worlds"
        )

    boot = np.asarray(boot_effects, dtype=float)
    se = float(np.std(boot, ddof=1)) if valid_reps > 1 else 0.0
    ci_lower = float(np.quantile(boot, alpha / 2.0))
    ci_upper = float(np.quantile(boot, 1.0 - alpha / 2.0))

    naive_se = naive_standard_error(y, w)
    naive_df = max(min(int(np.sum(w == 0)), int(np.sum(w == 1))) - 1, 1)
    naive_half = float(stats.t.ppf(1.0 - alpha / 2.0, naive_df)) * naive_se
    naive_ci_lower = observed - naive_half
    naive_ci_upper = observed + naive_half

    if se <= 0.0:
        p_value = 1.0
    else:
        t_stat = observed / se
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(n_clusters - 1, 1))))

    return ClusterATEEstimate(
        ate=observed,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        method="cluster_bootstrap_ate",
        alpha=float(alpha),
        n_observations=int(y.size),
        n_clusters=n_clusters,
        n_world0=int(np.sum(w == 0)),
        n_world1=int(np.sum(w == 1)),
        bootstrap_reps=int(bootstrap_reps),
        bootstrap_valid_reps=valid_reps,
        icc_estimate=empirical_intraclass_correlation(y, c),
        naive_se=naive_se,
        naive_ci_lower=naive_ci_lower,
        naive_ci_upper=naive_ci_upper,
    )


def generate_synthetic_clustered_two_world(
    *,
    n_clusters: int,
    cluster_size: int,
    icc: float,
    treatment_effect: float,
    intercept: float = 0.0,
    seed: int = 0,
    assignment: AssignmentMode = "cluster",
) -> ClusteredTwoWorldData:
    """Generate continuous clustered two-world data with known ICC and ATE."""

    if int(n_clusters) < 4:
        raise ValueError("n_clusters must be at least 4")
    if int(cluster_size) < 2:
        raise ValueError("cluster_size must be at least 2")
    if not (0.0 <= float(icc) < 1.0):
        raise ValueError("icc must be in [0, 1)")
    rng = np.random.default_rng(int(seed))
    n_clusters = int(n_clusters)
    cluster_size = int(cluster_size)
    icc = float(icc)

    cluster_sd = math.sqrt(icc)
    residual_sd = math.sqrt(max(1.0 - icc, 0.0))
    cluster_effects = rng.normal(0.0, cluster_sd, size=n_clusters)

    if assignment == "cluster":
        cluster_worlds = np.array([0, 1] * ((n_clusters + 1) // 2), dtype=int)[:n_clusters]
        rng.shuffle(cluster_worlds)
        world = np.repeat(cluster_worlds, cluster_size)
    elif assignment == "individual":
        world = rng.binomial(1, 0.5, size=n_clusters * cluster_size).astype(int)
        for cluster_id in range(n_clusters):
            lo = cluster_id * cluster_size
            hi = lo + cluster_size
            if not np.any(world[lo:hi] == 0):
                world[lo] = 0
            if not np.any(world[lo:hi] == 1):
                world[hi - 1] = 1
    else:
        raise ValueError("assignment must be 'cluster' or 'individual'")

    cluster = np.repeat(np.arange(n_clusters), cluster_size)
    residual = rng.normal(0.0, residual_sd, size=n_clusters * cluster_size)
    outcome = (
        float(intercept)
        + float(treatment_effect) * world
        + cluster_effects[cluster]
        + residual
    )
    return ClusteredTwoWorldData(
        outcome=outcome.astype(float),
        world=world.astype(int),
        cluster=cluster.astype(int),
        true_effect=float(treatment_effect),
        true_icc=icc,
        assignment_mode=assignment,
    )


def compare_naive_vs_clustered_uncertainty(
    *,
    n_clusters: int = 40,
    cluster_size: int = 8,
    icc: float = 0.25,
    treatment_effect: float = 0.2,
    bootstrap_reps: int = 1000,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Compare naive and cluster-bootstrap uncertainty on known-ICC data."""

    data = generate_synthetic_clustered_two_world(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        icc=icc,
        treatment_effect=treatment_effect,
        assignment="cluster",
        seed=seed,
    )
    estimate = estimate_clustered_ate(
        data.outcome,
        data.world,
        data.cluster,
        bootstrap_reps=bootstrap_reps,
        seed=seed + 1,
    )
    ratio = estimate.se / estimate.naive_se if estimate.naive_se > 0 else math.inf
    return {
        "method": estimate.method,
        "true_icc": float(icc),
        "estimated_icc": estimate.icc_estimate,
        "true_effect": float(treatment_effect),
        "ate": estimate.ate,
        "cluster_se": estimate.se,
        "naive_se": estimate.naive_se,
        "se_understatement_ratio": float(ratio),
        "n_clusters": int(n_clusters),
        "cluster_size": int(cluster_size),
        "bootstrap_valid_reps": int(estimate.bootstrap_valid_reps),
    }


def run_coverage_simulation(
    *,
    icc_values: Iterable[float],
    cluster_sizes: Iterable[int],
    n_clusters: int = 48,
    true_effect: float = 0.2,
    alpha: float = 0.05,
    monte_carlo_reps: int = 100,
    bootstrap_reps: int = 300,
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Run a coverage grid for clustered two-world confidence intervals."""

    rows: list[dict[str, float | int | str]] = []
    master_rng = np.random.default_rng(int(seed))
    nominal = 1.0 - float(alpha)
    for cluster_size in cluster_sizes:
        for icc in icc_values:
            covered = 0
            naive_covered = 0
            se_ratios: list[float] = []
            valid = 0
            for _ in range(int(monte_carlo_reps)):
                sim_seed = int(master_rng.integers(0, 2**31 - 1))
                data = generate_synthetic_clustered_two_world(
                    n_clusters=n_clusters,
                    cluster_size=int(cluster_size),
                    icc=float(icc),
                    treatment_effect=float(true_effect),
                    assignment="cluster",
                    seed=sim_seed,
                )
                estimate = estimate_clustered_ate(
                    data.outcome,
                    data.world,
                    data.cluster,
                    alpha=alpha,
                    bootstrap_reps=bootstrap_reps,
                    seed=sim_seed + 1,
                )
                valid += 1
                if estimate.ci_lower <= true_effect <= estimate.ci_upper:
                    covered += 1
                if estimate.naive_ci_lower <= true_effect <= estimate.naive_ci_upper:
                    naive_covered += 1
                if estimate.naive_se > 0.0:
                    se_ratios.append(estimate.se / estimate.naive_se)

            coverage = covered / max(valid, 1)
            naive_coverage = naive_covered / max(valid, 1)
            mc_se = math.sqrt(max(nominal * (1.0 - nominal) / max(valid, 1), 0.0))
            tolerance = max(0.10, 3.0 * mc_se)
            rows.append(
                {
                    "n_clusters": int(n_clusters),
                    "cluster_size": int(cluster_size),
                    "icc": float(icc),
                    "true_effect": float(true_effect),
                    "nominal_coverage": nominal,
                    "monte_carlo_reps": int(valid),
                    "bootstrap_reps": int(bootstrap_reps),
                    "cluster_bootstrap_coverage": coverage,
                    "naive_coverage": naive_coverage,
                    "coverage_tolerance": tolerance,
                    "within_tolerance": abs(coverage - nominal) <= tolerance,
                    "mean_se_understatement_ratio": float(np.mean(se_ratios)) if se_ratios else math.inf,
                }
            )
    return rows


__all__ = [
    "AssignmentMode",
    "ClusterATEEstimate",
    "ClusteredTwoWorldData",
    "compare_naive_vs_clustered_uncertainty",
    "difference_in_means",
    "empirical_intraclass_correlation",
    "estimate_clustered_ate",
    "generate_synthetic_clustered_two_world",
    "naive_standard_error",
    "run_coverage_simulation",
]
