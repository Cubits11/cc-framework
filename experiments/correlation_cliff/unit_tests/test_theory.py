"""
test_theory.py
==============

A maximal, research-grade pytest suite for theory.py.

What's included (a LOT):
- Analytic/golden tests for FH + composition rules
- Feasibility and boundary clipping tests
- Metamorphic tests (symmetries, monotonicity, invariance under transformations)
- Tight envelope tests (OR bounds derived from FH bounds, etc.)
- Two-world identifiability tests (JC/CC bounds, degeneracy handling)
- Deterministic Gaussian copula MC tests
- Optional SciPy deterministic Gaussian copula tests
- Optional Hypothesis property-based tests (skip if Hypothesis not installed)
- “Slow” performance tests gated behind env or marker (won't ruin CI)

Note: This suite assumes theory.py is importable as `import theory as th`.
If your project uses a package layout, change the import accordingly.

Run:
  pytest -q
Optional:
  pip install hypothesis scipy pandas
  pytest -q -m slow
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.correlation_cliff import theory as th_analysis
from experiments.correlation_cliff import theory_core as th

# ---------------------------------------------------------------------
# Fixtures: isolation
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_config():
    original = th.CONFIG
    yield
    th.set_config(**original.__dict__)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------
# Helper utilities (test-local)
# ---------------------------------------------------------------------


def rand_marginals(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    pA = rng.uniform(1e-6, 1 - 1e-6, size=n)
    pB = rng.uniform(1e-6, 1 - 1e-6, size=n)
    return pA, pB


def rand_feasible_joint(rng: np.random.Generator, pA: float, pB: float) -> float:
    lo, hi = th.fh_bounds(pA, pB)
    return float(lo if lo == hi else rng.uniform(lo, hi))


def _tw(pA0: float, pB0: float, pA1: float, pB1: float) -> th.TwoWorldMarginals:
    return th.TwoWorldMarginals(pA0=pA0, pB0=pB0, pA1=pA1, pB1=pB1)


def _tw_analysis(pA0: float, pB0: float, pA1: float, pB1: float) -> th_analysis.TwoWorldMarginals:
    return th_analysis.TwoWorldMarginals(
        w0=th_analysis.WorldMarginals(pA=pA0, pB=pB0),
        w1=th_analysis.WorldMarginals(pA=pA1, pB=pB1),
    )


def interval_gap_bruteforce(
    i0: tuple[float, float], i1: tuple[float, float], grid: int = 4001
) -> tuple[float, float]:
    a, b = i0
    c, d = i1
    xs = np.linspace(a, b, grid)
    ys = np.linspace(c, d, grid)
    # vectorized distance using broadcasting can be heavy; do smaller logic:
    # min gap in closed-form is known, but we want brute to validate formula:
    min_gap = float("inf")
    max_gap = 0.0
    # sample endpoints + a sparse grid for brute confirmation
    candidates_x = np.unique(np.concatenate([xs[:: max(1, grid // 50)], [a, b]]))
    candidates_y = np.unique(np.concatenate([ys[:: max(1, grid // 50)], [c, d]]))
    for x in candidates_x:
        dists = np.abs(candidates_y - x)
        min_gap = min(min_gap, float(np.min(dists)))
        max_gap = max(max_gap, float(np.max(dists)))
    return (min_gap, max_gap)


# ---------------------------------------------------------------------
# Basic contract tests: config + errors
# ---------------------------------------------------------------------


def test_set_config_unknown_field_raises():
    with pytest.raises(th.InputValidationError):
        th.set_config(not_a_field=123)  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", ["raise", "warn", "ignore", "RAISE", " Warn "])
def test_set_config_invariant_policy_accepts(policy):
    th.set_config(invariant_policy=policy)
    assert th.CONFIG.invariant_policy.strip().lower() == policy.strip().lower()


def test_invariant_warn_emits_warning():
    th.set_config(invariant_policy="warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        th._invariant("invariant warn check")  # type: ignore[attr-defined]
        assert any(isinstance(x.message, RuntimeWarning) for x in w)


def test_invariant_raise_raises():
    th.set_config(invariant_policy="raise")
    with pytest.raises(th.NumericalStabilityError):
        th._invariant("invariant raise check")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------
# FH bounds: golden + properties
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "pA,pB,lo,hi",
    [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (0.2, 0.3, 0.0, 0.2),
        (0.9, 0.9, 0.8, 0.9),
        (0.1, 0.95, 0.05, 0.1),
        (0.6, 0.2, 0.0, 0.2),
        (0.6, 0.8, 0.4, 0.6),
    ],
)
def test_fh_bounds_golden(pA, pB, lo, hi):
    got_lo, got_hi = th.fh_bounds(pA, pB)
    assert got_lo == pytest.approx(lo)
    assert got_hi == pytest.approx(hi)
    assert 0 <= got_lo <= got_hi <= 1


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.9, 0.9), (0.01, 0.99), (0.5, 0.5)])
def test_fh_bounds_symmetry(pA, pB):
    lo1, hi1 = th.fh_bounds(pA, pB)
    lo2, hi2 = th.fh_bounds(pB, pA)
    assert lo1 == pytest.approx(lo2)
    assert hi1 == pytest.approx(hi2)


def test_validate_joint_rejects_infeasible():
    with pytest.raises(th.FeasibilityError):
        th.validate_joint(0.2, 0.3, 0.25)  # hi=min=0.2
    with pytest.raises(th.FeasibilityError):
        th.validate_joint(0.9, 0.9, 0.7)  # lo=0.8


def test_validate_joint_clips_within_eps():
    th.set_config(eps_prob=1e-6)
    pA, pB = 0.2, 0.3
    lo, hi = th.fh_bounds(pA, pB)

    assert th.validate_joint(pA, pB, lo - 5e-7) == pytest.approx(lo)
    assert th.validate_joint(pA, pB, hi + 5e-7) == pytest.approx(hi)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("inf"), float("nan"), "x"])
def test_prob_validation_rejects_bad(p):
    with pytest.raises(th.InputValidationError):
        th.fh_bounds(p, 0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Composition: exact identities + bounds are tight
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule,pA,pB,p11,expected",
    [
        ("AND", 0.2, 0.3, 0.1, 0.1),
        ("AND", 0.9, 0.9, 0.85, 0.85),
        ("OR", 0.2, 0.3, 0.1, 0.4),
        ("OR", 0.2, 0.3, 0.0, 0.5),
        ("COND_OR", 0.2, 0.3, 0.0, 0.2 + (1 - 0.2) * 0.3),
        ("COND_OR", 0.2, 0.3, 0.2, 0.2 + (1 - 0.2) * 0.3),
    ],
)
def test_composed_rate_golden(rule, pA, pB, p11, expected):
    got = th.composed_rate(rule, pA, pB, p11)
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("rule", ["XOR", "", None])
def test_composed_rate_invalid_rule(rule):
    with pytest.raises(th.InputValidationError):
        th.composed_rate(rule, 0.2, 0.3, 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.6, 0.8), (0.01, 0.99), (0.9, 0.9)])
def test_composed_rate_bounds_match_fh_transforms(pA, pB):
    lo, hi = th.fh_bounds(pA, pB)

    # AND bounds are FH bounds
    and_lo, and_hi = th.composed_rate_bounds("AND", pA, pB)
    assert and_lo == pytest.approx(lo)
    assert and_hi == pytest.approx(hi)

    # OR bounds = pA+pB - p11, so endpoints swap
    or_lo, or_hi = th.composed_rate_bounds("OR", pA, pB)
    assert or_lo == pytest.approx(max(0.0, min(1.0, pA + pB - hi)))
    assert or_hi == pytest.approx(max(0.0, min(1.0, pA + pB - lo)))

    # COND_OR is degenerate
    c = pA + (1 - pA) * pB
    co_lo, co_hi = th.composed_rate_bounds("COND_OR", pA, pB)
    assert co_lo == pytest.approx(c)
    assert co_hi == pytest.approx(c)


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.6, 0.8), (0.9, 0.9)])
def test_or_monotone_in_p11(pA, pB):
    lo, hi = th.fh_bounds(pA, pB)
    # For OR: pC decreases as p11 increases.
    pC_lo = th.composed_rate("OR", pA, pB, lo)
    pC_hi = th.composed_rate("OR", pA, pB, hi)
    assert pC_lo >= pC_hi - 1e-12


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.6, 0.8), (0.9, 0.9)])
def test_and_monotone_in_p11(pA, pB):
    lo, hi = th.fh_bounds(pA, pB)
    pC_lo = th.composed_rate("AND", pA, pB, lo)
    pC_hi = th.composed_rate("AND", pA, pB, hi)
    assert pC_lo <= pC_hi + 1e-12


# ---------------------------------------------------------------------
# Dependence paths: FH family (golden + monotonicity + endpoints)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.9, 0.9), (0.1, 0.95), (0.55, 0.11)])
def test_fh_linear_endpoints(pA, pB):
    lo, hi = th.fh_bounds(pA, pB)
    assert th.p11_from_lambda("fh_linear", 0.0, pA, pB) == pytest.approx(lo)
    assert th.p11_from_lambda("fh_linear", 1.0, pA, pB) == pytest.approx(hi)


@pytest.mark.parametrize(
    "pA,pB,power", [(0.2, 0.3, 0.5), (0.2, 0.3, 2.0), (0.9, 0.9, 3.0), (0.55, 0.11, 1.7)]
)
def test_fh_power_endpoints(pA, pB, power):
    lo, hi = th.fh_bounds(pA, pB)
    assert th.p11_from_lambda("fh_power", 0.0, pA, pB, {"power": power}) == pytest.approx(lo)
    assert th.p11_from_lambda("fh_power", 1.0, pA, pB, {"power": power}) == pytest.approx(hi)


@pytest.mark.parametrize("pA,pB,alpha", [(0.2, 0.3, 4.0), (0.2, 0.3, 12.0), (0.9, 0.9, 10.0)])
def test_fh_scurve_midpoint(pA, pB, alpha):
    lo, hi = th.fh_bounds(pA, pB)
    mid = 0.5 * (lo + hi)
    got = th.p11_from_lambda("fh_scurve", 0.5, pA, pB, {"alpha": alpha})
    assert got == pytest.approx(mid, abs=1e-12)


@pytest.mark.parametrize(
    "path,params",
    [("fh_linear", None), ("fh_power", {"power": 2.3}), ("fh_scurve", {"alpha": 9.0})],
)
@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.9, 0.9), (0.33, 0.77)])
def test_fh_paths_monotone_in_lambda(path, params, pA, pB):
    grid = np.linspace(0.0, 1.0, 101)
    vals = [th.p11_from_lambda(path, float(l), pA, pB, params) for l in grid]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_p11_from_lambda_unknown_path_raises():
    with pytest.raises(th.InputValidationError):
        th.p11_from_lambda("nope", 0.5, 0.2, 0.3)


def test_p11_from_lambda_strict_params_rejects_unknown():
    with pytest.raises(th.InputValidationError):
        th.p11_from_lambda("fh_power", 0.5, 0.2, 0.3, {"power": 2.0, "extra": 1})


def test_p11_from_lambda_non_strict_params_allows_unknown():
    th.set_config(strict_params=False)
    got = th.p11_from_lambda("fh_power", 0.5, 0.2, 0.3, {"power": 2.0, "extra": 1})
    assert 0.0 <= got <= 1.0


@pytest.mark.parametrize("bad_lam", [-0.1, 1.1, float("nan"), "x"])
def test_p11_from_lambda_rejects_bad_lambda(bad_lam):
    with pytest.raises(th.InputValidationError):
        th.p11_from_lambda("fh_linear", bad_lam, 0.2, 0.3)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Clayton copula: limits + monotonicity + feasibility
# ---------------------------------------------------------------------


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.9, 0.9), (0.33, 0.77), (0.5, 0.5)])
def test_clayton_theta0_is_independence(pA, pB):
    assert th.p11_clayton_copula(pA, pB, theta=0.0) == pytest.approx(pA * pB, abs=1e-12)


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.33, 0.77), (0.5, 0.5)])
def test_clayton_monotone_in_theta(pA, pB):
    t0 = th.p11_clayton_copula(pA, pB, theta=0.0)
    t1 = th.p11_clayton_copula(pA, pB, theta=1.0)
    t2 = th.p11_clayton_copula(pA, pB, theta=5.0)
    assert t0 <= t1 + 1e-12
    assert t1 <= t2 + 1e-12


def test_clayton_rejects_negative_theta():
    with pytest.raises(th.InputValidationError):
        th.p11_clayton_copula(0.2, 0.3, theta=-1.0)


@pytest.mark.parametrize("pA,pB,theta", [(0.01, 0.99, 2.0), (1e-6, 0.7, 10.0), (0.7, 1e-6, 10.0)])
def test_clayton_output_is_fh_feasible(pA, pB, theta):
    p11 = th.p11_clayton_copula(pA, pB, theta=theta)
    lo, hi = th.fh_bounds(pA, pB)
    assert lo - 1e-12 <= p11 <= hi + 1e-12


# ---------------------------------------------------------------------
# Gaussian copula: MC determinism + sanity + monotonic trend
# ---------------------------------------------------------------------


def test_gaussian_mc_deterministic_given_seed():
    a = th.p11_gaussian_copula(
        0.33, 0.77, rho=0.2, method="mc", n_mc=20000, seed=123, antithetic=True
    )
    b = th.p11_gaussian_copula(
        0.33, 0.77, rho=0.2, method="mc", n_mc=20000, seed=123, antithetic=True
    )
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.33, 0.77), (0.5, 0.5)])
def test_gaussian_mc_rho0_approx_independence(pA, pB):
    got = th.p11_gaussian_copula(pA, pB, rho=0.0, method="mc", n_mc=40000, seed=0, antithetic=True)
    assert got == pytest.approx(pA * pB, abs=0.02)


@pytest.mark.parametrize("pA,pB", [(0.2, 0.3), (0.33, 0.77), (0.5, 0.5)])
def test_gaussian_mc_monotone_in_rho(pA, pB):
    low = th.p11_gaussian_copula(pA, pB, rho=-0.6, method="mc", n_mc=50000, seed=1, antithetic=True)
    mid = th.p11_gaussian_copula(pA, pB, rho=0.0, method="mc", n_mc=50000, seed=1, antithetic=True)
    high = th.p11_gaussian_copula(pA, pB, rho=0.6, method="mc", n_mc=50000, seed=1, antithetic=True)

    assert low <= mid + 0.03
    assert mid <= high + 0.03


def test_gaussian_rejects_bad_method():
    with pytest.raises(th.InputValidationError):
        th.p11_gaussian_copula(0.2, 0.3, rho=0.0, method="nope")  # type: ignore[arg-type]


def test_gaussian_rejects_bad_rho():
    with pytest.raises(th.InputValidationError):
        th.p11_gaussian_copula(0.2, 0.3, rho=2.0, method="mc")


def test_gaussian_output_is_fh_feasible():
    with pytest.warns(RuntimeWarning):
        p11 = th.p11_gaussian_copula(0.2, 0.3, rho=0.4, method="mc", n_mc=40000, seed=2)
    lo, hi = th.fh_bounds(0.2, 0.3)
    assert lo - 1e-12 <= p11 <= hi + 1e-12


# ---------------------------------------------------------------------
# Two-worlds + JC/CC bounds: golden + brute checks
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "w,JA,JB,Jbest",
    [
        (_tw(0.1, 0.2, 0.1, 0.2), 0.0, 0.0, 0.0),
        (_tw(0.1, 0.2, 0.3, 0.2), 0.2, 0.0, 0.2),
        (_tw(0.1, 0.2, 0.1, 0.8), 0.0, 0.6, 0.6),
        (_tw(0.9, 0.1, 0.2, 0.7), 0.7, 0.6, 0.7),
    ],
)
def test_singleton_gaps(w, JA, JB, Jbest):
    jA, jB, jbest = th.singleton_gaps(w)
    assert jA == pytest.approx(JA)
    assert jB == pytest.approx(JB)
    assert jbest == pytest.approx(Jbest)


@pytest.mark.parametrize("rule", ["AND", "OR", "COND_OR"])
def test_cc_bounds_when_jbest_zero(rule):
    w = _tw(0.2, 0.3, 0.2, 0.3)
    lo, hi = th.cc_bounds(w, rule)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


@pytest.mark.parametrize("rule", ["AND", "OR", "COND_OR"])
def test_jc_bounds_are_valid(rule):
    w = _tw(0.2, 0.3, 0.9, 0.9)
    lo, hi = th.jc_bounds(w, rule)
    assert 0.0 <= lo <= hi <= 1.0


@pytest.mark.parametrize("rule", ["AND", "OR", "COND_OR"])
def test_cc_bounds_match_jc_bounds_divided(rule):
    w = _tw(0.2, 0.3, 0.6, 0.1)
    _jA, _jB, jbest = th.singleton_gaps(w)
    jc_lo, jc_hi = th.jc_bounds(w, rule)
    cc_lo, cc_hi = th.cc_bounds(w, rule)
    if jbest == 0:
        assert (cc_lo, cc_hi) == (0.0, 0.0)
    else:
        assert cc_lo == pytest.approx(jc_lo / jbest)
        assert cc_hi == pytest.approx(jc_hi / jbest)


@pytest.mark.parametrize("rule", ["AND", "OR"])
def test_jc_bounds_match_bruteforce_interval_gap(rule):
    # Validate interval gap math by brute forcing on the pC intervals directly
    w = _tw(0.2, 0.3, 0.6, 0.1)
    I0 = th.composed_rate_bounds(rule, w.pA0, w.pB0)
    I1 = th.composed_rate_bounds(rule, w.pA1, w.pB1)
    brute = interval_gap_bruteforce(I0, I1)
    lo, hi = th.jc_bounds(w, rule)
    assert lo == pytest.approx(brute[0], abs=1e-6)
    assert hi == pytest.approx(brute[1], abs=1e-6)


def test_jc_bounds_overlap_case_min_zero():
    # Choose worlds where OR intervals overlap
    w = _tw(0.4, 0.4, 0.45, 0.35)
    lo, hi = th.jc_bounds(w, "OR")
    assert lo == pytest.approx(0.0)
    assert hi >= 0.0


def test_jc_bounds_nonoverlap_case_positive_min():
    w = _tw(0.05, 0.05, 0.9, 0.9)
    lo, hi = th.jc_bounds(w, "OR")
    assert lo > 0.0
    assert hi >= lo


# ---------------------------------------------------------------------
# compute_metrics_for_lambda: invariants + containment + degeneracy
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,params",
    [("fh_linear", None), ("fh_power", {"power": 2.0}), ("fh_scurve", {"alpha": 9.0})],
)
@pytest.mark.parametrize("rule", ["AND", "OR"])
@pytest.mark.parametrize("lam", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_compute_metrics_contains_bounds(path, params, rule, lam):
    w = _tw_analysis(0.2, 0.3, 0.6, 0.1)
    row = th_analysis.compute_metrics_for_lambda(w, rule, lam, path=path, path_params=params)
    assert 0.0 <= row["JC"] <= 1.0
    if math.isfinite(row["CC"]):
        assert row["CC"] >= 0.0
    b0 = th_analysis.fh_bounds(w.w0.pA, w.w0.pB)
    b1 = th_analysis.fh_bounds(w.w1.pA, w.w1.pB)
    assert b0.L - 1e-12 <= row["p11_0"] <= b0.U + 1e-12
    assert b1.L - 1e-12 <= row["p11_1"] <= b1.U + 1e-12


def test_compute_metrics_or_range_sane_across_lambda():
    w = _tw_analysis(0.2, 0.3, 0.6, 0.1)
    rows = [
        th_analysis.compute_metrics_for_lambda(w, "OR", lam, path="fh_linear")
        for lam in np.linspace(0, 1, 11)
    ]
    pC0 = [r["pC_0"] for r in rows]
    pC1 = [r["pC_1"] for r in rows]
    assert all(0.0 <= x <= 1.0 for x in pC0)
    assert all(0.0 <= x <= 1.0 for x in pC1)


def test_compute_metrics_jbest_zero_cc_zero():
    w = _tw_analysis(0.2, 0.3, 0.2, 0.3)
    row = th_analysis.compute_metrics_for_lambda(w, "OR", 0.5, path="fh_linear")
    assert row["Jbest"] == pytest.approx(0.0)
    assert math.isnan(row["CC"])


# ---------------------------------------------------------------------
# theory_curve + lambda grid + pandas optional behavior
# ---------------------------------------------------------------------


def test_default_lambda_grid_basic():
    g = th_analysis.default_lambda_grid(11)
    assert len(g.values) == 11
    assert g.values[0] == pytest.approx(0.0)
    assert g.values[-1] == pytest.approx(1.0)
    assert np.all(np.diff(g.values) > 0)


@pytest.mark.parametrize("n", [0, 1, -5])
def test_default_lambda_grid_rejects_small(n):
    with pytest.raises(th.InputValidationError):
        th_analysis.default_lambda_grid(n)  # type: ignore[arg-type]


def test_theory_curve_returns_dataframe_if_pandas():
    w = _tw_analysis(0.2, 0.3, 0.6, 0.1)
    grid = th_analysis.default_lambda_grid(9)
    out = th_analysis.theory_curve(w, "OR", grid.values, path="fh_linear")
    if th_analysis.pd is None:
        assert isinstance(out, list)
        assert len(out) == 9
        assert isinstance(out[0], dict)
        assert "CC" in out[0]
    else:
        import pandas as pd  # type: ignore

        assert isinstance(out, pd.DataFrame)
        assert len(out) == 9
        assert "CC" in out.columns


def test_approx_cc_ci_normal_basic():
    lo, hi = th_analysis.approx_cc_ci_normal(cc_hat=0.5, var_cc=0.01)
    assert lo < 0.5 < hi


def test_approx_cc_ci_normal_nan_var():
    lo, hi = th_analysis.approx_cc_ci_normal(cc_hat=0.5, var_cc=float("nan"))
    assert math.isnan(lo)
    assert math.isnan(hi)


# ---------------------------------------------------------------------
# Optional SciPy tests for deterministic Gaussian copula
# ---------------------------------------------------------------------


def test_gaussian_scipy_symmetry_if_available():
    pytest.importorskip("scipy", reason="SciPy not installed")
    pA, pB = 0.33, 0.77
    rho = 0.4
    a = th.p11_gaussian_copula(pA, pB, rho=rho, method="scipy")
    b = th.p11_gaussian_copula(pB, pA, rho=rho, method="scipy")
    assert a == pytest.approx(b, abs=1e-12)


def test_gaussian_scipy_rho0_exact_if_available():
    pytest.importorskip("scipy", reason="SciPy not installed")
    pA, pB = 0.2, 0.3
    got = th.p11_gaussian_copula(pA, pB, rho=0.0, method="scipy")
    assert got == pytest.approx(pA * pB, abs=1e-9)


# ---------------------------------------------------------------------
# Hypothesis property tests (optional)
# ---------------------------------------------------------------------

hypothesis = pytest.importorskip("hypothesis", reason="Hypothesis not installed")
from hypothesis import given, settings  # type: ignore
from hypothesis import strategies as st


@settings(max_examples=200, deadline=None)
@given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
def test_fh_bounds_property(pA, pB):
    lo, hi = th.fh_bounds(pA, pB)
    assert 0.0 <= lo <= hi <= 1.0
    # hi cannot exceed either marginal
    assert hi <= pA + 1e-12
    assert hi <= pB + 1e-12
    # lo is at least pA+pB-1
    assert lo >= max(0.0, pA + pB - 1.0) - 1e-12


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_validate_joint_accepts_only_feasible(pA, pB, lam):
    lo, hi = th.fh_bounds(pA, pB)
    p11 = lo + lam * (hi - lo)
    assert th.validate_joint(pA, pB, p11) == pytest.approx(p11)


@settings(max_examples=200, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_composed_bounds_contain_any_feasible_p11(pA, pB):
    lo, hi = th.fh_bounds(pA, pB)
    # sample a few p11s deterministically
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p11 = lo + lam * (hi - lo)
        for rule in ["AND", "OR", "COND_OR"]:
            pC = th.composed_rate(rule, pA, pB, p11)
            loC, hiC = th.composed_rate_bounds(rule, pA, pB)
            assert loC - 1e-12 <= pC <= hiC + 1e-12


@settings(max_examples=150, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_two_world_bounds_sane(pA0, pB0, pA1, pB1):
    w = _tw(pA0, pB0, pA1, pB1)
    for rule in ["AND", "OR", "COND_OR"]:
        jc_lo, jc_hi = th.jc_bounds(w, rule)
        assert 0.0 <= jc_lo <= jc_hi <= 1.0
        cc_lo, cc_hi = th.cc_bounds(w, rule)
        assert 0.0 <= cc_lo <= cc_hi or (cc_lo == 0.0 and cc_hi == 0.0)


# ---------------------------------------------------------------------
# Slow / performance tests (explicit marker)
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_theory_curve_large_grid_perf_smoke():
    # A smoke performance test; NOT a strict benchmark (CI machines vary).
    # Only runs if you do: pytest -m slow
    w = _tw_analysis(0.2, 0.3, 0.6, 0.1)
    grid = th_analysis.default_lambda_grid(2000)
    t0 = time.time()
    _ = th_analysis.theory_curve(w, "OR", grid.values, path="fh_linear")
    dt = time.time() - t0
    # loose: should be comfortably under a few seconds typically
    assert dt < 5.0
