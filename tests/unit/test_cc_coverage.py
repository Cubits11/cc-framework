# tests/unit/test_cc_coverage.py
import numpy as np

from cc.cartographer.intervals import (
    wilson_ci_from_counts, wilson_ci, cc_ci_wilson, cc_ci_bootstrap
)
from cc.cartographer.bounds import cc_confint, fh_var_envelope


def test_wilson_basic_sanity():
    # Symmetry near 0.5, monotonic in delta
    n = 200
    lo1, hi1 = wilson_ci(0.5, n, delta=0.10)
    lo2, hi2 = wilson_ci(0.5, n, delta=0.05)
    assert 0.0 <= lo2 <= lo1 <= 0.5 <= hi1 <= hi2 <= 1.0
    # Edge cases: k=0 and k=n
    lo0, hi0 = wilson_ci_from_counts(0, 50, delta=0.05)
    loN, hiN = wilson_ci_from_counts(50, 50, delta=0.05)
    assert 0.0 <= lo0 < 0.1 and hi0 < 0.2
    assert 0.8 < loN <= 1.0 and 0.9 <= hiN <= 1.0


def test_cc_ci_wilson_propagation():
    # If p1∈[0.6,0.7], p0∈[0.1,0.2], then Δ∈[0.4,0.6], CC=(1-Δ)/D
    # With D=0.5, CC∈[(1-0.6)/0.5, (1-0.4)/0.5] = [0.8, 1.2]
    D = 0.5
    lo, hi = cc_ci_from_known_intervals(D, (0.6, 0.7), (0.1, 0.2))
    assert abs(lo - 0.8) < 1e-9 and abs(hi - 1.2) < 1e-9

def cc_ci_from_known_intervals(D, p1_iv, p0_iv):
    from cc.cartographer.intervals import _diff_interval, cc_ci_from_diff_interval
    dlo, dhi = _diff_interval(p1_iv[0], p1_iv[1], p0_iv[0], p0_iv[1])
    return cc_ci_from_diff_interval(D, dlo, dhi)


def test_fh_bernstein_coverage_toy_bernoulli():
    # Empirical coverage ≥ nominal (allow small slack since bound is conservative)
    rng = np.random.default_rng(7)
    p1_true, p0_true = 0.62, 0.08
    D = 0.55
    n1, n0 = 200, 200
    delta = 0.05
    trials = 400

    cover = 0
    for _ in range(trials):
        y1 = rng.binomial(1, p1_true, size=n1).astype(float)
        y0 = rng.binomial(1, p0_true, size=n0).astype(float)
        p1_hat = float(np.mean(y1))
        p0_hat = float(np.mean(y0))

        # Use degenerate FH intervals so v̄ = p(1-p) (sharp variance) — still valid envelope
        I1 = (p1_true, p1_true)
        I0 = (p0_true, p0_true)

        lo, hi = cc_confint(
            n1=n1, n0=n0,
            p1_hat=p1_hat, p0_hat=p0_hat,
            D=D, I1=I1, I0=I0,
            delta=delta
        )
        cc_true = (1.0 - (p1_true - p0_true)) / D
        cover += (lo <= cc_true <= hi)

    # Allow slight Monte Carlo fluctuation; require ≥ 92% for nominal 95%
    assert cover / trials >= 0.92


def test_bootstrap_cc_smoke():
    rng = np.random.default_rng(123)
    p1_true, p0_true = 0.6, 0.2
    D = 0.5
    n1, n0 = 150, 150
    y1 = rng.binomial(1, p1_true, size=n1).astype(float)
    y0 = rng.binomial(1, p0_true, size=n0).astype(float)
    lo, hi = cc_ci_bootstrap(y1, y0, D, delta=0.10, B=800, seed=3)
    cc_true = (1.0 - (p1_true - p0_true)) / D
    assert lo <= cc_true <= hi
