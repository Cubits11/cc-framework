import numpy as np

from cc.cartographer.bounds import cc_confint


def test_fh_bernstein_coverage_on_toy_world():
    """
    Empirical coverage for FH–Bernstein CI on a toy Bernoulli world
    where FH intervals are degenerate at the true p (sharp variance).
    """
    rng = np.random.default_rng(7)
    p1_true, p0_true = 0.62, 0.08
    D = 0.55
    n1 = 200
    n0 = 200
    delta = 0.05
    trials = 400

    covered = 0
    for _ in range(trials):
        y1 = rng.binomial(1, p1_true, size=n1).astype(float)
        y0 = rng.binomial(1, p0_true, size=n0).astype(float)
        p1_hat = float(np.mean(y1))
        p0_hat = float(np.mean(y0))

        # Degenerate FH intervals → v̄ = p(1-p), still a valid envelope
        I1 = (p1_true, p1_true)
        I0 = (p0_true, p0_true)

        lo, hi = cc_confint(
            n1=n1,
            n0=n0,
            p1_hat=p1_hat,
            p0_hat=p0_hat,
            D=D,
            I1=I1,
            I0=I0,
            delta=delta,
        )
        cc_true = (1.0 - (p1_true - p0_true)) / D
        covered += int(lo <= cc_true <= hi)

    # Allow Monte Carlo wiggle; nominal 95%, require ≥ 92%
    assert covered / trials >= 0.92


def test_bernstein_tail_monotonicity():
    """
    Sanity: Half-width should shrink with larger n and grow with larger delta (looser CI).
    """
    from cc.cartographer.bounds import fh_var_envelope, invert_bernstein_eps

    I = (0.10, 0.20)  # excludes 0.5 ⇒ v̄ < 0.25
    vbar = fh_var_envelope(I)
    eps_small_n = invert_bernstein_eps(n=100, vbar=vbar, delta=0.05)
    eps_large_n = invert_bernstein_eps(n=400, vbar=vbar, delta=0.05)
    assert eps_large_n < eps_small_n

    eps_looser = invert_bernstein_eps(n=200, vbar=vbar, delta=0.10)
    eps_tighter = invert_bernstein_eps(n=200, vbar=vbar, delta=0.02)
    assert eps_looser > eps_tighter
