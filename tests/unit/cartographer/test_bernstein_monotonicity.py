# tests/unit/test_bernstein_monotonicity.py
import math
from cc.cartographer.bounds import bernstein_tail

def test_bernstein_tail_monotone_in_t_and_n_and_vbar():
    D = 0.55
    v = 0.0475  # from I0=[0.05,0.05]
    n = 200

    # Monotone in t: larger t => smaller tail probability
    p_t_small = bernstein_tail(t=0.05, n=n, vbar=v, D=D)
    p_t_big   = bernstein_tail(t=0.10, n=n, vbar=v, D=D)
    assert p_t_small > p_t_big, (p_t_small, p_t_big)

    # Monotone in n: larger n => smaller tail probability
    p_n_100 = bernstein_tail(t=0.08, n=100, vbar=v, D=D)
    p_n_400 = bernstein_tail(t=0.08, n=400, vbar=v, D=D)
    assert p_n_100 > p_n_400, (p_n_100, p_n_400)

    # Monotone in vbar: larger variance envelope => larger tail probability
    p_v_small = bernstein_tail(t=0.08, n=200, vbar=0.02, D=D)
    p_v_big   = bernstein_tail(t=0.08, n=200, vbar=0.25, D=D)
    assert p_v_small < p_v_big, (p_v_small, p_v_big)

    # Sanity: probabilities in (0, 1]
    for p in (p_t_small, p_t_big, p_n_100, p_n_400, p_v_small, p_v_big):
        assert 0.0 < p <= 1.0
