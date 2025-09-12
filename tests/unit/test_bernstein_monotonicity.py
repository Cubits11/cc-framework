import math
import pytest
from cc.cartographer.bounds import bernstein_tail

def test_bernstein_tail_monotone_in_t_and_n():
    # parameters
    D = 0.55
    v = 0.0475   # from I0=[0.05,0.05]
    n = 200
    # larger t => larger tail prob
    p_small = bernstein_tail(t=0.05, n=n, vbar=v, D=D)
    p_big   = bernstein_tail(t=0.10, n=n, vbar=v, D=D)
    assert p_big > p_small
    # larger n => smaller tail prob
    p_n_small = bernstein_tail(t=0.05, n=100, vbar=v, D=D)
    p_n_big   = bernstein_tail(t=0.05, n=400, vbar=v, D=D)
    assert p_n_big < p_n_small
