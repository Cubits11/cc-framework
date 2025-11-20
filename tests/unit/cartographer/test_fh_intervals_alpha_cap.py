import math
import pytest
from cc.cartographer.bounds import fh_intervals, fh_var_envelope

def test_fh_intervals_policy_binds_exact():
    I1, I0 = fh_intervals(tpr_a=0.72, tpr_b=0.65, fpr_a=0.035, fpr_b=0.05, alpha_cap=0.05)
    assert I1 == (pytest.approx(0.37), pytest.approx(0.65))
    assert I0 == (pytest.approx(0.05), pytest.approx(0.05))  # U0==alpha
    # variance envelopes
    assert fh_var_envelope(I1) == pytest.approx(0.25, 1e-12)
    assert fh_var_envelope(I0) == pytest.approx(0.05 * 0.95, 1e-12)

def test_fh_intervals_infeasible_alpha_raises():
    with pytest.raises(ValueError):
        fh_intervals(tpr_a=0.72, tpr_b=0.65, fpr_a=0.035, fpr_b=0.05, alpha_cap=0.04)

def test_fh_intervals_relaxed_no_cap():
    I1, I0 = fh_intervals(tpr_a=0.6, tpr_b=0.55, fpr_a=0.02, fpr_b=0.03, alpha_cap=None)
    assert I1 == (pytest.approx(0.15), pytest.approx(0.55))
    assert I0 == (pytest.approx(0.03), pytest.approx(0.05))
