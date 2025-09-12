import pytest
from cc.cartographer.bounds import fh_var_envelope

def test_var_envelope_contains_half():
    assert fh_var_envelope((0.2, 0.8)) == pytest.approx(0.25)

def test_var_envelope_left_of_half():
    v = fh_var_envelope((0.10, 0.40))
    assert v == pytest.approx(max(0.10*0.90, 0.40*0.60))

def test_var_envelope_right_of_half():
    v = fh_var_envelope((0.60, 0.80))
    assert v == pytest.approx(max(0.60*0.40, 0.80*0.20))
