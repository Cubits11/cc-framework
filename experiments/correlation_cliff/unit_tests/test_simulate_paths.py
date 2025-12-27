import math
import pytest

from experiments.correlation_cliff.simulate.paths import p11_from_path, InputValidationError
from experiments.correlation_cliff.simulate import utils as U


def test_p11_fh_linear_endpoints_match_fh_bounds():
    pA, pB = 0.2, 0.7
    b = U.fh_bounds(pA, pB)
    L, Uu = float(b.L), float(b.U)

    p11_0, meta0 = p11_from_path(pA, pB, 0.0, path="fh_linear", path_params={})
    p11_1, meta1 = p11_from_path(pA, pB, 1.0, path="fh_linear", path_params={})

    assert math.isfinite(p11_0) and math.isfinite(p11_1)
    assert abs(p11_0 - L) < 1e-12
    assert abs(p11_1 - Uu) < 1e-12
    assert meta0["L"] == L and meta0["U"] == Uu


def test_p11_from_path_rejects_bad_gamma():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_power", path_params={"gamma": 0.0})


def test_p11_from_path_rejects_unknown_path():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linearzzz", path_params={})  # type: ignore
