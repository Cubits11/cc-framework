import math
import pytest

from experiments.correlation_cliff.simulate.paths import (
    p11_from_path,
    InputValidationError,
    FeasibilityError,
)
from experiments.correlation_cliff.simulate import utils as U


def _bounds(pA, pB):
    b = U.fh_bounds(pA, pB)
    return float(b.L), float(b.U)


def _assert_meta_required(meta):
    req = ["L", "U", "FH_width", "lam", "lam_eff", "raw_p11", "clip_amt", "clipped", "fh_violation", "fh_violation_amt"]
    for k in req:
        assert k in meta, f"missing meta key {k}"
        assert isinstance(meta[k], float)
        assert math.isfinite(meta[k]) or (k in ("lam_eff", "raw_p11") and math.isnan(meta[k]) is False)

    # Numeric-only contract
    for k, v in meta.items():
        assert isinstance(k, str)
        assert isinstance(v, float)


@pytest.mark.parametrize("pA,pB", [(0.2, 0.7), (0.1, 0.1), (0.9, 0.4), (0.01, 0.99)])
def test_fh_linear_endpoints_match_bounds(pA, pB):
    L, Uu = _bounds(pA, pB)

    p11_0, meta0 = p11_from_path(pA, pB, 0.0, path="fh_linear", path_params={})
    p11_1, meta1 = p11_from_path(pA, pB, 1.0, path="fh_linear", path_params={})

    assert abs(p11_0 - L) < 1e-12
    assert abs(p11_1 - Uu) < 1e-12
    _assert_meta_required(meta0)
    _assert_meta_required(meta1)
    assert meta0["L"] == L and meta0["U"] == Uu


@pytest.mark.parametrize("path,params", [
    ("fh_power", {"gamma": 2.0}),
    ("fh_scurve", {"k": 10.0}),
])
def test_fh_power_and_scurve_respect_bounds_for_endpoints(path, params):
    pA, pB = 0.2, 0.7
    L, Uu = _bounds(pA, pB)

    p11_0, meta0 = p11_from_path(pA, pB, 0.0, path=path, path_params=params)
    p11_1, meta1 = p11_from_path(pA, pB, 1.0, path=path, path_params=params)

    assert abs(p11_0 - L) < 1e-12
    assert abs(p11_1 - Uu) < 1e-12
    _assert_meta_required(meta0)
    _assert_meta_required(meta1)


def test_rejects_unknown_path():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linearzzz", path_params={})  # type: ignore


def test_rejects_non_mapping_path_params():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linear", path_params=["nope"])  # type: ignore


def test_rejects_bad_clip_policy_and_tol():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linear", path_params={"clip_policy": "maybe"})

    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linear", path_params={"clip_tol": 1e-3})


def test_rejects_bad_gamma_and_k():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_power", path_params={"gamma": 0.0})

    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_scurve", path_params={"k": -1.0})


def test_clip_policy_raise_throws_on_large_violation(monkeypatch):
    # Force a raw p11 that violates FH massively to test FeasibilityError path.
    from experiments.correlation_cliff.simulate import paths as P

    def _bad_linear(pA, pB, lam):
        return 1.5  # intentionally out of [0,1] and out of FH

    monkeypatch.setattr(P.U, "p11_fh_linear", _bad_linear)

    with pytest.raises(FeasibilityError):
        p11_from_path(0.2, 0.7, 0.5, path="fh_linear", path_params={"clip_policy": "raise", "clip_tol": 0.0})


def test_clip_policy_clip_clips_and_marks_meta(monkeypatch):
    from experiments.correlation_cliff.simulate import paths as P

    def _bad_linear(pA, pB, lam):
        return 1.5

    monkeypatch.setattr(P.U, "p11_fh_linear", _bad_linear)

    pA, pB = 0.2, 0.7
    L, Uu = _bounds(pA, pB)

    p11, meta = p11_from_path(pA, pB, 0.5, path="fh_linear", path_params={"clip_policy": "clip"})
    assert L <= p11 <= Uu
    assert meta["clipped"] == 1.0
    assert meta["fh_violation"] == 1.0
    assert meta["fh_violation_amt"] > 0.0
    _assert_meta_required(meta)


def test_gaussian_tau_endpoints_do_not_require_scipy():
    pA, pB = 0.2, 0.7
    L, Uu = _bounds(pA, pB)

    # lam=0 => tau=-1 => lower FH bound, no SciPy
    p11_0, meta0 = p11_from_path(pA, pB, 0.0, path="gaussian_tau",
