# experiments/correlation_cliff/tests/test_paths.py
import math
import pytest

from experiments.correlation_cliff.simulate.paths import (
    p11_from_path,
    InputValidationError,
    FeasibilityError,
)
from experiments.correlation_cliff.simulate import utils as U
from experiments.correlation_cliff.simulate.config import Path


def _bounds(pA, pB):
    b = U.fh_bounds(pA, pB)
    return float(b.L), float(b.U)


def _assert_meta_required(meta):
    req = [
        "L",
        "U",
        "FH_width",
        "lam",
        "lam_eff",
        "raw_p11",
        "clip_amt",
        "clipped",
        "fh_violation",
        "fh_violation_amt",
    ]
    for k in req:
        assert k in meta, f"missing meta key {k}"
        assert type(meta[k]) is float, f"meta[{k}] must be Python float (not numpy scalar)"
        assert math.isfinite(meta[k]), f"meta[{k}] must be finite, got {meta[k]!r}"

    # Numeric-only contract
    for k, v in meta.items():
        assert isinstance(k, str)
        assert type(v) is float
        assert math.isfinite(v), f"meta[{k}] must be finite, got {v!r}"


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


@pytest.mark.parametrize(
    "path,params",
    [
        ("fh_power", {"gamma": 2.0}),
        ("fh_scurve", {"k": 10.0}),
    ],
)
def test_fh_power_and_scurve_respect_bounds_for_endpoints(path, params):
    pA, pB = 0.2, 0.7
    L, Uu = _bounds(pA, pB)

    p11_0, meta0 = p11_from_path(pA, pB, 0.0, path=path, path_params=params)
    p11_1, meta1 = p11_from_path(pA, pB, 1.0, path=path, path_params=params)

    assert abs(p11_0 - L) < 1e-12
    assert abs(p11_1 - Uu) < 1e-12
    _assert_meta_required(meta0)
    _assert_meta_required(meta1)


def test_enum_path_normalization_works():
    pA, pB = 0.2, 0.7
    L, Uu = _bounds(pA, pB)

    p11_0, meta0 = p11_from_path(pA, pB, 0.0, path="Path.FH_LINEAR", path_params={})
    p11_1, meta1 = p11_from_path(pA, pB, 1.0, path="Path.FH_LINEAR", path_params={})

    assert abs(p11_0 - L) < 1e-12
    assert abs(p11_1 - Uu) < 1e-12
    _assert_meta_required(meta0)
    _assert_meta_required(meta1)


def test_rejects_bool_as_number_everywhere():
    with pytest.raises(InputValidationError):
        p11_from_path(True, 0.2, 0.3, path="fh_linear", path_params={})  # type: ignore
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, False, 0.3, path="fh_linear", path_params={})  # type: ignore
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, True, path="fh_linear", path_params={})  # type: ignore

    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_power", path_params={"gamma": True})  # type: ignore
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_scurve", path_params={"k": False})  # type: ignore


def test_rejects_unknown_path():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linearzzz", path_params={})  # type: ignore


def test_rejects_non_mapping_path_params():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linear", path_params=["nope"])  # type: ignore


def test_rejects_non_string_keys_in_path_params():
    with pytest.raises(InputValidationError):
        p11_from_path(0.2, 0.3, 0.5, path="fh_linear", path_params={1: "nope"})  # type: ignore


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


def test_width_zero_cases_are_stable_and_exact():
    # When one marginal is 0 or 1, FH envelope collapses.
    for pA, pB in [(0.0, 0.7), (1.0, 0.7), (0.3, 0.0), (0.3, 1.0), (0.0, 0.0), (1.0, 1.0)]:
        L, Uu = _bounds(pA, pB)
        assert abs(L - Uu) < 1e-15

        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p11, meta = p11_from_path(pA, pB, lam, path="fh_linear", path_params={})
            assert abs(p11 - L) < 1e-12
            _assert_meta_required(meta)
            assert meta["FH_width"] == pytest.approx(float(Uu - L), abs=1e-12)


def test_deterministic_grid_no_clipping_for_fh_family():
    # For FH-linear-family paths, U.p11_fh_linear should already be feasible; clipping should not occur.
    pA, pB = 0.13, 0.77
    for path, params in [
        ("fh_linear", {}),
        ("fh_power", {"gamma": 3.0}),
        ("fh_scurve", {"k": 8.0}),
    ]:
        for lam in [i / 20.0 for i in range(21)]:
            p11, meta = p11_from_path(pA, pB, lam, path=path, path_params=params)
            L, Uu = _bounds(pA, pB)
            assert L <= p11 <= Uu
            assert meta["clipped"] in (0.0, 1.0)
            assert meta["fh_violation"] in (0.0, 1.0)
            # In a correct FH-linear impl, we expect no violation/clipping:
            assert meta["fh_violation"] == 0.0
            assert meta["clipped"] == 0.0
            _assert_meta_required(meta)


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
    p11_0, meta0 = p11_from_path(pA, pB, 0.0, path="gaussian_tau", path_params={})
    # lam=1 => tau=+1 => upper FH bound, no SciPy
    p11_1, meta1 = p11_from_path(pA, pB, 1.0, path="gaussian_tau", path_params={})

    assert abs(p11_0 - L) < 1e-12
    assert abs(p11_1 - Uu) < 1e-12
    _assert_meta_required(meta0)
    _assert_meta_required(meta1)


def test_gaussian_tau_interior_requires_scipy_or_skips():
    pA, pB = 0.2, 0.7

    pytest.importorskip("scipy")

    p11, meta = p11_from_path(pA, pB, 0.5, path="gaussian_tau", path_params={})
    L, Uu = _bounds(pA, pB)
    assert L <= p11 <= Uu
    _assert_meta_required(meta)
