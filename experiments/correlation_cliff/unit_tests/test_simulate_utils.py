import numpy as np
import pytest

from experiments.correlation_cliff.simulate import utils as U


def test_theory_surface_present_and_modes_valid():
    assert U.THEORY_IMPORT_MODE in ("package", "script", "package_failed", "no_parent_package")
    # Required surface
    for name in [
        "FHBounds",
        "TwoWorldMarginals",
        "WorldMarginals",
        "compute_fh_jc_envelope",
        "fh_bounds",
        "joint_cells_from_marginals",
        "kendall_tau_a_from_joint",
        "phi_from_joint",
        "p11_fh_linear",
        "pC_from_joint",
    ]:
        assert hasattr(U, name), f"missing export: {name}"


def test_build_linear_lambda_grid_basic_both():
    g = U.build_linear_lambda_grid(num=5, start=0.0, stop=1.0, closed="both")
    assert g.shape == (5,)
    assert np.isclose(g[0], 0.0)
    assert np.isclose(g[-1], 1.0)
    assert np.all(np.diff(g) > 0)


def test_build_linear_lambda_grid_neither_excludes_endpoints():
    g = U.build_linear_lambda_grid(num=3, start=0.0, stop=1.0, closed="neither")
    assert g.shape == (3,)
    assert g[0] > 0.0
    assert g[-1] < 1.0
    assert np.all(np.diff(g) > 0)


@pytest.mark.parametrize("closed", ["left", "right"])
def test_build_linear_lambda_grid_left_right_length_and_monotone(closed):
    g = U.build_linear_lambda_grid(num=4, start=0.0, stop=1.0, closed=closed)
    assert g.shape == (4,)
    assert np.all(np.diff(g) > 0)
    if closed == "left":
        assert np.isclose(g[0], 0.0)
        assert g[-1] < 1.0
    else:
        assert g[0] > 0.0
        assert np.isclose(g[-1], 1.0)


def test_build_linear_lambda_grid_num_strict_rejects_bool_and_floaty_int():
    with pytest.raises(TypeError):
        U.build_linear_lambda_grid(num=True, closed="neither")  # bool rejected

    with pytest.raises(TypeError):
        U.build_linear_lambda_grid(num=3.0, closed="neither")  # no silent coercion


def test_build_linear_lambda_grid_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        U.build_linear_lambda_grid(num=0)

    with pytest.raises(ValueError):
        U.build_linear_lambda_grid(num=2, start=1.0, stop=0.0)

    with pytest.raises(ValueError):
        U.build_linear_lambda_grid(num=2, start=-0.1, stop=1.0)

    with pytest.raises(ValueError):
        U.build_linear_lambda_grid(num=2, start=0.0, stop=1.1)

    with pytest.raises(ValueError):
        U.build_linear_lambda_grid(num=2, closed="middle")  # type: ignore


def test_build_linear_lambda_grid_closed_both_requires_num_ge_2():
    with pytest.raises(ValueError):
        U.build_linear_lambda_grid(num=1, closed="both")


def test_build_linear_lambda_grid_snap_eps_snaps_endpoints():
    g = U.build_linear_lambda_grid(num=5, start=0.0, stop=1.0, closed="both", snap_eps=1e-12)
    assert np.isclose(g[0], 0.0)
    assert np.isclose(g[-1], 1.0)


def test_build_linear_lambda_grid_snap_eps_rejects_negative():
    with pytest.raises(ValueError):
        U.build_linear_lambda_grid(num=3, closed="neither", snap_eps=-1.0)
