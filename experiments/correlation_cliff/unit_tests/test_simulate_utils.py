import numpy as np
import pytest

from experiments.correlation_cliff.simulate import utils as U


def test_build_linear_lambda_grid_basic():
    g = U.build_linear_lambda_grid(num=5, start=0.0, stop=1.0, closed="both")
    assert g.shape == (5,)
    assert np.isclose(g[0], 0.0)
    assert np.isclose(g[-1], 1.0)
    assert np.all(np.diff(g) > 0)


def test_build_linear_lambda_grid_neither():
    g = U.build_linear_lambda_grid(num=3, start=0.0, stop=1.0, closed="neither")
    assert len(g) == 3
    assert g[0] > 0.0
    assert g[-1] < 1.0


def test_theory_surface_present():
    # If this import works, required exports should exist
    assert hasattr(U, "fh_bounds")
    assert hasattr(U, "p11_fh_linear")
    assert hasattr(U, "TwoWorldMarginals")
