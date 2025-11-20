"""Unit tests for planner utilities."""

from __future__ import annotations

import pytest

from cc.cartographer.planner import needed_n_bernstein


def test_needed_n_bernstein_happy_path() -> None:
    n1, n0 = needed_n_bernstein(
        t=0.10,
        D=0.55,
        delta=0.05,
        I1=(0.49, 0.51),
        I0=(0.04, 0.05),
    )
    assert (n1, n0) == (778, 191)


def test_needed_n_bernstein_monotonicity() -> None:
    I1 = (0.49, 0.51)
    I0 = (0.04, 0.05)
    base = needed_n_bernstein(0.10, 0.55, 0.05, I1, I0)
    t_down = needed_n_bernstein(0.05, 0.55, 0.05, I1, I0)
    assert t_down[0] > base[0] and t_down[1] > base[1]
    D_down = needed_n_bernstein(0.10, 0.40, 0.05, I1, I0)
    assert D_down[0] > base[0] and D_down[1] > base[1]
    delta_down = needed_n_bernstein(0.10, 0.55, 0.01, I1, I0)
    assert delta_down[0] > base[0] and delta_down[1] > base[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t": 0.0},
        {"D": 0.0},
        {"delta": 0.0},
        {"delta": 1.0},
        {"I1": (0.6, 0.5)},
        {"I0": (0.2, -0.1)},
    ],
)
def test_needed_n_bernstein_invalid_inputs(kwargs: dict) -> None:
    params = dict(t=0.1, D=0.55, delta=0.05, I1=(0.49, 0.51), I0=(0.04, 0.05))
    params.update(kwargs)
    with pytest.raises(ValueError):
        needed_n_bernstein(**params)
