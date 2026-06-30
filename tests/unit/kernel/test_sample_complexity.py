from __future__ import annotations

import math

import pytest

from cc.kernel.sample_complexity import (
    bernoulli_rate_count,
    hoeffding_radius,
    pairwise_rate_count,
    sample_size_for_radius,
    simultaneous_bernoulli_radius,
    simultaneous_sample_size,
)


def test_hoeffding_radius_and_sample_size_are_dual_enough() -> None:
    n = sample_size_for_radius(0.05, 0.05)

    assert n == math.ceil(math.log(2 / 0.05) / (2 * 0.05**2))
    assert hoeffding_radius(n, 0.05) <= 0.05


def test_simultaneous_radius_uses_union_bound_rate_count() -> None:
    radius = simultaneous_bernoulli_radius(200, 3, 0.05)
    expected = math.sqrt(math.log(6 / 0.05) / 400)

    assert radius == pytest.approx(expected)
    assert simultaneous_sample_size(0.1, 3, 0.05) == math.ceil(
        math.log(6 / 0.05) / (2 * 0.1**2)
    )


def test_rate_count_helpers() -> None:
    assert pairwise_rate_count(1) == 0
    assert pairwise_rate_count(4) == 6
    assert bernoulli_rate_count(4) == 10
    assert bernoulli_rate_count(4, include_pairwise=False) == 4


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (hoeffding_radius, (0, 0.05)),
        (sample_size_for_radius, (0.0, 0.05)),
        (sample_size_for_radius, (0.1, 1.0)),
        (simultaneous_bernoulli_radius, (10, 0, 0.05)),
        (pairwise_rate_count, (-1,)),
    ],
)
def test_invalid_inputs_raise(fn, args) -> None:
    with pytest.raises((TypeError, ValueError)):
        fn(*args)
