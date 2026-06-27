from __future__ import annotations

import pytest

from cc.kernel.metrics import MetricDomainError, fh_position, fh_width


def test_fh_width_clips_tiny_negative_width_to_zero() -> None:
    assert fh_width(0.5 + 5.0e-10, 0.5, tol=1.0e-9) == 0.0


def test_fh_position_returns_none_for_degenerate_interval() -> None:
    assert fh_position(0.5, 0.5, 0.5, tol=1.0e-9) is None
    assert fh_position(0.5, 0.5, 0.5 + 5.0e-10, tol=1.0e-9) is None


def test_fh_position_clamps_only_within_tolerance() -> None:
    tol = 1.0e-3

    assert fh_position(0.2 - 0.5 * tol, 0.2, 0.8, tol=tol) == pytest.approx(0.0)
    assert fh_position(0.8 + 0.5 * tol, 0.2, 0.8, tol=tol) == pytest.approx(1.0)

    with pytest.raises(MetricDomainError):
        fh_position(0.2 - 2.0 * tol, 0.2, 0.8, tol=tol)
    with pytest.raises(MetricDomainError):
        fh_position(0.8 + 2.0 * tol, 0.2, 0.8, tol=tol)


def test_fh_position_stays_inside_unit_interval() -> None:
    assert fh_position(0.2, 0.2, 0.8) == pytest.approx(0.0)
    assert fh_position(0.5, 0.2, 0.8) == pytest.approx(0.5)
    assert fh_position(0.8, 0.2, 0.8) == pytest.approx(1.0)
