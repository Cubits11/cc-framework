from __future__ import annotations

import pytest

from cc.kernel.metrics import MetricDomainError, cc_gain, cc_shift, independence_regret


def test_independence_regret_is_observed_minus_product_coupling() -> None:
    assert independence_regret(0.18, 0.12) == pytest.approx(0.06)
    assert independence_regret(0.04, 0.12) == pytest.approx(-0.08)


def test_cc_gain_accepts_mapping_and_sequence_inputs() -> None:
    assert cc_gain(0.18, {"input_filter": 0.04, "judge": 0.06}) == pytest.approx(3.0)
    assert cc_gain(0.18, [0.04, 0.06]) == pytest.approx(3.0)
    assert cc_gain(0.18, {"input_filter": 0.0, "judge": 0.0}) is None


def test_cc_shift_uses_deployed_minus_baseline_and_absolute_singleton_denominator() -> None:
    value = cc_shift(
        0.10,
        0.16,
        {"input_filter": 0.08, "judge": 0.11},
        {"input_filter": 0.10, "judge": 0.08},
    )

    assert value == pytest.approx(2.0)

    negative = cc_shift(
        0.16,
        0.10,
        {"input_filter": 0.08, "judge": 0.11},
        {"input_filter": 0.10, "judge": 0.08},
    )
    assert negative == pytest.approx(-2.0)


def test_cc_shift_validates_labels_and_sequence_lengths() -> None:
    with pytest.raises(MetricDomainError, match="matching keys"):
        cc_shift(0.1, 0.2, {"A": 0.1}, {"B": 0.2})

    with pytest.raises(MetricDomainError, match="same shape"):
        cc_shift(0.1, 0.2, {"A": 0.1}, [0.2])

    with pytest.raises(MetricDomainError, match="equal length"):
        cc_shift(0.1, 0.2, [0.1, 0.2], [0.2])


def test_cc_shift_is_undefined_when_singletons_do_not_move() -> None:
    assert cc_shift(0.1, 0.2, {"A": 0.1, "B": 0.2}, {"A": 0.1, "B": 0.2}) is None
