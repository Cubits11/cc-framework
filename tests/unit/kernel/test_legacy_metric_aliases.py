from __future__ import annotations

import pytest

from cc.core import metrics as legacy_metrics


def test_legacy_metric_aliases_warn_and_preserve_formulas() -> None:
    with pytest.warns(FutureWarning, match="cc_max"):
        assert legacy_metrics.cc_max(0.6, 0.2, 0.3) == pytest.approx(2.0)

    with pytest.warns(FutureWarning, match="cc_rel"):
        assert legacy_metrics.cc_rel(0.6, 0.2, 0.3) == pytest.approx(0.6 / 0.44)

    with pytest.warns(FutureWarning, match="delta_add"):
        assert legacy_metrics.delta_add(0.6, 0.2, 0.3) == pytest.approx(0.16)

    with pytest.warns(FutureWarning, match="delta_mult"):
        assert legacy_metrics.delta_mult(0.6, 0.2, 0.3) == pytest.approx(0.16)
