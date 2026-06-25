import numpy as np
import pytest

from cc.cartographer.bounds import ensure_anchors, envelope_over_rocs, frechet_upper


def test_frechet_upper_rejects_out_of_range_roc_by_default() -> None:
    roc_a = [(0.0, 0.0), (1.2, 0.8)]
    roc_b = [(0.0, 0.0), (0.2, 0.9)]

    with pytest.raises(ValueError, match="out-of-range"):
        frechet_upper(roc_a, roc_b)


def test_frechet_upper_allows_explicit_legacy_clipping() -> None:
    roc_a = [(0.0, 0.0), (1.2, 0.8)]
    roc_b = [(0.0, 0.0), (0.2, 0.9)]

    value = frechet_upper(roc_a, roc_b, clip="silent")

    assert -1.0 <= value <= 1.0


def test_ensure_anchors_rejects_out_of_range_by_default() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        ensure_anchors(np.array([[0.0, 0.0], [0.5, 1.1]]))


def test_envelope_over_rocs_rejects_out_of_range_by_default() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        envelope_over_rocs([(0.0, 0.0), (0.2, 0.8)], [(0.0, 0.0), (-0.1, 0.7)])
