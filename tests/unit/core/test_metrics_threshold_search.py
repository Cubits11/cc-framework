from __future__ import annotations

import math

import numpy as np
import pytest

from cc.core.metrics import (
    Confusion,
    _search_best_threshold,
    confusion_from_scores,
    f1_score,
    rates_from_confusion,
    youden_j,
)


def _brute_force_search(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: str,
    pos_label: int = 1,
) -> tuple[float, Confusion, float]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    thresholds = np.concatenate(([np.inf], np.unique(s)[::-1], [-np.inf]))
    best_thr = float(thresholds[0])
    best_val = -np.inf
    best_cf: Confusion | None = None
    for threshold in thresholds:
        cf = confusion_from_scores(y, s, float(threshold), pos_label=pos_label)
        rates = rates_from_confusion(cf)
        value = (
            youden_j(rates.tpr, rates.fpr)
            if objective == "youden"
            else f1_score(cf.tp, cf.fp, cf.fn)
        )
        if value > best_val:
            best_thr = float(threshold)
            best_val = float(value)
            best_cf = cf
    assert best_cf is not None
    return best_thr, best_cf, best_val


def _assert_matches_brute_force(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: str,
    *,
    pos_label: int = 1,
) -> None:
    expected_thr, expected_cf, expected_value = _brute_force_search(
        scores,
        labels,
        objective,
        pos_label=pos_label,
    )
    actual_thr, actual_cf, actual_rates, actual_value = _search_best_threshold(
        scores,
        labels,
        objective,  # type: ignore[arg-type]
        pos_label,
    )

    if math.isnan(expected_thr):
        assert math.isnan(actual_thr)
    else:
        assert actual_thr == expected_thr
    assert actual_cf == expected_cf
    assert actual_rates == rates_from_confusion(expected_cf)
    assert actual_value == pytest.approx(expected_value)


@pytest.mark.parametrize("objective", ["youden", "f1"])
def test_search_best_threshold_duplicate_scores_match_brute_force(objective: str) -> None:
    scores = np.asarray([0.9, 0.8, 0.8, 0.8, 0.1], dtype=float)
    labels = np.asarray([1, 1, 0, 1, 0], dtype=int)

    _assert_matches_brute_force(scores, labels, objective)


@pytest.mark.parametrize("objective", ["youden", "f1"])
def test_search_best_threshold_constant_scores_match_brute_force(objective: str) -> None:
    scores = np.ones(10, dtype=float)
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)

    _assert_matches_brute_force(scores, labels, objective)


@pytest.mark.parametrize("objective", ["youden", "f1"])
@pytest.mark.parametrize(
    "labels",
    [
        np.ones(8, dtype=int),
        np.zeros(8, dtype=int),
        np.asarray([], dtype=int),
    ],
)
def test_search_best_threshold_one_class_and_empty_inputs_match_brute_force(
    objective: str,
    labels: np.ndarray,
) -> None:
    scores = np.linspace(0.1, 0.9, labels.size, dtype=float)

    _assert_matches_brute_force(scores, labels, objective)


@pytest.mark.parametrize("objective", ["youden", "f1"])
def test_search_best_threshold_nan_and_infinity_behavior_matches_brute_force(
    objective: str,
) -> None:
    scores = np.asarray([np.nan, np.inf, 0.7, 0.7, -np.inf, 0.2], dtype=float)
    labels = np.asarray([1, 1, 0, 1, 0, 0], dtype=int)

    _assert_matches_brute_force(scores, labels, objective)


@pytest.mark.parametrize("objective", ["youden", "f1"])
def test_search_best_threshold_randomized_matches_brute_force(objective: str) -> None:
    rng = np.random.default_rng(42)
    for n in range(1, 80):
        scores = np.round(rng.normal(size=n), 1)
        labels = rng.integers(0, 2, size=n)
        _assert_matches_brute_force(scores, labels, objective)


@pytest.mark.parametrize("objective", ["youden", "f1"])
def test_search_best_threshold_respects_non_default_positive_label(objective: str) -> None:
    scores = np.asarray([0.4, 0.8, 0.8, 0.1, 0.6], dtype=float)
    labels = np.asarray([2, 1, 2, 1, 2], dtype=int)

    _assert_matches_brute_force(scores, labels, objective, pos_label=2)
