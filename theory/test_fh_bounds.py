import math
from typing import Any, List, Sequence, Tuple

import pytest

from cc.theory import fh_bounds as fh

# ---------------------------------------------------------------------
# Helpers for constructing ComposedJBounds directly
# ---------------------------------------------------------------------


def _make_composed_j_bounds_for_cc(
    j_lower: float,
    j_upper: float,
    max_individual_j: float,
    k_rails: int = 2,
    composition_type: str = "serial_or",
) -> fh.ComposedJBounds:
    """
    Helper to build a mathematically consistent ComposedJBounds object
    for regime-classification tests.

    We set TPR ≡ 1 and FPR interval so that:
        j_lower = 1 - fpr_upper
        j_upper = 1 - fpr_lower
    with 0 <= fpr_lower <= fpr_upper <= 1.

    We use dummy miss/alarm bounds that satisfy FHBounds invariants but
    do not attempt to encode a real joint model; ComposedJBounds does
    not enforce that cross-consistency.
    """
    assert 0.0 <= j_lower <= j_upper <= 1.0

    tpr_lower = 1.0
    tpr_upper = 1.0
    fpr_upper = 1.0 - j_lower
    fpr_lower = 1.0 - j_upper

    assert 0.0 <= fpr_lower <= fpr_upper <= 1.0

    tpr_bounds = fh.FHBounds(
        lower=tpr_lower,
        upper=tpr_upper,
        marginals=tuple([tpr_lower] * k_rails),
        bound_type="tpr_test",
        k_rails=k_rails,
    )

    fpr_bounds = fh.FHBounds(
        lower=fpr_lower,
        upper=fpr_upper,
        marginals=tuple([fpr_lower] * k_rails),
        bound_type="fpr_test",
        k_rails=k_rails,
    )

    # Dummy miss/alarm bounds - only need to be valid FHBounds.
    miss_bounds = fh.FHBounds(
        lower=0.0,
        upper=0.1,
        marginals=tuple([0.0] * k_rails),
        bound_type="miss_test",
        k_rails=k_rails,
    )

    alarm_bounds = fh.FHBounds(
        lower=fpr_lower,
        upper=fpr_upper,
        marginals=tuple([fpr_lower] * k_rails),
        bound_type="alarm_test",
        k_rails=k_rails,
    )

    individual_j_stats = tuple([max_individual_j] * k_rails)

    return fh.ComposedJBounds(
        j_lower=j_lower,
        j_upper=j_upper,
        tpr_bounds=tpr_bounds,
        fpr_bounds=fpr_bounds,
        miss_bounds=miss_bounds,
        alarm_bounds=alarm_bounds,
        individual_j_stats=individual_j_stats,
        composition_type=composition_type,
        k_rails=k_rails,
    )


# ---------------------------------------------------------------------
# validate_probability_vector
# ---------------------------------------------------------------------


def test_validate_probability_vector_accepts_valid_values() -> None:
    fh.validate_probability_vector([0.0, 0.25, 0.5, 1.0], "p")  # should not raise
    fh.validate_probability_vector((0.1, 0.9), "p")  # tuples also ok


@pytest.mark.parametrize(
    "probs",
    [
        [],
        [-0.1],
        [1.1],
        [math.nan],
        [math.inf],
        ["not-a-number"],
    ],
)
def test_validate_probability_vector_rejects_invalid_values(probs: Sequence[Any]) -> None:
    with pytest.raises(ValueError):
        fh.validate_probability_vector(probs, "p")


# ---------------------------------------------------------------------
# Basic FH intersection / union bounds
# ---------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_frechet_intersection_single_event_identity(p: float) -> None:
    assert fh.frechet_intersection_lower_bound([p]) == pytest.approx(p)
    b = fh.intersection_bounds([p])
    assert b.lower == pytest.approx(p)
    assert b.upper == pytest.approx(p)
    assert b.is_degenerate


def test_intersection_bounds_two_events_sharp() -> None:
    p1, p2 = 0.9, 0.8
    b = fh.intersection_bounds([p1, p2])
    expected_lower = max(0.0, p1 + p2 - 1.0)
    expected_upper = min(p1, p2)
    assert b.lower == pytest.approx(expected_lower)
    assert b.upper == pytest.approx(expected_upper)
    assert b.k_rails == 2
    assert b.bound_type == "intersection"


def test_union_bounds_three_events_sharp() -> None:
    marginals = [0.3, 0.4, 0.5]
    b = fh.union_bounds(marginals)
    expected_lower = max(marginals)
    expected_upper = min(1.0, sum(marginals))
    assert b.lower == pytest.approx(expected_lower)
    assert b.upper == pytest.approx(expected_upper)
    assert b.k_rails == 3
    assert b.bound_type == "union"


def test_union_vs_intersection_ordering() -> None:
    marginals = [0.3, 0.4, 0.5]
    int_b = fh.intersection_bounds(marginals)
    uni_b = fh.union_bounds(marginals)
    # Any feasible joint law must satisfy P(n) <= P(U),
    # so the bounds must allow that partial ordering.
    assert uni_b.lower >= int_b.lower - fh.MATHEMATICAL_TOLERANCE
    assert uni_b.upper >= int_b.upper - fh.MATHEMATICAL_TOLERANCE


# ---------------------------------------------------------------------
# serial_or_composition_bounds / parallel_and_composition_bounds
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "miss,fpr",
    [
        (0.0, 0.0),
        (0.2, 0.1),
        (0.7, 0.05),
    ],
)
def test_serial_or_composition_single_rail_matches_direct(miss: float, fpr: float) -> None:
    bounds = fh.serial_or_composition_bounds([miss], [fpr])

    tpr = 1.0 - miss
    j = tpr - fpr

    assert bounds.tpr_bounds.lower == pytest.approx(tpr)
    assert bounds.tpr_bounds.upper == pytest.approx(tpr)
    assert bounds.fpr_bounds.lower == pytest.approx(fpr)
    assert bounds.fpr_bounds.upper == pytest.approx(fpr)
    assert bounds.j_lower == pytest.approx(j)
    assert bounds.j_upper == pytest.approx(j)
    assert len(bounds.individual_j_stats) == 1
    assert bounds.individual_j_stats[0] == pytest.approx(j)
    assert bounds.width == pytest.approx(0.0)


@pytest.mark.parametrize(
    "miss,fpr",
    [
        (0.1, 0.05),
        (0.3, 0.1),
    ],
)
def test_parallel_and_composition_single_rail_matches_direct(miss: float, fpr: float) -> None:
    bounds = fh.parallel_and_composition_bounds([miss], [fpr])

    tpr = 1.0 - miss
    j = tpr - fpr

    assert bounds.tpr_bounds.lower == pytest.approx(tpr)
    assert bounds.tpr_bounds.upper == pytest.approx(tpr)
    assert bounds.fpr_bounds.lower == pytest.approx(fpr)
    assert bounds.fpr_bounds.upper == pytest.approx(fpr)
    assert bounds.j_lower == pytest.approx(j)
    assert bounds.j_upper == pytest.approx(j)
    assert len(bounds.individual_j_stats) == 1
    assert bounds.individual_j_stats[0] == pytest.approx(j)


def test_serial_or_composition_two_rails_bounds_are_sane() -> None:
    miss_rates = [0.2, 0.3]
    fpr_rates = [0.05, 0.1]
    bounds = fh.serial_or_composition_bounds(miss_rates, fpr_rates)

    # J must lie in [-1, 1]
    assert -1.0 <= bounds.j_lower <= bounds.j_upper <= 1.0

    # TPR/FPR bounds must be in [0, 1]
    assert 0.0 <= bounds.tpr_bounds.lower <= bounds.tpr_bounds.upper <= 1.0
    assert 0.0 <= bounds.fpr_bounds.lower <= bounds.fpr_bounds.upper <= 1.0

    # Individual J stats exist and match the per-rail formula
    expected_individual = [(1.0 - m) - f for m, f in zip(miss_rates, fpr_rates, strict=False)]
    assert len(bounds.individual_j_stats) == 2
    for j_ind, j_exp in zip(bounds.individual_j_stats, expected_individual, strict=False):
        assert j_ind == pytest.approx(j_exp)


# ---------------------------------------------------------------------
# Independence-based composition
# ---------------------------------------------------------------------


def test_independence_serial_or_j_matches_manual_formula() -> None:
    tprs = [0.7, 0.8, 0.9]
    fprs = [0.05, 0.1, 0.02]

    j = fh.independence_serial_or_j(tprs, fprs)

    miss_prod = 1.0
    for t in tprs:
        miss_prod *= 1.0 - t
    tpr_expected = 1.0 - miss_prod

    no_fp_prod = 1.0
    for f in fprs:
        no_fp_prod *= 1.0 - f
    fpr_expected = 1.0 - no_fp_prod

    j_expected = tpr_expected - fpr_expected
    assert j == pytest.approx(j_expected)
    assert -1.0 <= j <= 1.0


def test_independence_parallel_and_j_matches_manual_formula() -> None:
    tprs = [0.7, 0.8]
    fprs = [0.05, 0.1]

    j = fh.independence_parallel_and_j(tprs, fprs)

    tpr_expected = 1.0
    fpr_expected = 1.0
    for t in tprs:
        tpr_expected *= t
    for f in fprs:
        fpr_expected *= f

    j_expected = tpr_expected - fpr_expected
    assert j == pytest.approx(j_expected)
    assert -1.0 <= j <= 1.0


# ---------------------------------------------------------------------
# ComposedJBounds.classify_regime
# ---------------------------------------------------------------------


def test_classify_regime_constructive() -> None:
    # CC interval [0.5, 0.7] < threshold_constructive (0.95) => constructive
    bounds = _make_composed_j_bounds_for_cc(
        j_lower=0.5,
        j_upper=0.7,
        max_individual_j=1.0,
    )
    out = bounds.classify_regime()
    assert out["regime"] == "constructive"
    assert out["cc_bounds"][1] < 0.95


def test_classify_regime_destructive() -> None:
    # j_lower = 0.6, max_individual_j = 0.4 => cc_lower = 1.5 > 1.05 => destructive
    bounds = _make_composed_j_bounds_for_cc(
        j_lower=0.6,
        j_upper=0.8,
        max_individual_j=0.4,
    )
    out = bounds.classify_regime()
    assert out["regime"] == "destructive"
    assert out["confidence"] == pytest.approx(1.0)


def test_classify_regime_independent_band() -> None:
    # CC interval [0.95, 1.0] ⊆ [0.95, 1.05] => independent
    bounds = _make_composed_j_bounds_for_cc(
        j_lower=0.95,
        j_upper=1.0,
        max_individual_j=1.0,
    )
    out = bounds.classify_regime()
    assert out["regime"] == "independent"
    assert out["confidence"] == pytest.approx(1.0)


def test_classify_regime_uncertain_span() -> None:
    # CC interval spans constructive + destructive thresholds => uncertain
    bounds = _make_composed_j_bounds_for_cc(
        j_lower=0.8,
        j_upper=1.0,
        max_individual_j=0.9,
    )
    out = bounds.classify_regime()
    assert out["regime"] == "uncertain"
    assert 0.0 <= out["confidence"] <= 1.0


# ---------------------------------------------------------------------
# CII (Composability Interference Index)
# ---------------------------------------------------------------------


def test_cii_independence_baseline_gives_kappa_zero_for_true_independent() -> None:
    miss_rates = [0.3, 0.2]
    fpr_rates = [0.05, 0.1]
    bounds = fh.serial_or_composition_bounds(miss_rates, fpr_rates)

    individual_tprs = [1.0 - m for m in miss_rates]
    individual_fprs = list(fpr_rates)

    j_indep = fh.independence_serial_or_j(individual_tprs, individual_fprs)

    res = fh.compute_composability_interference_index(
        observed_j=j_indep,
        bounds=bounds,
        individual_tprs=individual_tprs,
        individual_fprs=individual_fprs,
        use_independence_baseline=True,
    )

    assert res["baseline_type"] == "independence"
    assert res["baseline_within_bounds"] is True
    assert res["interpretation"] == "independent"
    assert abs(res["cii"]) < 1e-6


def test_cii_fh_midpoint_baseline_when_no_individual_rates() -> None:
    miss_rates = [0.2, 0.3]
    fpr_rates = [0.05, 0.08]
    bounds = fh.serial_or_composition_bounds(miss_rates, fpr_rates)

    # No individual_tprs/fprs => fh_midpoint baseline
    res = fh.compute_composability_interference_index(
        observed_j=bounds.j_upper,
        bounds=bounds,
        individual_tprs=None,
        individual_fprs=None,
        use_independence_baseline=True,
    )

    assert res["baseline_type"] == "fh_midpoint"
    assert "cii" in res
    assert math.isfinite(float(res["cii"]))


def test_cii_degenerate_when_baseline_equals_worst() -> None:
    # Single-rail composition yields degenerate J interval
    miss_rates = [0.2]
    fpr_rates = [0.0]
    bounds = fh.serial_or_composition_bounds(miss_rates, fpr_rates)

    res = fh.compute_composability_interference_index(
        observed_j=bounds.j_lower,
        bounds=bounds,
        individual_tprs=None,
        individual_fprs=None,
        use_independence_baseline=False,  # baseline = j_lower
    )

    assert res["interpretation"] == "degenerate"
    assert res["reliability"] == "low"
    assert res["cii"] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# robust_inverse_normal & Wilson CI
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,sign",
    [
        (0.001, -1),
        (0.01, -1),
        (0.5, 0),
        (0.99, 1),
        (0.999, 1),
    ],
)
def test_robust_inverse_normal_basic_signs(p: float, sign: int) -> None:
    z = fh.robust_inverse_normal(p)
    if sign < 0:
        assert z < 0
    elif sign > 0:
        assert z > 0
    else:
        assert abs(z) < 1e-6


def test_robust_inverse_normal_monotonicity() -> None:
    p1, p2 = 0.1, 0.9
    z1 = fh.robust_inverse_normal(p1)
    z2 = fh.robust_inverse_normal(p2)
    assert z1 < z2  # CDF inverse must be monotone increasing


def test_wilson_interval_handles_extremes() -> None:
    # All failures
    lo, hi = fh.wilson_score_interval(0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi < 0.5

    # All successes
    lo2, hi2 = fh.wilson_score_interval(10, 10)
    assert 0.0 <= lo2 <= hi2 <= 1.0
    assert lo2 > 0.5

    # Symmetry-ish around 0.5 for 5/10
    lo_mid, hi_mid = fh.wilson_score_interval(5, 10)
    assert lo_mid < 0.5 < hi_mid


def test_wilson_interval_monotone_in_successes() -> None:
    lo1, hi1 = fh.wilson_score_interval(3, 10)
    lo2, hi2 = fh.wilson_score_interval(7, 10)
    # As successes increase, interval should shift upward
    assert lo2 > lo1
    assert hi2 > hi1


# ---------------------------------------------------------------------
# stratified_bootstrap_j_statistic
# ---------------------------------------------------------------------


class _AttackResultStub:
    def __init__(self, world_bit: int, success: bool) -> None:
        self.world_bit = world_bit
        self.success = success


def _fake_compute_j_statistic(results: Sequence[_AttackResultStub]) -> Tuple[float, float, float]:
    world0 = [r for r in results if r.world_bit == 0]
    world1 = [r for r in results if r.world_bit == 1]
    if not world0 or not world1:
        raise ValueError("Need both worlds")

    fp = sum(r.success for r in world0) / len(world0)
    tp = sum(r.success for r in world1) / len(world1)
    j = tp - fp
    return j, tp, fp


@pytest.mark.parametrize("n_bootstrap", [50, 200])
def test_stratified_bootstrap_j_statistic_respects_worlds(
    monkeypatch: pytest.MonkeyPatch,
    n_bootstrap: int,
) -> None:
    # Patch compute_j_statistic inside fh_bounds
    monkeypatch.setattr(fh, "compute_j_statistic", _fake_compute_j_statistic)

    # World 0: ~20% false positives
    w0: List[_AttackResultStub] = [_AttackResultStub(0, False) for _ in range(80)] + [
        _AttackResultStub(0, True) for _ in range(20)
    ]

    # World 1: ~90% true positives
    w1: List[_AttackResultStub] = [_AttackResultStub(1, True) for _ in range(90)] + [
        _AttackResultStub(1, False) for _ in range(10)
    ]

    bootstrap_j, ci = fh.stratified_bootstrap_j_statistic(
        w0,
        w1,
        n_bootstrap=n_bootstrap,
        random_seed=0,
    )

    assert len(bootstrap_j) == n_bootstrap
    assert isinstance(ci, tuple) and len(ci) == 2
    assert ci[0] <= ci[1]

    # True J ≈ 0.9 - 0.2 = 0.7; bootstrap mean should be close-ish.
    mean_j = sum(bootstrap_j) / len(bootstrap_j)
    assert mean_j == pytest.approx(0.7, rel=0.2)


# ---------------------------------------------------------------------
# extract_rates_from_attack_results
# ---------------------------------------------------------------------


def test_extract_rates_from_attack_results_basic() -> None:
    # 10 benign, 2 false positives
    benign = [_AttackResultStub(world_bit=0, success=(i < 2)) for i in range(10)]
    # 10 adversarial, 7 true positives
    adv = [_AttackResultStub(world_bit=1, success=(i < 7)) for i in range(10)]

    miss_rates, fpr_rates = fh.extract_rates_from_attack_results(benign + adv)

    assert len(miss_rates) == 1
    assert len(fpr_rates) == 1

    fpr = fpr_rates[0]
    miss = miss_rates[0]

    assert fpr == pytest.approx(2 / 10)
    tpr = 1.0 - miss
    assert tpr == pytest.approx(7 / 10)


def test_extract_rates_from_attack_results_warns_on_unknown_world_bit() -> None:
    results: List[_AttackResultStub] = [
        _AttackResultStub(world_bit=0, success=False),
        _AttackResultStub(world_bit=1, success=True),
        _AttackResultStub(world_bit=2, success=True),  # unknown
    ]
    with pytest.warns(UserWarning):
        miss_rates, fpr_rates = fh.extract_rates_from_attack_results(results)
    assert len(miss_rates) == 1
    assert len(fpr_rates) == 1


# ---------------------------------------------------------------------
# validate_fh_bounds_against_empirical
# ---------------------------------------------------------------------


def test_validate_fh_bounds_against_empirical_inside_and_overlap() -> None:
    miss_rates = [0.2, 0.3]
    fpr_rates = [0.05, 0.1]
    bounds = fh.serial_or_composition_bounds(miss_rates, fpr_rates)

    # Pick a J well inside the bounds
    observed_j = 0.5 * (bounds.j_lower + bounds.j_upper)

    # Confidence interval nested strictly inside the FH bounds
    ci = (bounds.j_lower + 0.01, bounds.j_upper - 0.01)

    report = fh.validate_fh_bounds_against_empirical(bounds, observed_j, ci)

    assert report["bounds_contain_observation"] is True
    assert report["position_interpretation"] in {"central", "near_lower_bound", "near_upper_bound"}
    assert report["ci_bounds_overlap"] is True
    assert report["statistical_consistency"] in {"good", "moderate"}


def test_validate_fh_bounds_against_empirical_no_overlap() -> None:
    miss_rates = [0.2, 0.3]
    fpr_rates = [0.05, 0.1]
    bounds = fh.serial_or_composition_bounds(miss_rates, fpr_rates)

    observed_j = bounds.j_upper + 0.05
    ci = (bounds.j_upper + 0.01, bounds.j_upper + 0.02)

    report = fh.validate_fh_bounds_against_empirical(bounds, observed_j, ci)

    assert report["bounds_contain_observation"] is False
    assert report["ci_bounds_overlap"] is False
    assert report["statistical_consistency"] == "poor"


# ---------------------------------------------------------------------
# sensitivity_analysis_fh_bounds
# ---------------------------------------------------------------------


def test_sensitivity_analysis_fh_bounds_basic() -> None:
    miss_rates = [0.2, 0.3]
    fpr_rates = [0.05, 0.1]

    result = fh.sensitivity_analysis_fh_bounds(
        miss_rates,
        fpr_rates,
        perturbation_size=0.01,
        n_perturbations=50,
    )

    assert result["sensitivity_analysis"] != "failed"
    assert result["baseline_width"] >= 0.0
    assert result["width_std"] >= 0.0
    assert result["j_lower_std"] >= 0.0
    assert result["j_upper_std"] >= 0.0
    assert result["n_successful_perturbations"] > 0
    assert result["sensitivity_interpretation"] in {"low", "moderate", "high"}


# ---------------------------------------------------------------------
# verify_fh_bound_properties
# ---------------------------------------------------------------------


def test_verify_fh_bound_properties_all_pass() -> None:
    results = fh.verify_fh_bound_properties()
    assert isinstance(results, dict)
    assert results  # non-empty
    # All internal checks should pass in a research-ready implementation.
    assert all(results.values())
