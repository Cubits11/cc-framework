"""Acceptance gate: reproduce an external consumer's published result.

A downstream project needed Frechet-Hoeffding bounds over four deterministic
controls. It read this repository's most discoverable composition entry point,
``cc.core.composition_theory``, found that it operates on ROC point sets and
bounds the Youden J statistic, and concluded that its controls -- which refuse
if and only if a predicate holds, with no threshold and no operating curve --
did not fit. It implemented the inequality itself in 214 lines of JavaScript.

This module is the gate on that failure being fixed. It is not "the API
exists": it is *their published numbers reproduce from this library*, so those
214 lines could be deleted.

The numbers below are quoted from that project's published composition
document, not recomputed from its source. They are therefore an **independent
implementation**: different language, different purpose, written without
reference to this corpus or this code, and published before this corpus existed.

What that buys, and what it does not. It buys implementation independence: two
codebases of the same understanding can still disagree on arithmetic, and this
catches it when they do. It does NOT buy authorial independence -- that project
and this one share an author. A misreading of the estimand, a wrong assumption
about the event space, or a shared conceptual error survives this check
untouched, because there is no second mind to hold it. Earlier revisions of this
docstring called it "the only check in this repository that is not same-author."
That was wrong, and it claimed credibility this repository has not earned. No
check in this repository is currently author-independent.

Their non-claim travels with their numbers and is asserted here too: every
detection rate is ASSUMED. Zero records had been issued when they were
published, so no rate was ever observed. These are bounds over assumed inputs,
never measurements.
"""

from __future__ import annotations

import pytest

from cc.compose import compose_bounds, sensitivity

#: Published detection rates, small-chapter scenario. ASSUMED, never observed.
PUBLISHED_DETECTION_RATES = {
    "SELF_REPORTED": 0.70,
    "STALE": 0.99,
    "IMMUTABLE": 0.99,
    "SEPARATION": 0.60,
}

#: Published output, quoted from the consumer's document:
#:
#:     if independent (the fiction)   0.0012%
#:     Frechet-Hoeffding lower        0%
#:     Frechet-Hoeffding upper        1.00%
#:     width of the bound             1.00%
#:
#:     Assuming independence understates the admissible worst case by 833x.
PUBLISHED_LOWER = 0.00
PUBLISHED_UPPER = 0.01
PUBLISHED_WIDTH = 0.01
PUBLISHED_INDEPENDENCE = 0.000012
PUBLISHED_UNDERSTATEMENT_FACTOR = 833


def _evasion_marginals(detection: dict[str, float]) -> dict[str, float]:
    """A control's evasion probability is one minus its detection rate.

    The composed failure event is a conjunction: an inflated entry must evade
    *every* control. Getting this backwards -- treating it as a union -- would
    invert every number, which is why the consumer argued it explicitly rather
    than assuming it.
    """
    return {name: 1.0 - rate for name, rate in detection.items()}


def test_published_interval_reproduces():
    bounds = compose_bounds(
        _evasion_marginals(PUBLISHED_DETECTION_RATES),
        event="all",
        marginal_provenance="assumed",
    )
    assert bounds.lower == pytest.approx(PUBLISHED_LOWER, abs=1e-12)
    assert bounds.upper == pytest.approx(PUBLISHED_UPPER, abs=1e-12)
    assert bounds.width == pytest.approx(PUBLISHED_WIDTH, abs=1e-12)


def test_published_independence_baseline_reproduces():
    bounds = compose_bounds(
        _evasion_marginals(PUBLISHED_DETECTION_RATES),
        event="all",
        marginal_provenance="assumed",
    )
    assert bounds.independence_point == pytest.approx(PUBLISHED_INDEPENDENCE, abs=1e-12)


def test_published_understatement_factor_reproduces():
    """The headline: independence understates the admissible worst case by 833x."""
    bounds = compose_bounds(
        _evasion_marginals(PUBLISHED_DETECTION_RATES),
        event="all",
        marginal_provenance="assumed",
    )
    assert round(bounds.understatement_factor) == PUBLISHED_UNDERSTATEMENT_FACTOR


def test_published_sensitivity_finding_reproduces():
    """Their finding: improving a weak control moves the upper bound by 0.00pp.

    Quoted: "In the small-chapter scenario, a 5-point improvement to
    SELF_REPORTED or SEPARATION moves it by 0.00pp. Only the strongest controls
    move it."
    """
    rows = sensitivity(
        _evasion_marginals(PUBLISHED_DETECTION_RATES),
        event="all",
        delta=0.05,
        direction="decrease",
        marginal_provenance="assumed",
    )
    by_event = {row.event: row for row in rows}

    assert by_event["SELF_REPORTED"].upper_delta == pytest.approx(0.0, abs=1e-12)
    assert by_event["SEPARATION"].upper_delta == pytest.approx(0.0, abs=1e-12)
    # The two strong controls are tied at the minimum, so each moves the bound.
    assert by_event["STALE"].upper_delta == pytest.approx(-0.01, abs=1e-12)
    assert by_event["IMMUTABLE"].upper_delta == pytest.approx(-0.01, abs=1e-12)


def test_the_tie_is_reported_as_a_tie():
    """Two controls tie at the minimum, so no single event binds.

    Naming one of them would misdirect effort: improving either leaves the
    bound pinned by the other. The consumer's sensitivity table shows exactly
    this -- both STALE and IMMUTABLE move the bound.
    """
    bounds = compose_bounds(
        _evasion_marginals(PUBLISHED_DETECTION_RATES),
        event="all",
        marginal_provenance="assumed",
    )
    assert bounds.binding_event is None


@pytest.mark.parametrize(
    ("scenario", "detection", "expected_upper"),
    [
        ("optimistic", {"a": 0.95, "b": 0.99, "c": 0.99, "d": 0.90}, 0.01),
        ("small-chapter", {"a": 0.70, "b": 0.99, "c": 0.99, "d": 0.60}, 0.01),
        ("pessimistic", {"a": 0.50, "b": 0.90, "c": 0.95, "d": 0.40}, 0.05),
    ],
)
def test_all_three_published_scenarios_reproduce(scenario, detection, expected_upper):
    bounds = compose_bounds(
        _evasion_marginals(detection), event="all", marginal_provenance="assumed"
    )
    assert bounds.upper == pytest.approx(expected_upper, abs=1e-12), scenario
    assert bounds.lower == pytest.approx(0.0, abs=1e-12), scenario


def test_the_finding_holds_a_fifth_control_cannot_help():
    """Quoted: "Adding a fifth control cannot improve the bound."

    Under a conjunction the upper bound is min(p), so a further control can
    only lower the floor. This is the correlation cliff stated as an
    operational fact.
    """
    base = compose_bounds(
        _evasion_marginals(PUBLISHED_DETECTION_RATES),
        event="all",
        marginal_provenance="assumed",
    )
    with_fifth = compose_bounds(
        {**_evasion_marginals(PUBLISHED_DETECTION_RATES), "A_FIFTH_CONTROL": 0.20},
        event="all",
        marginal_provenance="assumed",
    )
    assert with_fifth.upper <= base.upper + 1e-12
    assert with_fifth.lower <= base.lower + 1e-12


def test_the_assumed_provenance_travels_with_the_result():
    """A bound over assumed rates must not be quotable as a measurement."""
    bounds = compose_bounds(
        _evasion_marginals(PUBLISHED_DETECTION_RATES),
        event="all",
        marginal_provenance="assumed",
    )
    assert bounds.marginal_provenance == "assumed"
    assert bounds.to_json()["marginal_provenance"] == "assumed"
    assert any("not a measurement" in claim for claim in bounds.non_claims)
