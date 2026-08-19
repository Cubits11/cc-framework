"""Unit tests for the ROC-free composition surface."""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cc.compose import (
    CountermonotoneUndefinedError,
    compose_bounds,
    marginal_from_operating_point,
    sensitivity,
)
from cc.kernel.sensitivity import AssumptionSet, LinearQuery, identified_region

PROBS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _lp_interval(marginals: dict[str, float], event: str) -> tuple[float, float]:
    names = tuple(marginals)
    assumptions = AssumptionSet.empty(names)
    for name in names:
        assumptions = assumptions.with_marginal_interval(name, marginals[name], marginals[name])
    query = (
        LinearQuery.intersection(names, names)
        if event == "all"
        else LinearQuery.union(names, names)
    )
    result = identified_region(query, assumptions)
    return result.lower_bound, result.upper_bound


# --- the surface must not speak ROC ------------------------------------------


def test_public_surface_mentions_no_detector_vocabulary():
    """The rejected-adoption report named ROC framing as the obstacle.

    A deterministic refusal rule has no threshold and no operating curve. If
    that vocabulary reappears in the public signature, the surface has drifted
    back to the shape a consumer already walked away from.
    """
    import inspect

    from cc.compose import _bounds

    forbidden = ("roc", "youden", "tpr", "fpr", "threshold", "operating_point")
    for name in ("compose_bounds", "sensitivity"):
        signature = str(inspect.signature(getattr(_bounds, name))).lower()
        for word in forbidden:
            assert word not in signature, f"{name} signature mentions {word!r}"


def test_operating_point_adapter_points_inward():
    """Detectors reach the event calculus, not the other way round."""
    assert marginal_from_operating_point(false_negative_rate=0.2) == pytest.approx(0.2)
    assert marginal_from_operating_point(false_negative_rate=0.2, prevalence=0.5) == pytest.approx(
        0.1
    )
    with pytest.raises(ValueError):
        marginal_from_operating_point(false_negative_rate=1.5)


# --- agreement with the LP kernel --------------------------------------------


@settings(max_examples=250, deadline=None)
@given(
    values=st.lists(PROBS, min_size=1, max_size=6),
    event=st.sampled_from(["all", "any"]),
)
def test_closed_form_agrees_with_finite_atom_lp(values, event):
    """The fast closed form and the general LP must not diverge.

    Two code paths compute the same object. If they disagree, one is wrong and
    every downstream artifact inherits it.
    """
    marginals = {f"E{i}": v for i, v in enumerate(values)}
    bounds = compose_bounds(marginals, event=event)
    lp_lower, lp_upper = _lp_interval(marginals, event)
    assert bounds.lower == pytest.approx(lp_lower, abs=1e-9)
    assert bounds.upper == pytest.approx(lp_upper, abs=1e-9)


# --- invariants ---------------------------------------------------------------


@settings(max_examples=400, deadline=None)
@given(
    values=st.lists(PROBS, min_size=1, max_size=8),
    event=st.sampled_from(["all", "any"]),
)
def test_interval_invariants(values, event):
    marginals = {f"E{i}": v for i, v in enumerate(values)}
    bounds = compose_bounds(marginals, event=event)
    assert 0.0 <= bounds.lower <= 1.0
    assert 0.0 <= bounds.upper <= 1.0
    assert bounds.lower <= bounds.upper + 1e-12
    assert bounds.width == pytest.approx(bounds.upper - bounds.lower, abs=1e-12)
    # The independence point must lie inside the sharp interval: it is one
    # admissible dependence structure among many.
    assert bounds.lower - 1e-12 <= bounds.independence_point <= bounds.upper + 1e-12


@settings(max_examples=200, deadline=None)
@given(values=st.lists(PROBS, min_size=2, max_size=6))
def test_conjunction_upper_bound_does_not_depend_on_count(values):
    """The correlation cliff, as an invariant.

    The conjunction upper bound is min(p). Adding an event can only lower it,
    never raise it -- so stacking controls cannot improve the worst-case bound.
    """
    marginals = {f"E{i}": v for i, v in enumerate(values)}
    before = compose_bounds(marginals, event="all")
    marginals["EXTRA"] = 1.0  # the weakest possible additional control
    after = compose_bounds(marginals, event="all")
    assert after.upper <= before.upper + 1e-12


def test_cliff_is_reproduced_at_scale():
    """Ten filters at p=0.1 bound to [0, 0.1]; independence predicts 1e-10."""
    bounds = compose_bounds({f"G{i}": 0.1 for i in range(10)}, event="all")
    assert bounds.lower == pytest.approx(0.0)
    assert bounds.upper == pytest.approx(0.1)
    assert bounds.independence_point == pytest.approx(1e-10)
    assert bounds.understatement_factor > 1e8


# --- binding event ------------------------------------------------------------


def test_binding_event_names_the_argmin():
    bounds = compose_bounds({"A": 0.3, "B": 0.01, "C": 0.4}, event="all")
    assert bounds.binding_event == "B"


def test_binding_event_is_none_on_a_tie():
    """A tie means no single event binds; naming one would misdirect effort."""
    bounds = compose_bounds({"A": 0.01, "B": 0.01, "C": 0.4}, event="all")
    assert bounds.binding_event is None


def test_binding_event_is_none_for_a_multi_event_union():
    """The union upper bound is a sum, not a single event."""
    assert compose_bounds({"A": 0.1, "B": 0.2}, event="any").binding_event is None
    assert compose_bounds({"A": 0.1}, event="any").binding_event == "A"


def test_binding_event_is_none_in_a_stipulated_regime():
    bounds = compose_bounds({"A": 0.3, "B": 0.01}, event="all", dependence="comonotone")
    assert bounds.binding_event is None


# --- countermonotonicity ------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 4, 8])
def test_countermonotone_is_refused_for_other_than_two_events(n):
    """Countermonotonicity is strictly bivariate.

    There is no n-dimensional countermonotonic structure for n > 2, and the FH
    lower bound is not a copula in dimension >= 3 though it stays pointwise
    sharp. The n=1 case was found by differential fuzz: the reference raised
    IndexError and the Node implementation returned NaN.
    """
    marginals = {f"E{i}": 0.3 for i in range(n)}
    with pytest.raises(CountermonotoneUndefinedError):
        compose_bounds(marginals, event="all", dependence="countermonotone")


def test_countermonotone_is_permitted_for_exactly_two():
    bounds = compose_bounds({"A": 0.7, "B": 0.8}, event="all", dependence="countermonotone")
    assert bounds.lower == bounds.upper == pytest.approx(0.5)


def test_unknown_dependence_fails_closed():
    with pytest.raises(ValueError, match="unknown dependence"):
        compose_bounds({"A": 0.3}, dependence="sorta-dependent")  # type: ignore[arg-type]


# --- input validation ---------------------------------------------------------


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf")])
def test_marginals_outside_the_unit_interval_are_refused(bad):
    with pytest.raises(ValueError):
        compose_bounds({"A": bad, "B": 0.2})


def test_empty_marginals_are_refused():
    with pytest.raises(ValueError, match="at least one event"):
        compose_bounds({})


def test_a_bare_sequence_is_refused():
    """An unnamed marginal cannot be reported against a binding event."""
    with pytest.raises(TypeError, match="mapping"):
        compose_bounds([0.1, 0.2])  # type: ignore[arg-type]


def test_unknown_event_kind_is_refused():
    with pytest.raises(ValueError, match="unknown event kind"):
        compose_bounds({"A": 0.1}, event="xor")


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("and", "all"),
        ("AND", "all"),
        ("intersection", "all"),
        ("or", "any"),
        ("OR", "any"),
        ("union", "any"),
    ],
)
def test_event_aliases_canonicalize(alias, canonical):
    assert compose_bounds({"A": 0.1, "B": 0.2}, event=alias).event == canonical


# --- the result object carries its context ------------------------------------


def test_result_carries_provenance_and_non_claims():
    bounds = compose_bounds({"A": 0.3}, marginal_provenance="assumed")
    assert bounds.marginal_provenance == "assumed"
    assert bounds.non_claims
    assert any("not a measurement" in c for c in bounds.non_claims)


def test_to_json_is_json_native_and_round_trips():
    bounds = compose_bounds({"A": 0.3, "B": 0.4}, event="all")
    payload = bounds.to_json()
    assert json.loads(json.dumps(payload)) == payload


def test_understatement_factor_serializes_as_null_when_undefined():
    """JSON has no infinity. Null means undefined, not large."""
    bounds = compose_bounds({"A": 0.0, "B": 0.4}, event="all")
    assert math.isinf(bounds.understatement_factor)
    assert bounds.to_json()["understatement_factor"] is None


# --- sensitivity --------------------------------------------------------------


def test_sensitivity_shows_only_the_binding_event_moves_the_upper_bound():
    """The actionable finding: effort on a weak control buys nothing."""
    rows = sensitivity({"WEAK": 0.40, "STRONG": 0.01, "MIDDLE": 0.30}, event="all", delta=0.05)
    by_event = {r.event: r for r in rows}
    assert by_event["STRONG"].moves_upper
    assert not by_event["WEAK"].moves_upper
    assert not by_event["MIDDLE"].moves_upper
    assert by_event["WEAK"].upper_delta == pytest.approx(0.0)


def test_sensitivity_rejects_an_out_of_range_delta():
    with pytest.raises(ValueError, match="delta"):
        sensitivity({"A": 0.3}, delta=0.0)
