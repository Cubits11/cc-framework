from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cc.kernel.frechet_classes import (
    PairwiseDependence,
    atom_matrix,
    distribution_moments,
    event_probability,
    frechet_bounds,
)
from cc.kernel.sensitivity import (
    AssumptionSet,
    LinearConstraint,
    LinearQuery,
    PartialIdentificationInfeasibleError,
    atom_predicate_mask,
    atoms_matching,
    atoms_where,
    enumerate_atoms,
    event_mask,
    identify,
)


def _exact_marginal_assumptions(guardrails: tuple[str, ...], marginals: np.ndarray) -> AssumptionSet:
    assumptions = AssumptionSet.empty(guardrails)
    for guardrail, marginal in zip(guardrails, marginals, strict=True):
        assumptions = assumptions.with_marginal_interval(guardrail, float(marginal), float(marginal))
    return assumptions


def _exact_pairwise_assumptions(
    assumptions: AssumptionSet,
    pairwise: np.ndarray,
    pairs: list[tuple[int, int]],
) -> AssumptionSet:
    guardrails = assumptions.guardrails
    out = assumptions
    for i, j in pairs:
        value = float(pairwise[i, j])
        out = out.with_pairwise_joint_interval(guardrails[i], guardrails[j], value, value)
    return out


def _query_for_event(guardrails: tuple[str, ...], event: str) -> LinearQuery:
    if event == "and":
        return LinearQuery.intersection(guardrails, guardrails, name="all")
    return LinearQuery.union(guardrails, guardrails, name="any")


@st.composite
def feasible_atom_laws(draw: st.DrawFn) -> tuple[tuple[str, ...], np.ndarray]:
    n_events = draw(st.integers(min_value=2, max_value=5))
    n_atoms = 1 << n_events
    weights = draw(
        st.lists(
            st.floats(
                min_value=1.0e-3,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n_atoms,
            max_size=n_atoms,
        )
    )
    distribution = np.asarray(weights, dtype=float)
    distribution = distribution / float(np.sum(distribution))
    guardrails = tuple(f"G{i}" for i in range(n_events))
    return guardrails, distribution


def test_atom_helpers_preserve_guardrail_order_and_support_predicates() -> None:
    guardrails = ("A", "B", "C")

    atoms = enumerate_atoms(guardrails)

    assert np.array_equal(atoms, atom_matrix(3))
    assert np.array_equal(event_mask(guardrails, "A"), np.asarray([False, True] * 4))
    assert np.array_equal(atoms_where(guardrails, "B", value=False), atoms[[0, 1, 4, 5]])

    def predicate(atom: dict[str, int]) -> bool:
        return atom["A"] == 1 and atom["C"] == 0

    assert np.array_equal(
        atom_predicate_mask(guardrails, predicate),
        [False, True, False, True, False, False, False, False],
    )
    assert np.array_equal(atoms_matching(guardrails, predicate), atoms[[1, 3]])


@given(feasible_atom_laws())
@settings(max_examples=35, deadline=None)
def test_marginal_only_engine_matches_classical_frechet_bounds(
    case: tuple[tuple[str, ...], np.ndarray],
) -> None:
    guardrails, distribution = case
    marginals, _ = distribution_moments(distribution, len(guardrails))
    assumptions = _exact_marginal_assumptions(guardrails, marginals)

    for event in ("and", "or"):
        result = identify(_query_for_event(guardrails, event), assumptions)
        classical = frechet_bounds(marginals, event=event)

        assert result.lower_bound == pytest.approx(classical.lower, abs=1.0e-8)
        assert result.upper_bound == pytest.approx(classical.upper, abs=1.0e-8)


@given(feasible_atom_laws())
@settings(max_examples=35, deadline=None)
def test_valid_side_constraints_only_narrow_identified_intervals(
    case: tuple[tuple[str, ...], np.ndarray],
) -> None:
    guardrails, distribution = case
    n_events = len(guardrails)
    marginals, pairwise = distribution_moments(distribution, n_events)
    base = _exact_marginal_assumptions(guardrails, marginals)
    pairs = list(combinations(range(n_events), 2))[: min(2, n_events)]
    tightened = _exact_pairwise_assumptions(base, pairwise, pairs)

    for event in ("and", "or"):
        query = _query_for_event(guardrails, event)
        base_result = identify(query, base)
        tightened_result = identify(query, tightened)
        true_probability = event_probability(distribution, event=event, n_events=n_events)

        assert tightened_result.lower_bound + 1.0e-8 >= base_result.lower_bound
        assert tightened_result.upper_bound <= base_result.upper_bound + 1.0e-8
        assert tightened_result.width <= base_result.width + 1.0e-8
        assert tightened_result.lower_bound <= true_probability + 1.0e-8
        assert true_probability <= tightened_result.upper_bound + 1.0e-8


def test_pairwise_equality_generalizes_existing_two_event_frechet_lp() -> None:
    guardrails = ("A", "B")
    assumptions = (
        AssumptionSet.empty(guardrails)
        .with_marginal_interval("A", 0.4, 0.4)
        .with_marginal_interval("B", 0.6, 0.6)
        .with_pairwise_joint_interval("A", "B", 0.3, 0.3)
    )

    result = identify(LinearQuery.joint(guardrails, "A", "B"), assumptions)
    frechet = frechet_bounds(
        [0.4, 0.6],
        pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.3)],
        event="and",
    )

    assert result.lower_bound == pytest.approx(0.3)
    assert result.upper_bound == pytest.approx(0.3)
    assert result.lower_bound == pytest.approx(frechet.lower)
    assert result.upper_bound == pytest.approx(frechet.upper)
    assert np.sum(result.lower_solution) == pytest.approx(1.0)
    assert np.all(result.lower_solution >= -1.0e-10)
    assert result.solver_status == "optimal"
    assert "pairwise:A&B:lower" in result.active_constraints
    assert len(result.assumptions_hash) == 64


def test_monotonicity_constraint_enforces_implication() -> None:
    guardrails = ("A", "B")
    assumptions = (
        AssumptionSet.empty(guardrails)
        .with_marginal_interval("A", 0.4, 0.4)
        .with_marginal_interval("B", 0.6, 0.6)
        .with_monotonicity("A", "B")
    )

    bad_atom_query = LinearQuery.from_predicate(
        guardrails,
        "P(A and not B)",
        lambda atom: atom["A"] == 1 and atom["B"] == 0,
    )
    bad_atom_result = assumptions.identify(bad_atom_query)
    joint_result = assumptions.identify(LinearQuery.joint(guardrails, "A", "B"))

    assert bad_atom_result.lower_bound == pytest.approx(0.0)
    assert bad_atom_result.upper_bound == pytest.approx(0.0)
    assert joint_result.lower_bound == pytest.approx(0.4)
    assert joint_result.upper_bound == pytest.approx(0.4)


def test_lower_bound_constraints_are_converted_to_standard_lp_form() -> None:
    guardrails = ("A", "B")
    assumptions = AssumptionSet.empty(guardrails).with_marginal_interval("A", 0.3, 1.0)

    result = assumptions.identify(LinearQuery.marginal(guardrails, "A"))

    assert result.lower_bound == pytest.approx(0.3)
    assert result.upper_bound == pytest.approx(1.0)


def test_infeasible_constraints_certify_inconsistent_assumptions() -> None:
    guardrails = ("A", "B")
    assumptions = (
        AssumptionSet.empty(guardrails)
        .with_marginal_interval("A", 0.9, 0.9)
        .with_marginal_interval("B", 0.9, 0.9)
        .with_pairwise_joint_interval("A", "B", 0.0, 0.1)
    )

    with pytest.raises(PartialIdentificationInfeasibleError, match="No atom distribution"):
        assumptions.identify(LinearQuery.joint(guardrails, "A", "B"))


def test_arbitrary_coefficients_are_validated_against_atom_count() -> None:
    guardrails = ("A", "B")

    with pytest.raises(ValueError, match=r"2\*\*K"):
        LinearQuery.from_coefficients(guardrails, "bad", [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="finite"):
        LinearConstraint.from_coefficients(
            guardrails,
            "bad",
            [0.0, 1.0, float("nan"), 0.0],
            "<=",
            0.5,
        )


def test_guardrail_and_interval_validation() -> None:
    with pytest.raises(ValueError, match="unique"):
        AssumptionSet.empty(("A", "A"))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AssumptionSet.empty(("A", "B")).with_marginal_interval("A", -0.1, 0.5)

    with pytest.raises(ValueError, match="lower <= upper"):
        AssumptionSet.empty(("A", "B")).with_pairwise_joint_interval("A", "B", 0.4, 0.2)


def test_stable_hash_depends_on_declared_assumption_surface() -> None:
    first = AssumptionSet.empty(("A", "B"), metadata={"source": "unit"}).with_marginal_interval(
        "A",
        0.2,
        0.4,
    )
    second = AssumptionSet.empty(("A", "B"), metadata={"source": "unit"}).with_marginal_interval(
        "A",
        0.2,
        0.4,
    )
    changed = AssumptionSet.empty(("A", "B"), metadata={"source": "unit"}).with_marginal_interval(
        "A",
        0.2,
        0.5,
    )

    assert first.stable_hash() == second.stable_hash()
    assert first.stable_hash() != changed.stable_hash()
