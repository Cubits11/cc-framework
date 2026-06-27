from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import cc.kernel as kernel
import cc.kernel.sensitivity as sensitivity
from cc.kernel.frechet_classes import (
    PairwiseDependence,
    atom_matrix,
    distribution_moments,
    event_probability,
    frechet_bounds,
)
from cc.kernel.sensitivity import (
    AssumptionSet,
    IdentificationInfeasibleError,
    LinearConstraint,
    LinearQuery,
    atom_predicate_mask,
    atoms_matching,
    atoms_where,
    enumerate_atoms,
    event_mask,
    identified_region,
)

ASSERT_TOL = 1.0e-8


def _exact_marginal_assumptions(
    guardrails: tuple[str, ...],
    marginals: np.ndarray | list[float],
) -> AssumptionSet:
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


def _constraint_by_name(assumptions: AssumptionSet, name: str) -> LinearConstraint:
    matches = [constraint for constraint in assumptions.constraints if constraint.name == name]
    assert len(matches) == 1
    return matches[0]


def _assert_solution_feasible(
    solution: np.ndarray,
    assumptions: AssumptionSet,
    *,
    tol: float = ASSERT_TOL,
) -> None:
    assert np.all(np.isfinite(solution))
    assert np.all(solution >= -tol)
    assert np.sum(solution) == pytest.approx(1.0, abs=tol)
    assert solution.shape == (1 << len(assumptions.guardrails),)

    for constraint in assumptions.constraints:
        lhs = float(constraint.coefficients @ solution)
        if constraint.sense == "==":
            assert lhs == pytest.approx(constraint.rhs, abs=tol)
        elif constraint.sense == "<=":
            assert lhs <= constraint.rhs + tol
        else:
            assert lhs + tol >= constraint.rhs


def _assert_result_solutions_feasible(
    result: sensitivity.IdentificationResult,
    assumptions: AssumptionSet,
) -> None:
    _assert_solution_feasible(result.lower_solution, assumptions)
    _assert_solution_feasible(result.upper_solution, assumptions)


def _bad_implication_mass(
    solution: np.ndarray,
    guardrails: tuple[str, ...],
    antecedent: str,
    consequent: str,
) -> float:
    atoms = enumerate_atoms(guardrails)
    antecedent_index = guardrails.index(antecedent)
    consequent_index = guardrails.index(consequent)
    bad_mask = (atoms[:, antecedent_index] == 1) & (atoms[:, consequent_index] == 0)
    return float(np.sum(solution[bad_mask]))


@st.composite
def feasible_atom_laws(draw: st.DrawFn) -> tuple[tuple[str, ...], np.ndarray]:
    n_events = draw(st.integers(min_value=2, max_value=4))
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


def probabilities() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


@st.composite
def two_event_joint_subinterval(draw: st.DrawFn) -> tuple[float, float, float, float]:
    alpha = draw(probabilities())
    beta = draw(probabilities())
    fh_lower = max(0.0, alpha + beta - 1.0)
    fh_upper = min(alpha, beta)
    joint_lower = draw(
        st.floats(
            min_value=fh_lower,
            max_value=fh_upper,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    joint_upper = draw(
        st.floats(
            min_value=joint_lower,
            max_value=fh_upper,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    return alpha, beta, joint_lower, joint_upper


@st.composite
def monotone_two_event_marginals(draw: st.DrawFn) -> tuple[float, float]:
    consequent_probability = draw(probabilities())
    antecedent_probability = draw(
        st.floats(
            min_value=0.0,
            max_value=consequent_probability,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    return antecedent_probability, consequent_probability


class TestAtomHelpers:
    def test_atom_order_is_little_endian_and_explicit_for_two_guardrails(self) -> None:
        guardrails = ("A", "B")

        atoms = enumerate_atoms(guardrails)

        assert np.array_equal(atoms, np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]))
        assert np.array_equal(atoms, atom_matrix(2))
        assert np.array_equal(event_mask(guardrails, "A"), [False, True, False, True])
        assert np.array_equal(event_mask(guardrails, "B"), [False, False, True, True])

    def test_atom_helpers_preserve_guardrail_order_and_support_predicates(self) -> None:
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

    def test_optimizer_solution_uses_the_documented_atom_order(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 1.0, 1.0)
            .with_marginal_interval("B", 0.0, 0.0)
        )
        query = LinearQuery.from_coefficients(guardrails, "atom-index", [0.0, 1.0, 2.0, 3.0])

        result = identified_region(query, assumptions)

        assert result.lower_bound == pytest.approx(1.0)
        assert result.upper_bound == pytest.approx(1.0)
        assert np.array_equal(result.lower_solution, [0.0, 1.0, 0.0, 0.0])
        assert np.array_equal(result.upper_solution, [0.0, 1.0, 0.0, 0.0])


class TestQueryConstruction:
    def test_two_guardrail_builtin_query_vectors_are_manual_coefficients(self) -> None:
        guardrails = ("A", "B")

        assert np.array_equal(LinearQuery.marginal(guardrails, "A").coefficients, [0, 1, 0, 1])
        assert np.array_equal(LinearQuery.marginal(guardrails, "B").coefficients, [0, 0, 1, 1])
        assert np.array_equal(LinearQuery.joint(guardrails, "A", "B").coefficients, [0, 0, 0, 1])
        assert np.array_equal(LinearQuery.union(guardrails, guardrails).coefficients, [0, 1, 1, 1])

    def test_three_guardrail_all_events_query_and_arbitrary_coefficients(self) -> None:
        guardrails = ("A", "B", "C")
        all_events = LinearQuery.intersection(guardrails, guardrails)
        coefficients = np.arange(8, dtype=float)

        assert np.array_equal(all_events.coefficients, [0, 0, 0, 0, 0, 0, 0, 1])
        assert np.array_equal(
            LinearQuery.from_coefficients(guardrails, "linear", coefficients).coefficients,
            coefficients,
        )

    def test_query_coefficients_are_validated_against_atom_count(self) -> None:
        with pytest.raises(ValueError, match=r"2\*\*K"):
            LinearQuery.from_coefficients(("A", "B"), "bad", [1.0, 0.0, 0.0])


class TestConstraintConstruction:
    def test_marginal_interval_and_exact_marginal_vectors(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 0.2, 0.4)
            .with_marginal_interval("B", 0.7, 0.7)
        )

        a_lower = _constraint_by_name(assumptions, "marginal:A:lower")
        a_upper = _constraint_by_name(assumptions, "marginal:A:upper")
        b_lower = _constraint_by_name(assumptions, "marginal:B:lower")
        b_upper = _constraint_by_name(assumptions, "marginal:B:upper")

        assert np.array_equal(a_lower.coefficients, [0, 1, 0, 1])
        assert a_lower.sense == ">="
        assert a_lower.rhs == pytest.approx(0.2)
        assert np.array_equal(a_upper.coefficients, [0, 1, 0, 1])
        assert a_upper.sense == "<="
        assert a_upper.rhs == pytest.approx(0.4)
        assert np.array_equal(b_lower.coefficients, [0, 0, 1, 1])
        assert b_lower.sense == ">="
        assert b_lower.rhs == pytest.approx(0.7)
        assert np.array_equal(b_upper.coefficients, [0, 0, 1, 1])
        assert b_upper.sense == "<="
        assert b_upper.rhs == pytest.approx(0.7)

    def test_pairwise_joint_interval_and_monotonicity_vectors(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_pairwise_joint_interval("A", "B", 0.1, 0.3)
            .with_monotonicity("A", "B")
        )

        joint_lower = _constraint_by_name(assumptions, "pairwise:A&B:lower")
        joint_upper = _constraint_by_name(assumptions, "pairwise:A&B:upper")
        monotone = _constraint_by_name(assumptions, "monotonicity:A=>B")

        assert np.array_equal(joint_lower.coefficients, [0, 0, 0, 1])
        assert joint_lower.sense == ">="
        assert joint_lower.rhs == pytest.approx(0.1)
        assert np.array_equal(joint_upper.coefficients, [0, 0, 0, 1])
        assert joint_upper.sense == "<="
        assert joint_upper.rhs == pytest.approx(0.3)
        assert np.array_equal(monotone.coefficients, [0, 1, 0, 0])
        assert monotone.sense == "=="
        assert monotone.rhs == pytest.approx(0.0)

    def test_arbitrary_equality_upper_and_lower_bound_constraints(self) -> None:
        guardrails = ("A", "B")
        coefficients = [0.0, 1.0, 1.0, 2.0]

        equality = LinearConstraint.equality(guardrails, "eq", coefficients, 0.5)
        upper = LinearConstraint.upper_bound(guardrails, "le", coefficients, 0.8)
        lower = LinearConstraint.lower_bound(guardrails, "ge", coefficients, 0.2)

        assert equality.sense == "=="
        assert np.array_equal(equality.coefficients, coefficients)
        assert equality.rhs == pytest.approx(0.5)
        assert upper.sense == "<="
        assert np.array_equal(upper.coefficients, coefficients)
        assert upper.rhs == pytest.approx(0.8)
        assert lower.sense == ">="
        assert np.array_equal(lower.coefficients, coefficients)
        assert lower.rhs == pytest.approx(0.2)

    def test_greater_equal_constraints_are_not_sign_flipped_in_the_public_surface(self) -> None:
        guardrails = ("A", "B")
        assumptions = AssumptionSet.empty(guardrails).with_linear_constraint(
            "P(A)>=0.3",
            LinearQuery.marginal(guardrails, "A").coefficients,
            ">=",
            0.3,
        )

        result = identified_region(LinearQuery.marginal(guardrails, "A"), assumptions)

        assert assumptions.constraints[0].sense == ">="
        assert np.array_equal(assumptions.constraints[0].coefficients, [0, 1, 0, 1])
        assert result.lower_bound == pytest.approx(0.3)
        assert result.upper_bound == pytest.approx(1.0)

    def test_arbitrary_constraint_coefficients_are_validated(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            LinearConstraint.from_coefficients(
                ("A", "B"),
                "bad",
                [0.0, 1.0, float("nan"), 0.0],
                "<=",
                0.5,
            )

        with pytest.raises(ValueError, match=r"2\*\*K"):
            LinearConstraint.from_coefficients(("A", "B"), "bad", [1.0, 0.0], "<=", 0.5)


class TestIdentifiedRegionFH:
    def test_marginal_only_engine_matches_classical_frechet_bounds_for_all_and_any(self) -> None:
        guardrails = ("A", "B", "C")
        marginals = np.asarray([0.2, 0.5, 0.7])
        assumptions = _exact_marginal_assumptions(guardrails, marginals)

        for event in ("and", "or"):
            result = identified_region(_query_for_event(guardrails, event), assumptions)
            classical = frechet_bounds(marginals, event=event)

            assert result.lower_bound == pytest.approx(classical.lower, abs=ASSERT_TOL)
            assert result.upper_bound == pytest.approx(classical.upper, abs=ASSERT_TOL)
            _assert_result_solutions_feasible(result, assumptions)

    def test_three_guardrails_all_marginals_fixed_have_hand_computed_intersection_bounds(
        self,
    ) -> None:
        guardrails = ("A", "B", "C")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 0.5, 0.5)
            .with_marginal_interval("B", 0.5, 0.5)
            .with_marginal_interval("C", 0.5, 0.5)
        )

        result = identified_region(LinearQuery.intersection(guardrails, guardrails), assumptions)

        assert result.lower_bound == pytest.approx(0.0, abs=ASSERT_TOL)
        assert result.upper_bound == pytest.approx(0.5, abs=ASSERT_TOL)
        _assert_result_solutions_feasible(result, assumptions)

    def test_pairwise_equality_generalizes_existing_two_event_frechet_lp(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 0.4, 0.4)
            .with_marginal_interval("B", 0.6, 0.6)
            .with_pairwise_joint_interval("A", "B", 0.3, 0.3)
        )

        result = identified_region(LinearQuery.joint(guardrails, "A", "B"), assumptions)
        frechet = frechet_bounds(
            [0.4, 0.6],
            pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.3)],
            event="and",
        )

        assert result.lower_bound == pytest.approx(0.3)
        assert result.upper_bound == pytest.approx(0.3)
        assert result.lower_bound == pytest.approx(frechet.lower)
        assert result.upper_bound == pytest.approx(frechet.upper)
        assert result.solver_status == "optimal"
        assert "pairwise:A&B:lower" in result.active_constraints
        assert len(result.assumptions_hash) == 64
        _assert_result_solutions_feasible(result, assumptions)

    def test_pairwise_joint_exact_constraint_identifies_union_probability(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 0.6, 0.6)
            .with_marginal_interval("B", 0.7, 0.7)
            .with_pairwise_joint_interval("A", "B", 0.4, 0.4)
        )

        result = identified_region(LinearQuery.union(guardrails, guardrails), assumptions)

        assert result.lower_bound == pytest.approx(0.9, abs=ASSERT_TOL)
        assert result.upper_bound == pytest.approx(0.9, abs=ASSERT_TOL)
        _assert_result_solutions_feasible(result, assumptions)


class TestIdentifiedRegionMonotonicity:
    def test_monotonic_chain_respects_implications_and_identifies_a_and_c(self) -> None:
        guardrails = ("A", "B", "C")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_monotonicity("A", "B")
            .with_monotonicity("B", "C")
            .with_marginal_interval("A", 0.3, 0.3)
            .with_marginal_interval("C", 0.7, 0.7)
        )

        result = identified_region(LinearQuery.intersection(guardrails, ("A", "C")), assumptions)

        assert result.lower_bound == pytest.approx(0.3, abs=ASSERT_TOL)
        assert result.upper_bound == pytest.approx(0.3, abs=ASSERT_TOL)
        assert _bad_implication_mass(result.lower_solution, guardrails, "A", "B") == pytest.approx(
            0.0,
            abs=ASSERT_TOL,
        )
        assert _bad_implication_mass(result.upper_solution, guardrails, "A", "B") == pytest.approx(
            0.0,
            abs=ASSERT_TOL,
        )
        assert _bad_implication_mass(result.lower_solution, guardrails, "B", "C") == pytest.approx(
            0.0,
            abs=ASSERT_TOL,
        )
        assert _bad_implication_mass(result.upper_solution, guardrails, "B", "C") == pytest.approx(
            0.0,
            abs=ASSERT_TOL,
        )
        _assert_result_solutions_feasible(result, assumptions)

    def test_active_constraints_are_endpoint_specific(self) -> None:
        guardrails = ("A", "B")
        assumptions = AssumptionSet.empty(guardrails).with_marginal_interval("A", 0.25, 0.75)

        result = identified_region(LinearQuery.marginal(guardrails, "A"), assumptions)

        assert result.lower_bound == pytest.approx(0.25)
        assert result.upper_bound == pytest.approx(0.75)
        assert result.active_constraints_lower == ("marginal:A:lower",)
        assert result.active_constraints_upper == ("marginal:A:upper",)
        assert result.active_constraints == ("marginal:A:lower", "marginal:A:upper")

    def test_valid_side_constraints_only_narrow_identified_intervals(self) -> None:
        guardrails = ("A", "B", "C")
        distribution = np.asarray([0.05, 0.10, 0.15, 0.05, 0.20, 0.10, 0.15, 0.20])
        marginals, pairwise = distribution_moments(distribution, len(guardrails))
        base = _exact_marginal_assumptions(guardrails, marginals)
        tightened = _exact_pairwise_assumptions(base, pairwise, list(combinations(range(3), 2)))

        for event in ("and", "or"):
            query = _query_for_event(guardrails, event)
            base_result = identified_region(query, base)
            tightened_result = identified_region(query, tightened)
            true_probability = event_probability(distribution, event=event, n_events=3)

            assert tightened_result.lower_bound + ASSERT_TOL >= base_result.lower_bound
            assert tightened_result.upper_bound <= base_result.upper_bound + ASSERT_TOL
            assert tightened_result.width <= base_result.width + ASSERT_TOL
            assert tightened_result.lower_bound <= true_probability + ASSERT_TOL
            assert true_probability <= tightened_result.upper_bound + ASSERT_TOL


class TestInfeasibility:
    def test_guardrail_and_interval_validation(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            AssumptionSet.empty(("A", "A"))

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            AssumptionSet.empty(("A", "B")).with_marginal_interval("A", -0.1, 0.5)

        with pytest.raises(ValueError, match="lower <= upper"):
            AssumptionSet.empty(("A", "B")).with_pairwise_joint_interval("A", "B", 0.4, 0.2)

    def test_marginal_lower_bound_above_upper_bound_is_validation_error(self) -> None:
        with pytest.raises(ValueError, match="lower <= upper"):
            AssumptionSet.empty(("A", "B")).with_marginal_interval("A", 0.8, 0.2)

    def test_conflicting_exact_marginals_are_infeasible(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 0.8, 0.8)
            .with_marginal_interval("A", 0.2, 0.2)
        )

        with pytest.raises(IdentificationInfeasibleError, match="No atom distribution"):
            identified_region(LinearQuery.marginal(guardrails, "A"), assumptions)

    def test_joint_probability_cannot_exceed_its_marginal(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 0.4, 0.4)
            .with_pairwise_joint_interval("A", "B", 0.5, 0.5)
        )

        with pytest.raises(IdentificationInfeasibleError, match="No atom distribution"):
            identified_region(LinearQuery.joint(guardrails, "A", "B"), assumptions)

    def test_monotonicity_is_infeasible_when_antecedent_marginal_exceeds_consequent(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_monotonicity("A", "B")
            .with_marginal_interval("A", 0.7, 0.7)
            .with_marginal_interval("B", 0.4, 0.4)
        )

        with pytest.raises(IdentificationInfeasibleError, match="No atom distribution"):
            identified_region(LinearQuery.marginal(guardrails, "A"), assumptions)

    def test_no_silent_fake_interval_for_inconsistent_pairwise_assumptions(self) -> None:
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_marginal_interval("A", 0.9, 0.9)
            .with_marginal_interval("B", 0.9, 0.9)
            .with_pairwise_joint_interval("A", "B", 0.0, 0.1)
        )

        with pytest.raises(IdentificationInfeasibleError, match="No atom distribution"):
            identified_region(LinearQuery.joint(guardrails, "A", "B"), assumptions)


class TestAssumptionHashing:
    def test_hash_is_stable_for_same_assumption_surface(self) -> None:
        first = AssumptionSet.empty(("A", "B"), metadata={"source": "unit"}).with_marginal_interval(
            "A",
            0.2,
            0.4,
        )
        second = AssumptionSet.empty(
            ("A", "B"),
            metadata={"source": "unit"},
        ).with_marginal_interval("A", 0.2, 0.4)

        assert first.stable_hash() == second.stable_hash()
        assert first.stable_hash() == first.stable_hash()

    def test_hash_uses_deterministic_float_and_metadata_serialization(self) -> None:
        first = AssumptionSet.empty(("A",), metadata={"b": 2, "a": 0.125}).with_linear_constraint(
            "float",
            [0.0, 0.1],
            "<=",
            0.3,
        )
        second = AssumptionSet.empty(("A",), metadata={"a": 0.125, "b": 2}).with_linear_constraint(
            "float",
            np.asarray([0.0, 0.1]),
            "<=",
            0.3,
        )

        assert first.stable_hash() == second.stable_hash()

    def test_hash_changes_when_constraint_value_name_or_guardrail_order_changes(self) -> None:
        base = AssumptionSet.empty(("A", "B")).with_marginal_interval("A", 0.2, 0.4)
        changed_value = AssumptionSet.empty(("A", "B")).with_marginal_interval("A", 0.2, 0.5)
        changed_order = AssumptionSet.empty(("B", "A")).with_marginal_interval("A", 0.2, 0.4)
        changed_name = AssumptionSet.empty(("A", "B")).with_linear_constraint(
            "renamed",
            LinearQuery.marginal(("A", "B"), "A").coefficients,
            ">=",
            0.2,
        )

        assert base.stable_hash() != changed_value.stable_hash()
        assert base.stable_hash() != changed_order.stable_hash()
        assert (
            AssumptionSet.empty(("A", "B"))
            .with_linear_constraint(
                "name-1",
                LinearQuery.marginal(("A", "B"), "A").coefficients,
                ">=",
                0.2,
            )
            .stable_hash()
            != changed_name.stable_hash()
        )

    def test_constraint_descriptions_do_not_change_the_hash_by_design(self) -> None:
        guardrails = ("A", "B")
        coefficients = LinearQuery.marginal(guardrails, "A").coefficients
        first = AssumptionSet.empty(guardrails).with_linear_constraint(
            "same",
            coefficients,
            ">=",
            0.2,
            description="first wording",
        )
        second = AssumptionSet.empty(guardrails).with_linear_constraint(
            "same",
            coefficients,
            ">=",
            0.2,
            description="second wording",
        )

        assert first.stable_hash() == second.stable_hash()


class TestApiSurface:
    def test_sensitivity_all_exposes_only_the_stable_theorem_api(self) -> None:
        assert set(sensitivity.__all__) == {
            "AssumptionSet",
            "IdentificationInfeasibleError",
            "IdentificationResult",
            "LinearConstraint",
            "LinearQuery",
            "identified_region",
        }

    def test_kernel_package_does_not_reexport_low_level_atom_helpers(self) -> None:
        assert "identified_region" in kernel.__all__
        assert "identify" not in kernel.__all__
        assert "event_mask" not in kernel.__all__
        assert not hasattr(kernel, "identify")
        assert not hasattr(kernel, "event_mask")


class TestPropertyBasedSensitivity:
    @given(probabilities(), probabilities())
    @settings(max_examples=60, deadline=None)
    def test_random_two_event_joint_bounds_match_fh_formulas(
        self,
        alpha: float,
        beta: float,
    ) -> None:
        guardrails = ("A", "B")
        assumptions = _exact_marginal_assumptions(guardrails, [alpha, beta])

        result = identified_region(LinearQuery.joint(guardrails, "A", "B"), assumptions)

        assert result.lower_bound == pytest.approx(max(0.0, alpha + beta - 1.0), abs=ASSERT_TOL)
        assert result.upper_bound == pytest.approx(min(alpha, beta), abs=ASSERT_TOL)
        _assert_result_solutions_feasible(result, assumptions)

    @given(two_event_joint_subinterval())
    @settings(max_examples=60, deadline=None)
    def test_valid_joint_subinterval_cannot_widen_identified_interval(
        self,
        case: tuple[float, float, float, float],
    ) -> None:
        alpha, beta, joint_lower, joint_upper = case
        guardrails = ("A", "B")
        base = _exact_marginal_assumptions(guardrails, [alpha, beta])
        tightened = base.with_pairwise_joint_interval("A", "B", joint_lower, joint_upper)
        query = LinearQuery.joint(guardrails, "A", "B")

        base_result = identified_region(query, base)
        tightened_result = identified_region(query, tightened)

        assert tightened_result.lower_bound + ASSERT_TOL >= base_result.lower_bound
        assert tightened_result.upper_bound <= base_result.upper_bound + ASSERT_TOL
        assert tightened_result.width <= base_result.width + ASSERT_TOL
        _assert_result_solutions_feasible(tightened_result, tightened)

    @given(feasible_atom_laws())
    @settings(max_examples=45, deadline=None)
    def test_returned_optimizer_solutions_are_feasible_probability_vectors(
        self,
        case: tuple[tuple[str, ...], np.ndarray],
    ) -> None:
        guardrails, distribution = case
        marginals, pairwise = distribution_moments(distribution, len(guardrails))
        assumptions = _exact_marginal_assumptions(guardrails, marginals)
        pairs = list(combinations(range(len(guardrails)), 2))[:2]
        assumptions = _exact_pairwise_assumptions(assumptions, pairwise, pairs)

        result = identified_region(LinearQuery.union(guardrails, guardrails), assumptions)

        _assert_result_solutions_feasible(result, assumptions)

    @given(monotone_two_event_marginals())
    @settings(max_examples=60, deadline=None)
    def test_random_monotonic_implication_places_no_mass_on_forbidden_atom(
        self,
        marginals: tuple[float, float],
    ) -> None:
        p_a, p_b = marginals
        guardrails = ("A", "B")
        assumptions = (
            AssumptionSet.empty(guardrails)
            .with_monotonicity("A", "B")
            .with_marginal_interval("A", p_a, p_a)
            .with_marginal_interval("B", p_b, p_b)
        )

        result = identified_region(LinearQuery.union(guardrails, guardrails), assumptions)

        assert _bad_implication_mass(result.lower_solution, guardrails, "A", "B") == pytest.approx(
            0.0,
            abs=ASSERT_TOL,
        )
        assert _bad_implication_mass(result.upper_solution, guardrails, "A", "B") == pytest.approx(
            0.0,
            abs=ASSERT_TOL,
        )
        _assert_result_solutions_feasible(result, assumptions)
