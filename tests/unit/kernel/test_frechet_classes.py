from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cc.kernel import frechet_classes as fc
from cc.kernel.frechet_classes import (
    FrechetClassInfeasibleError,
    PairwiseDependence,
    atom_matrix,
    classical_frechet_bounds,
    dependence_to_joint_probability,
    distribution_moments,
    event_probability,
    frechet_bounds,
    joint_probability_to_dependence,
    pairwise_correlation_bounds,
    random_feasible_distribution,
    sample_binary_vectors,
)


@st.composite
def feasible_binary_laws(draw: st.DrawFn) -> tuple[int, np.ndarray, list[PairwiseDependence]]:
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
    marginals, pairwise_joint = distribution_moments(distribution, n_events)

    pairs = list(combinations(range(n_events), 2))
    constraint_count = draw(st.integers(min_value=0, max_value=min(n_events, len(pairs))))
    constraints: list[PairwiseDependence] = []
    for k, (i, j) in enumerate(pairs[:constraint_count]):
        kind = "spearman_rho" if k % 2 == 0 else "kendall_tau"
        value = joint_probability_to_dependence(
            float(marginals[i]),
            float(marginals[j]),
            float(pairwise_joint[i, j]),
            kind=kind,
        )
        constraints.append(PairwiseDependence(i=i, j=j, kind=kind, value=value))

    return n_events, distribution, constraints


@given(feasible_binary_laws())
@settings(max_examples=45, deadline=None)
def test_bounds_contain_feasible_distribution(
    case: tuple[int, np.ndarray, list[PairwiseDependence]],
) -> None:
    n_events, distribution, constraints = case
    marginals, _ = distribution_moments(distribution, n_events)

    for event in ("and", "or"):
        result = frechet_bounds(marginals, pairwise=constraints, event=event)
        true_probability = event_probability(distribution, event=event, n_events=n_events)

        assert result.lower <= true_probability + 1.0e-8
        assert true_probability <= result.upper + 1.0e-8


@given(feasible_binary_laws())
@settings(max_examples=45, deadline=None)
def test_side_information_only_tightens_classical_bounds(
    case: tuple[int, np.ndarray, list[PairwiseDependence]],
) -> None:
    n_events, distribution, constraints = case
    marginals, _ = distribution_moments(distribution, n_events)

    for event in ("and", "or"):
        classical = frechet_bounds(marginals, event=event)
        improved = frechet_bounds(marginals, pairwise=constraints, event=event)

        assert improved.lower + 1.0e-8 >= classical.lower
        assert improved.upper <= classical.upper + 1.0e-8
        assert improved.width <= classical.width + 1.0e-8


@pytest.mark.parametrize(
    ("marginals", "event", "expected"),
    [
        ([0.3, 0.7], "and", (0.0, 0.3)),
        ([0.3, 0.7], "or", (0.7, 1.0)),
        ([0.2, 0.5, 0.8], "and", (0.0, 0.2)),
        ([0.2, 0.5, 0.8], "or", (0.8, 1.0)),
        ([0.8, 0.9, 0.95], "and", (0.65, 0.8)),
        ([0.8, 0.9, 0.95], "or", (0.95, 1.0)),
    ],
)
def test_textbook_classical_frechet_cases(
    marginals: list[float],
    event: str,
    expected: tuple[float, float],
) -> None:
    assert classical_frechet_bounds(marginals, event=event) == pytest.approx(expected)
    result = frechet_bounds(marginals, event=event)
    assert (result.lower, result.upper) == pytest.approx(expected)


def test_rank_side_information_fixes_two_event_joint_law() -> None:
    marginals = np.asarray([0.4, 0.6], dtype=float)
    joint = 0.3
    rho = joint_probability_to_dependence(0.4, 0.6, joint, kind="spearman_rho")

    and_result = frechet_bounds(
        marginals,
        pairwise=[PairwiseDependence(0, 1, "kendall_tau", rho)],
        event="and",
    )
    or_result = frechet_bounds(
        marginals,
        pairwise=[PairwiseDependence(0, 1, "spearman_rho", rho)],
        event="or",
    )

    assert and_result.lower == pytest.approx(joint)
    assert and_result.upper == pytest.approx(joint)
    assert or_result.lower == pytest.approx(0.7)
    assert or_result.upper == pytest.approx(0.7)


def test_pairwise_correlation_bounds_and_conversion_are_inverse() -> None:
    lower, upper = pairwise_correlation_bounds(0.2, 0.8)
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(0.25)

    joint = dependence_to_joint_probability(0.2, 0.8, upper, "spearman_rho")
    assert joint == pytest.approx(0.2)
    assert joint_probability_to_dependence(0.2, 0.8, joint, kind="kendall_tau") == pytest.approx(
        upper
    )


def test_incompatible_pairwise_correlation_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        dependence_to_joint_probability(0.2, 0.8, 1.0, "spearman_rho")


def test_globally_infeasible_pairwise_constraints_are_rejected() -> None:
    constraints = [
        PairwiseDependence(0, 1, "joint_probability", 0.0),
        PairwiseDependence(0, 2, "joint_probability", 0.0),
        PairwiseDependence(1, 2, "joint_probability", 0.0),
    ]
    with pytest.raises(FrechetClassInfeasibleError):
        frechet_bounds([0.5, 0.5, 0.5], pairwise=constraints)


def test_duplicate_pairwise_constraints_must_agree() -> None:
    constraints = [
        PairwiseDependence(0, 1, "joint_probability", 0.2),
        PairwiseDependence(1, 0, "joint_probability", 0.25),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        frechet_bounds([0.5, 0.5], pairwise=constraints)


def test_degenerate_rank_dependence_requires_joint_probability() -> None:
    with pytest.raises(ValueError, match="undefined"):
        dependence_to_joint_probability(1.0, 0.4, 0.0, "kendall_tau")

    assert dependence_to_joint_probability(1.0, 0.4, 0.4, "joint_probability") == pytest.approx(0.4)


def test_random_feasible_distribution_matches_requested_moments() -> None:
    marginals = np.asarray([0.4, 0.5, 0.6], dtype=float)
    constraints = [
        PairwiseDependence(0, 1, "spearman_rho", 0.0),
        PairwiseDependence(1, 2, "kendall_tau", 0.0),
    ]
    distribution = random_feasible_distribution(marginals, pairwise=constraints, rng=123)
    observed_marginals, observed_pairwise = distribution_moments(distribution, 3)

    assert observed_marginals == pytest.approx(marginals)
    assert observed_pairwise[0, 1] == pytest.approx(0.2)
    assert observed_pairwise[1, 2] == pytest.approx(0.3)


def test_sampling_uses_atom_matrix_ordering() -> None:
    distribution = np.zeros(8, dtype=float)
    distribution[-1] = 1.0
    samples = sample_binary_vectors(distribution, 4, n_events=3, rng=1)

    assert samples.shape == (4, 3)
    assert np.array_equal(samples, np.ones((4, 3), dtype=int))
    assert np.array_equal(atom_matrix(3)[-1], np.ones(3, dtype=int))


def test_invalid_atom_matrix_inputs() -> None:
    with pytest.raises(TypeError):
        atom_matrix(2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        atom_matrix(0)
    with pytest.raises(ValueError, match="too large"):
        atom_matrix(25)


def test_improved_alias_and_joint_dependence_conversion() -> None:
    result = fc.improved_frechet_bounds(
        [0.5, 0.5],
        pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.25)],
    )
    assert result.lower == pytest.approx(0.25)
    assert joint_probability_to_dependence(
        0.5,
        0.5,
        0.25,
        kind="joint_probability",
    ) == pytest.approx(0.25)


def test_invalid_dependence_values_and_kind() -> None:
    with pytest.raises(ValueError, match="finite"):
        dependence_to_joint_probability(0.5, 0.5, float("nan"), "spearman_rho")
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        dependence_to_joint_probability(0.5, 0.5, 2.0, "spearman_rho")
    with pytest.raises(ValueError, match="Unknown"):
        dependence_to_joint_probability(0.5, 0.5, 0.0, "mystery")  # type: ignore[arg-type]


def test_invalid_marginal_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        frechet_bounds([[0.2, 0.3]])
    with pytest.raises(ValueError, match="at least two"):
        frechet_bounds([0.2])
    with pytest.raises(ValueError, match="finite"):
        frechet_bounds([0.2, float("inf")])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        frechet_bounds([0.2, 1.2])
    with pytest.raises(ValueError, match="event"):
        frechet_bounds([0.2, 0.3], event="xor")  # type: ignore[arg-type]


def test_invalid_distribution_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        event_probability([[0.25, 0.25], [0.25, 0.25]])
    with pytest.raises(ValueError, match="finite"):
        event_probability([0.5, float("nan"), 0.25, 0.25])
    with pytest.raises(ValueError, match="nonnegative"):
        event_probability([0.5, -0.1, 0.3, 0.3])
    with pytest.raises(ValueError, match="sum to 1"):
        event_probability([0.2, 0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="power of two"):
        event_probability([0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="positive integer"):
        event_probability([0.5, 0.5], n_events=0)
    with pytest.raises(ValueError, match="does not equal"):
        event_probability([0.25, 0.25, 0.25, 0.25], n_events=3)


def test_invalid_pairwise_constraint_metadata() -> None:
    with pytest.raises(TypeError, match="indices"):
        frechet_bounds(
            [0.5, 0.5],
            pairwise=[PairwiseDependence("0", 1, "joint_probability", 0.25)],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="distinct"):
        frechet_bounds(
            [0.5, 0.5],
            pairwise=[PairwiseDependence(0, 0, "joint_probability", 0.5)],
        )
    with pytest.raises(ValueError, match="outside"):
        frechet_bounds(
            [0.5, 0.5],
            pairwise=[PairwiseDependence(0, 2, "joint_probability", 0.25)],
        )


def test_invalid_sample_size() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        sample_binary_vectors([0.25, 0.25, 0.25, 0.25], -1)


def test_defensive_empty_interval_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_solve(
        objective: np.ndarray,
        a_eq: np.ndarray,
        b_eq: np.ndarray,
        *,
        maximize: bool,
        feasibility_tol: float,
    ) -> tuple[float, np.ndarray]:
        del objective, a_eq, b_eq, feasibility_tol
        return (0.2 if maximize else 0.8), np.full(4, 0.25)

    monkeypatch.setattr(fc, "_solve_event_lp", fake_solve)
    with pytest.raises(FrechetClassInfeasibleError, match="empty interval"):
        frechet_bounds([0.5, 0.5], return_distributions=True)


def test_defensive_lp_residual_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_linprog(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(success=True, x=np.asarray([1.0, 0.0, 0.0, 0.0]), message="")

    monkeypatch.setattr(fc, "linprog", fake_linprog)
    with pytest.raises(FrechetClassInfeasibleError, match="violates equality"):
        frechet_bounds([0.5, 0.5], return_distributions=True)


def test_defensive_lp_negative_and_zero_mass_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_negative_linprog(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(success=True, x=np.asarray([-0.2, 1.2, 0.0, 0.0]), message="")

    monkeypatch.setattr(fc, "linprog", fake_negative_linprog)
    with pytest.raises(FrechetClassInfeasibleError, match="negative"):
        frechet_bounds([0.5, 0.5], return_distributions=True)

    def fake_zero_linprog(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(success=True, x=np.zeros(4, dtype=float), message="")

    monkeypatch.setattr(fc, "linprog", fake_zero_linprog)
    with pytest.raises(FrechetClassInfeasibleError, match="zero-mass"):
        frechet_bounds([0.5, 0.5], return_distributions=True)
