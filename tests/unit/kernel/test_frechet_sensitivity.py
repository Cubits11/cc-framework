from __future__ import annotations

import numpy as np
import pytest

import cc.kernel as kernel
from cc.kernel.frechet_sensitivity import (
    LinearAtomQuery,
    PairwiseJointEquality,
    all_events_query,
    any_event_query,
    binary_atoms,
    sharp_frechet_bounds,
)
from cc.kernel.sensitivity import IdentificationInfeasibleError


def test_classical_and_bounds_are_recovered_from_marginal_equalities() -> None:
    result = sharp_frechet_bounds([0.8, 0.9, 0.95], all_events_query(3))

    assert result.lower_bound == pytest.approx(0.65)
    assert result.upper_bound == pytest.approx(0.8)
    assert result.width == pytest.approx(0.15)
    assert np.sum(result.lower_distribution) == pytest.approx(1.0)
    assert np.sum(result.upper_distribution) == pytest.approx(1.0)


def test_classical_or_bounds_are_recovered_from_marginal_equalities() -> None:
    result = sharp_frechet_bounds([0.2, 0.5, 0.8], any_event_query(3))

    assert result.lower_bound == pytest.approx(0.8)
    assert result.upper_bound == pytest.approx(1.0)


def test_pairwise_constraints_contain_generating_distribution() -> None:
    distribution = np.asarray([0.1, 0.2, 0.15, 0.05, 0.05, 0.1, 0.2, 0.15], dtype=float)
    atoms = binary_atoms(3).astype(float)
    marginals = atoms.T @ distribution
    pairwise = [
        PairwiseJointEquality(0, 1, float((atoms[:, 0] * atoms[:, 1]) @ distribution)),
        PairwiseJointEquality(1, 2, float((atoms[:, 1] * atoms[:, 2]) @ distribution)),
    ]
    query = all_events_query(3)
    true_probability = float(query.coefficients @ distribution)

    result = sharp_frechet_bounds(marginals, query, pairwise=pairwise)

    assert result.lower_bound <= true_probability + 1.0e-8
    assert true_probability <= result.upper_bound + 1.0e-8
    assert result.width <= sharp_frechet_bounds(marginals, query).width + 1.0e-8


def test_query_coefficients_must_match_atom_count() -> None:
    bad_query = LinearAtomQuery(name="bad", coefficients=np.asarray([0.0, 1.0, 0.0]))

    with pytest.raises(ValueError, match="2\\*\\*n_events"):
        sharp_frechet_bounds([0.5, 0.5], bad_query)


def test_infeasible_pairwise_constraints_raise_identification_error() -> None:
    pairwise = [
        PairwiseJointEquality(0, 1, 0.0),
        PairwiseJointEquality(0, 2, 0.0),
        PairwiseJointEquality(1, 2, 0.0),
    ]

    with pytest.raises(IdentificationInfeasibleError):
        sharp_frechet_bounds([0.5, 0.5, 0.5], all_events_query(3), pairwise=pairwise)


def test_kernel_lazy_exports_new_surface() -> None:
    assert kernel.sharp_frechet_bounds is sharp_frechet_bounds
    assert kernel.all_events_query(2).coefficients.shape == (4,)
