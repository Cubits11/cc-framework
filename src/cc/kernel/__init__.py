"""Numerical kernels for dependency-sensitive probability bounds."""

from cc.kernel.frechet_classes import (
    FrechetBoundResult,
    FrechetClassInfeasibleError,
    PairwiseDependence,
    PairwiseJointConstraint,
    atom_matrix,
    classical_frechet_bounds,
    dependence_to_joint_probability,
    distribution_moments,
    event_probability,
    frechet_bounds,
    improved_frechet_bounds,
    joint_probability_to_dependence,
    pairwise_correlation_bounds,
    random_feasible_distribution,
    sample_binary_vectors,
)

__all__ = [
    "FrechetBoundResult",
    "FrechetClassInfeasibleError",
    "PairwiseDependence",
    "PairwiseJointConstraint",
    "atom_matrix",
    "classical_frechet_bounds",
    "dependence_to_joint_probability",
    "distribution_moments",
    "event_probability",
    "frechet_bounds",
    "improved_frechet_bounds",
    "joint_probability_to_dependence",
    "pairwise_correlation_bounds",
    "random_feasible_distribution",
    "sample_binary_vectors",
]
