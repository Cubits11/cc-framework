"""Narrow paper-core public surface for the kernel.

This module is the docs-backed import boundary for Paper Core v0.3.  The
broader ``cc.kernel`` package remains available for compatibility and
experimental modules, but paper-facing examples and artifact scripts should
import through this surface.
"""

from __future__ import annotations

from cc.kernel import frechet_classes, metrics, sample_complexity, sensitivity
from cc.kernel.frechet_classes import (
    DependenceKind,
    EventKind,
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
from cc.kernel.metrics import (
    MetricDomainError,
    cc_gain,
    cc_shift,
    fh_position,
    fh_width,
    independence_regret,
    independent_event_probability,
)
from cc.kernel.sample_complexity import (
    BernoulliRateInterval,
    FiniteSampleIdentificationResult,
    PairwiseCountEvidence,
    SingletonCountEvidence,
    assumption_set_from_counts,
    bernoulli_confidence_interval,
    bernoulli_rate_count,
    composition_bounds_from_counts,
    hoeffding_radius,
    pairwise_rate_count,
    sample_size_for_radius,
    simultaneous_bernoulli_radius,
    simultaneous_sample_size,
)
from cc.kernel.sensitivity import (
    AssumptionSet,
    IdentificationInfeasibleError,
    IdentificationResult,
    LinearConstraint,
    LinearQuery,
    enumerate_atoms,
    identified_region,
)

__all__ = [
    "AssumptionSet",
    "BernoulliRateInterval",
    "DependenceKind",
    "EventKind",
    "FiniteSampleIdentificationResult",
    "FrechetBoundResult",
    "FrechetClassInfeasibleError",
    "IdentificationInfeasibleError",
    "IdentificationResult",
    "LinearConstraint",
    "LinearQuery",
    "MetricDomainError",
    "PairwiseCountEvidence",
    "PairwiseDependence",
    "PairwiseJointConstraint",
    "SingletonCountEvidence",
    "assumption_set_from_counts",
    "atom_matrix",
    "bernoulli_confidence_interval",
    "bernoulli_rate_count",
    "cc_gain",
    "cc_shift",
    "classical_frechet_bounds",
    "composition_bounds_from_counts",
    "dependence_to_joint_probability",
    "distribution_moments",
    "enumerate_atoms",
    "event_probability",
    "fh_position",
    "fh_width",
    "frechet_bounds",
    "frechet_classes",
    "hoeffding_radius",
    "identified_region",
    "improved_frechet_bounds",
    "independence_regret",
    "independent_event_probability",
    "joint_probability_to_dependence",
    "metrics",
    "pairwise_correlation_bounds",
    "pairwise_rate_count",
    "random_feasible_distribution",
    "sample_binary_vectors",
    "sample_complexity",
    "sample_size_for_radius",
    "sensitivity",
    "simultaneous_bernoulli_radius",
    "simultaneous_sample_size",
]
