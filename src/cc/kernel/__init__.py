"""Numerical kernels for dependency-sensitive probability bounds.

The broad ``cc.kernel`` package is a compatibility aggregate.  Exports are
loaded lazily so importing the narrower ``cc.kernel.strict`` surface does not
eagerly initialize experimental kernel modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "cc.kernel.causal": (
        "ClusterATEEstimate",
        "ClusteredTwoWorldData",
        "compare_naive_vs_clustered_uncertainty",
        "difference_in_means",
        "empirical_intraclass_correlation",
        "estimate_clustered_ate",
        "generate_synthetic_clustered_two_world",
        "naive_standard_error",
        "run_coverage_simulation",
    ),
    "cc.kernel.ccf_models": (
        "DependenceEvidence",
        "ModelRecommendation",
        "alpha_factor",
        "alpha_factor_basic_event_probabilities",
        "assert_within_fh_envelope",
        "beta_factor",
        "beta_factor_basic_event_probabilities",
        "mgl_basic_event_probabilities",
        "multiple_greek_letter",
        "partition_failure_probability",
        "recommend_model",
    ),
    "cc.kernel.cliff": (
        "CliffCertificate",
        "CopulaCandidateFit",
        "CopulaFitResult",
        "TailDependenceCI",
        "TailDependenceEstimate",
        "bootstrap_tail_dependence_ci",
        "cliff_certificate",
        "estimate_tail_dependence",
        "fit_copula_family",
        "sample_copula",
        "theoretical_tail_dependence",
    ),
    "cc.kernel.frechet_classes": (
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
    ),
    "cc.kernel.frechet_sensitivity": (
        "FrechetSensitivityResult",
        "LinearAtomQuery",
        "MarginalEquality",
        "PairwiseJointEquality",
        "all_events_query",
        "any_event_query",
        "binary_atoms",
        "event_query",
        "pairwise_joint_from_phi",
        "sharp_frechet_bounds",
    ),
    "cc.kernel.metrics": (
        "MetricDomainError",
        "cc_gain",
        "cc_shift",
        "fh_position",
        "fh_width",
        "independence_regret",
        "independent_event_probability",
    ),
    "cc.kernel.sample_complexity": (
        "bernoulli_rate_count",
        "hoeffding_radius",
        "pairwise_rate_count",
        "sample_size_for_radius",
        "simultaneous_bernoulli_radius",
        "simultaneous_sample_size",
    ),
    "cc.kernel.sensitivity": (
        "AssumptionSet",
        "IdentificationInfeasibleError",
        "IdentificationResult",
        "LinearConstraint",
        "LinearQuery",
        "identified_region",
    ),
    "cc.kernel.sequential": (
        "AnytimeBernoulliResult",
        "AnytimeBernoulliTester",
        "CalibrationResult",
        "PowerResult",
        "betting_fractions",
        "calibrate_false_stop_rate",
        "fixed_sample_size_one_sided",
        "independent_joint_miss_baseline",
        "simulate_power_curve",
    ),
    "cc.kernel.stress": (
        "BaselineDependence",
        "ConditionalComposedRiskResult",
        "StressBudget",
        "StressMetric",
        "StressTestResult",
        "composed_risk_given_guardrail_X_failure",
        "composed_risk_given_guardrail_x_failure",
        "stress_test",
    ),
}

_SUBMODULES = frozenset(module.rpartition(".")[2] for module in _MODULE_EXPORTS)
_EXPORT_TO_MODULE = {
    name: module for module, exports in _MODULE_EXPORTS.items() for name in exports
}

__all__ = [
    "AnytimeBernoulliResult",
    "AnytimeBernoulliTester",
    "AssumptionSet",
    "BaselineDependence",
    "CalibrationResult",
    "CliffCertificate",
    "ClusterATEEstimate",
    "ClusteredTwoWorldData",
    "ConditionalComposedRiskResult",
    "CopulaCandidateFit",
    "CopulaFitResult",
    "DependenceEvidence",
    "FrechetBoundResult",
    "FrechetClassInfeasibleError",
    "FrechetSensitivityResult",
    "IdentificationInfeasibleError",
    "IdentificationResult",
    "LinearAtomQuery",
    "LinearConstraint",
    "LinearQuery",
    "MarginalEquality",
    "MetricDomainError",
    "ModelRecommendation",
    "PairwiseDependence",
    "PairwiseJointConstraint",
    "PairwiseJointEquality",
    "PowerResult",
    "StressBudget",
    "StressMetric",
    "StressTestResult",
    "TailDependenceCI",
    "TailDependenceEstimate",
    "alpha_factor",
    "alpha_factor_basic_event_probabilities",
    "all_events_query",
    "any_event_query",
    "assert_within_fh_envelope",
    "atom_matrix",
    "bernoulli_rate_count",
    "binary_atoms",
    "beta_factor",
    "beta_factor_basic_event_probabilities",
    "betting_fractions",
    "bootstrap_tail_dependence_ci",
    "calibrate_false_stop_rate",
    "cc_gain",
    "cc_shift",
    "classical_frechet_bounds",
    "cliff_certificate",
    "compare_naive_vs_clustered_uncertainty",
    "composed_risk_given_guardrail_X_failure",
    "composed_risk_given_guardrail_x_failure",
    "dependence_to_joint_probability",
    "difference_in_means",
    "distribution_moments",
    "empirical_intraclass_correlation",
    "estimate_clustered_ate",
    "estimate_tail_dependence",
    "event_probability",
    "event_query",
    "fh_position",
    "fh_width",
    "fit_copula_family",
    "fixed_sample_size_one_sided",
    "frechet_bounds",
    "generate_synthetic_clustered_two_world",
    "hoeffding_radius",
    "identified_region",
    "improved_frechet_bounds",
    "independence_regret",
    "independent_event_probability",
    "independent_joint_miss_baseline",
    "joint_probability_to_dependence",
    "mgl_basic_event_probabilities",
    "multiple_greek_letter",
    "naive_standard_error",
    "pairwise_correlation_bounds",
    "pairwise_joint_from_phi",
    "pairwise_rate_count",
    "partition_failure_probability",
    "random_feasible_distribution",
    "recommend_model",
    "run_coverage_simulation",
    "sample_binary_vectors",
    "sharp_frechet_bounds",
    "sample_copula",
    "sample_size_for_radius",
    "simulate_power_curve",
    "simultaneous_bernoulli_radius",
    "simultaneous_sample_size",
    "stress_test",
    "theoretical_tail_dependence",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"cc.kernel.{name}")
        globals()[name] = module
        return module
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__, *_SUBMODULES))
