"""Classical common-cause-failure point models for redundant guardrails.

The functions in this module implement the beta-factor, alpha-factor, and
multiple-Greek-letter (MGL) parameterizations used in reliability fault trees.
They are intentionally assumption-laden complements to the Frechet-Hoeffding
kernel: each model converts a homogeneous common-cause component group into
basic event probabilities and then returns a single all-rails-fail point
estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb, isclose
from typing import Literal, TypeAlias, cast

import numpy as np

from cc.kernel.frechet_classes import FrechetBoundResult, frechet_bounds

AlphaTestingScheme: TypeAlias = Literal["staggered", "non_staggered", "non-staggered"]
EvidenceLevel: TypeAlias = Literal[
    "none",
    "pairwise_correlation_only",
    "full_historical_cofailure_counts",
]
RecommendationKind: TypeAlias = Literal[
    "FH-only",
    "FH+CCF-point-estimate",
    "full empirical estimation",
]

_TOL = 1.0e-10

__all__ = [
    "AlphaTestingScheme",
    "DependenceEvidence",
    "EvidenceLevel",
    "ModelRecommendation",
    "RecommendationKind",
    "alpha_factor",
    "alpha_factor_basic_event_probabilities",
    "assert_within_fh_envelope",
    "beta_factor",
    "beta_factor_basic_event_probabilities",
    "mgl_basic_event_probabilities",
    "multiple_greek_letter",
    "partition_failure_probability",
    "recommend_model",
]


@dataclass(frozen=True)
class DependenceEvidence:
    """Summary of dependence evidence available for model selection."""

    level: EvidenceLevel | None = None
    has_pairwise_correlation: bool = False
    has_historical_cofailure_counts: bool = False


@dataclass(frozen=True)
class ModelRecommendation:
    """Recommendation returned by :func:`recommend_model`."""

    evidence_level: EvidenceLevel
    recommendation: RecommendationKind
    justification: str
    required_assumptions: tuple[str, ...]


def beta_factor(
    failure_rates: Sequence[float],
    beta: float,
    *,
    homogeneity_tol: float = _TOL,
) -> float:
    """Return the beta-factor all-rails-fail point estimate.

    ``failure_rates`` are per-rail failure probabilities for one common demand
    or mission interval.  The classical beta-factor parameterization assumes a
    homogeneous common-cause component group, so all entries must agree within
    ``homogeneity_tol``.
    """

    q_values = beta_factor_basic_event_probabilities(
        failure_rates,
        beta,
        homogeneity_tol=homogeneity_tol,
    )
    return partition_failure_probability(q_values)


def beta_factor_basic_event_probabilities(
    failure_rates: Sequence[float],
    beta: float,
    *,
    homogeneity_tol: float = _TOL,
) -> tuple[float, ...]:
    """Return beta-factor ``Q_k`` values for a group of size ``m``.

    ``Q_k`` is the probability of a basic event involving a specific subset of
    exactly ``k`` components.  For beta-factor, ``Q_1 = (1 - beta) Q_t``,
    ``Q_m = beta Q_t``, and intermediate ``Q_k`` values are zero.
    """

    q_t, m = _homogeneous_rate(failure_rates, homogeneity_tol)
    beta_value = _probability(beta, "beta")
    q_values = [0.0] * m
    q_values[0] = (1.0 - beta_value) * q_t
    q_values[-1] = beta_value * q_t
    return tuple(q_values)


def alpha_factor(
    failure_rates: Sequence[float],
    alpha_factors: Sequence[float],
    *,
    testing_scheme: AlphaTestingScheme = "staggered",
    homogeneity_tol: float = _TOL,
) -> float:
    """Return the alpha-factor all-rails-fail point estimate."""

    q_values = alpha_factor_basic_event_probabilities(
        failure_rates,
        alpha_factors,
        testing_scheme=testing_scheme,
        homogeneity_tol=homogeneity_tol,
    )
    return partition_failure_probability(q_values)


def alpha_factor_basic_event_probabilities(
    failure_rates: Sequence[float],
    alpha_factors: Sequence[float],
    *,
    testing_scheme: AlphaTestingScheme = "staggered",
    homogeneity_tol: float = _TOL,
) -> tuple[float, ...]:
    """Return alpha-factor ``Q_k`` values for a group of size ``m``.

    Staggered testing uses ``Q_k = alpha_k Q_t / C(m - 1, k - 1)``.  The
    non-staggered form uses
    ``Q_k = k alpha_k Q_t / (C(m - 1, k - 1) alpha_t)``, where
    ``alpha_t = sum_k k alpha_k``.
    """

    q_t, m = _homogeneous_rate(failure_rates, homogeneity_tol)
    alphas = _probability_vector(alpha_factors, "alpha_factors", expected_size=m)
    total_alpha = float(np.sum(alphas))
    if not isclose(total_alpha, 1.0, abs_tol=1.0e-8):
        raise ValueError(f"alpha_factors must sum to 1. Got {total_alpha}.")

    scheme = _canonical_testing_scheme(testing_scheme)
    if scheme == "staggered":
        return tuple(float(alphas[k - 1]) * q_t / comb(m - 1, k - 1) for k in range(1, m + 1))

    alpha_t = float(sum(k * float(alphas[k - 1]) for k in range(1, m + 1)))
    if alpha_t <= _TOL:
        raise ValueError("alpha_t must be positive for non-staggered alpha-factor modeling.")
    return tuple(
        k * float(alphas[k - 1]) * q_t / (comb(m - 1, k - 1) * alpha_t) for k in range(1, m + 1)
    )


def multiple_greek_letter(
    failure_rates: Sequence[float],
    greek_parameters: Sequence[float],
    *,
    homogeneity_tol: float = _TOL,
) -> float:
    """Return the multiple-Greek-letter all-rails-fail point estimate.

    ``greek_parameters`` must contain ``m - 1`` conditional probabilities.  For
    a three-rail group these are beta and gamma.
    """

    q_values = mgl_basic_event_probabilities(
        failure_rates,
        greek_parameters,
        homogeneity_tol=homogeneity_tol,
    )
    return partition_failure_probability(q_values)


def mgl_basic_event_probabilities(
    failure_rates: Sequence[float],
    greek_parameters: Sequence[float],
    *,
    homogeneity_tol: float = _TOL,
) -> tuple[float, ...]:
    """Return MGL ``Q_k`` values for a group of size ``m``.

    With ``rho_1 = 1``, ``rho_2 = beta``, ``rho_3 = gamma``, ...,
    ``rho_{m+1} = 0``, the classical MGL formula is
    ``Q_k = Q_t prod_{i=1}^k rho_i (1 - rho_{k+1}) / C(m - 1, k - 1)``.
    """

    q_t, m = _homogeneous_rate(failure_rates, homogeneity_tol)
    greek = _probability_vector(greek_parameters, "greek_parameters", expected_size=m - 1)
    rho = [1.0, *[float(value) for value in greek], 0.0]
    q_values: list[float] = []
    running_product = 1.0
    for k in range(1, m + 1):
        running_product *= rho[k - 1]
        q_values.append(running_product * (1.0 - rho[k]) * q_t / comb(m - 1, k - 1))
    return tuple(q_values)


def partition_failure_probability(basic_event_probabilities: Sequence[float]) -> float:
    """Evaluate the standard all-fail set-partition polynomial.

    If ``Q_k`` is the probability of a basic event involving a specific subset
    of ``k`` rails, the one-out-of-``m`` success criterion fails when the rails
    are covered by a partition of independent/common-cause basic events.  The
    recurrence chooses the block containing a distinguished component:

    ``B_0 = 1`` and
    ``B_n = sum_{k=1}^n C(n - 1, k - 1) Q_k B_{n-k}``.
    """

    q_values = _probability_vector(
        basic_event_probabilities,
        "basic_event_probabilities",
        min_size=2,
    )
    m = q_values.size
    bell = [0.0] * (m + 1)
    bell[0] = 1.0
    for n in range(1, m + 1):
        bell[n] = sum(
            comb(n - 1, k - 1) * float(q_values[k - 1]) * bell[n - k] for k in range(1, n + 1)
        )
    return _clip_probability(float(bell[m]), "partition failure probability")


def assert_within_fh_envelope(
    point_estimate: float,
    failure_rates: Sequence[float],
    *,
    tol: float = _TOL,
) -> FrechetBoundResult:
    """Assert that a CCF point estimate lies inside the FH all-fail envelope.

    An estimate outside the envelope cannot be the probability of ``all rails
    fail`` for any Bernoulli joint law with the supplied marginals.  The helper
    returns the computed FH result on success so tests can also inspect the
    envelope.
    """

    estimate = _probability(point_estimate, "point_estimate", tol=tol)
    bounds = frechet_bounds(failure_rates, event="and")
    if estimate < bounds.lower - tol or estimate > bounds.upper + tol:
        raise AssertionError(
            "CCF point estimate is outside the Frechet-Hoeffding envelope: "
            f"estimate={estimate}, envelope=[{bounds.lower}, {bounds.upper}], "
            f"failure_rates={tuple(float(x) for x in failure_rates)}."
        )
    return bounds


def recommend_model(
    evidence: DependenceEvidence | Mapping[str, object] | str,
) -> ModelRecommendation:
    """Recommend an estimation strategy from the available dependence evidence."""

    level = _evidence_level(evidence)
    if level == "none":
        return ModelRecommendation(
            evidence_level=level,
            recommendation="FH-only",
            justification=(
                "No dependence evidence is available, so any CCF parameter would be "
                "unidentified. Report the assumption-free FH envelope."
            ),
            required_assumptions=(
                "Correct guardrail-level marginal failure probabilities.",
                "No structural dependence model asserted.",
            ),
        )
    if level == "pairwise_correlation_only":
        return ModelRecommendation(
            evidence_level=level,
            recommendation="FH+CCF-point-estimate",
            justification=(
                "Pairwise dependence can tighten FH bounds, but it does not identify "
                "higher-order common-cause structure. Use FH as the assumption-bound "
                "reference interval and a "
                "CCF point estimate only as an explicit sensitivity assumption."
            ),
            required_assumptions=(
                "Pairwise side information is feasible for the supplied marginals.",
                "Chosen beta-factor, alpha-factor, or MGL parameters describe a "
                "homogeneous common-cause component group.",
                "The CCF point estimate passes the FH consistency check.",
            ),
        )
    return ModelRecommendation(
        evidence_level=level,
        recommendation="full empirical estimation",
        justification=(
            "Historical co-failure counts can estimate the joint failure law or "
            "basic-event probabilities directly. Use parametric CCF models as "
            "regularizers or diagnostics, not as the primary source of dependence."
        ),
        required_assumptions=(
            "Counts are representative of the deployment environment.",
            "Demand or mission-interval exposure is measured consistently.",
            "Sparse cells receive explicit uncertainty treatment.",
        ),
    )


def _homogeneous_rate(failure_rates: Sequence[float], tol: float) -> tuple[float, int]:
    rates = _probability_vector(failure_rates, "failure_rates", min_size=2)
    first = float(rates[0])
    if np.any(np.abs(rates - first) > tol):
        raise ValueError(
            "Classical CCF parameterizations require a homogeneous common-cause "
            "component group; all failure_rates must agree within homogeneity_tol."
        )
    return first, int(rates.size)


def _probability_vector(
    values: Sequence[float],
    name: str,
    *,
    expected_size: int | None = None,
    min_size: int | None = None,
    tol: float = _TOL,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values. Got {arr.size}.")
    if min_size is not None and arr.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(arr < -tol) or np.any(arr > 1.0 + tol):
        raise ValueError(f"{name} entries must lie in [0, 1].")
    return cast(np.ndarray, np.clip(arr, 0.0, 1.0).astype(float, copy=False))


def _probability(value: float, name: str, *, tol: float = _TOL) -> float:
    if not np.isfinite(value) or value < -tol or value > 1.0 + tol:
        raise ValueError(f"{name} must be a finite probability in [0, 1]. Got {value}.")
    return float(np.clip(value, 0.0, 1.0))


def _clip_probability(value: float, name: str, *, tol: float = _TOL) -> float:
    if not np.isfinite(value) or value < -tol or value > 1.0 + tol:
        raise ValueError(f"{name} is not a valid probability: {value}.")
    return float(np.clip(value, 0.0, 1.0))


def _canonical_testing_scheme(
    testing_scheme: AlphaTestingScheme,
) -> Literal["staggered", "non_staggered"]:
    scheme = str(testing_scheme).lower().replace("-", "_")
    if scheme == "staggered":
        return "staggered"
    if scheme == "non_staggered":
        return "non_staggered"
    raise ValueError('testing_scheme must be "staggered" or "non_staggered".')


def _evidence_level(evidence: DependenceEvidence | Mapping[str, object] | str) -> EvidenceLevel:
    if isinstance(evidence, str):
        normalized = evidence.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "none": "none",
            "no_dependence": "none",
            "pairwise": "pairwise_correlation_only",
            "pairwise_correlation": "pairwise_correlation_only",
            "pairwise_correlation_only": "pairwise_correlation_only",
            "full": "full_historical_cofailure_counts",
            "full_counts": "full_historical_cofailure_counts",
            "cofailure_counts": "full_historical_cofailure_counts",
            "full_historical_cofailure_counts": "full_historical_cofailure_counts",
        }
        try:
            return cast(EvidenceLevel, aliases[normalized])
        except KeyError as exc:
            raise ValueError(f"Unknown dependence evidence level: {evidence!r}.") from exc

    if isinstance(evidence, DependenceEvidence):
        if evidence.level is not None:
            return _evidence_level(evidence.level)
        if evidence.has_historical_cofailure_counts:
            return "full_historical_cofailure_counts"
        if evidence.has_pairwise_correlation:
            return "pairwise_correlation_only"
        return "none"

    level = evidence.get("level")
    if isinstance(level, str):
        return _evidence_level(level)

    full_keys = (
        "full_historical_cofailure_counts",
        "historical_cofailure_counts",
        "cofailure_counts",
        "has_historical_cofailure_counts",
        "has_cofailure_counts",
    )
    pairwise_keys = (
        "pairwise_correlation_only",
        "pairwise_correlation",
        "has_pairwise_correlation",
        "correlation",
    )
    if any(bool(evidence.get(key)) for key in full_keys):
        return "full_historical_cofailure_counts"
    if any(bool(evidence.get(key)) for key in pairwise_keys):
        return "pairwise_correlation_only"
    return "none"
