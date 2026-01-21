"""
fh_bounds.py

Fréchet-Hoeffding bounds and compositional analysis utilities for multi-rail
detection systems.

This module is designed for research-grade analysis of composable detection
systems.  It provides:

* Mathematically correct Fréchet-Hoeffding bounds for intersections and unions
  of events with known marginals.
* Composition helpers for serial OR and parallel AND detector topologies.
* A defensively validated representation of probability bounds (FHBounds).
* A coherently constrained representation of composed Youden's J bounds
  (ComposedJBounds).
* Support for independence-based baselines and a Composability Interference
  Index (κ).
* Wilson-score-based uncertainty quantification for Youden's J.
* World-stratified bootstrap utilities over generic AttackResult-style
  evaluation objects.

The philosophy is "publication-ready by default":

* Every public class/function documents its semantics and assumptions.
* Invariants are explicitly checked and violations raise specific exceptions.
* Numerical comparisons use a configurable tolerance.
* Ambiguous data and silent defaults are avoided; when we must make an
  assumption, we either warn loudly or raise.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

try:  # SciPy is preferred for the normal quantile
    import scipy.stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - fallback will be used
    _scipy_stats = None  # type: ignore

# Optional WorldBit enum hook; this will be project-specific in real use.
try:  # pragma: no cover - this import is project-specific
    from your_codebase import WorldBit  # type: ignore
except Exception:  # pragma: no cover
    WorldBit = None  # type: ignore

# ---------------------------------------------------------------------------
# Module configuration
# ---------------------------------------------------------------------------

#: Tolerance for floating point comparisons in mathematical identities.
MATHEMATICAL_TOLERANCE: float = 1e-9


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class FHBoundsError(Exception):
    """Base exception for all errors raised by the fh_bounds module."""


class FHBoundViolationError(FHBoundsError):
    """
    Raised when an invariant on FHBounds or ComposedJBounds is violated.

    Examples:
        * lower > upper (beyond numerical tolerance)
        * probability outside [0, 1]
        * structural mismatches such as k_rails inconsistency
    """


class CompositionIncoherenceError(FHBoundsError):
    """
    Raised when J bounds are incoherent with the supplied TPR/FPR bounds.

    Semantically, ComposedJBounds must satisfy:

        j_lower ≈ tpr_bounds.lower - fpr_bounds.upper
        j_upper ≈ tpr_bounds.upper - fpr_bounds.lower

    If these equalities fail beyond MATHEMATICAL_TOLERANCE, this exception is
    raised.
    """


class NumericalInstabilityError(FHBoundsError):
    """
    Raised when numerically suspicious configurations are detected.

    In particular, if upper and lower bounds differ by less than
    MATHEMATICAL_TOLERANCE but are not exactly equal, this suggests that a
    mathematically sharp identity has been computed in an unstable way.
    """


class InvalidProbabilityError(FHBoundsError):
    """
    Raised when a probability-like quantity is invalid.

    This covers values outside [0, 1], NaN, Inf, or non-numeric objects that
    cannot be cast to float.
    """


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _approx_le(a: float, b: float, tol: float = MATHEMATICAL_TOLERANCE) -> bool:
    """Return True if a <= b within a numerical tolerance."""
    return a <= b + tol


def _approx_ge(a: float, b: float, tol: float = MATHEMATICAL_TOLERANCE) -> bool:
    """Return True if a ≥ b within a numerical tolerance."""
    return a + tol >= b


def _approx_eq(a: float, b: float, tol: float = MATHEMATICAL_TOLERANCE) -> bool:
    """Return True if a ≈ b within a numerical tolerance."""
    return abs(a - b) <= tol


def _clip(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return min(hi, max(lo, value))


def validate_probability_vector(probs: Sequence[float], name: str) -> None:
    """
    Validate that all values in ``probs`` are valid probabilities in [0, 1].

    Args:
        probs:
            Sequence of probability-like values.
        name:
            Human-readable name for this vector, used in error messages.

    Raises:
        InvalidProbabilityError:
            If any entry is non-numeric, NaN, Inf, or outside [0, 1].

    Notes:
        Values are explicitly cast to float so that exotic numeric types
        (e.g., numpy scalars, Decimal) are supported if they are convertible.
    """
    invalid_values: List[Tuple[int, Any, Any]] = []

    for i, p in enumerate(probs):
        try:
            fp = float(p)
        except (TypeError, ValueError) as e:  # pragma: no cover - defensive
            invalid_values.append((i, p, str(e)))
            continue

        if math.isnan(fp) or math.isinf(fp) or not (0.0 <= fp <= 1.0):
            invalid_values.append((i, p, fp))

    if invalid_values:
        details = "\n  ".join(f"Index {i}: {p!r} (as float: {fp})" for i, p, fp in invalid_values)
        raise InvalidProbabilityError(
            f"{name} contains {len(invalid_values)} invalid values:\n  {details}"
        )


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FHBounds:
    """
    Fréchet-Hoeffding bounds for a single composite event.

    Attributes:
        lower:
            Fréchet-Hoeffding lower bound on the event probability.
        upper:
            Hoeffding (union/Boole) upper bound on the event probability.
        marginals:
            Individual marginal probabilities of the constituent events.
        k_rails:
            Number of constituent events (should equal ``len(marginals)``).
        confidence_level:
            Optional nominal confidence level, if these bounds arise from a
            statistical procedure (e.g., bootstrap).  Defaults to 0.95.

    Invariants (enforced in ``__post_init__``):
        * ``len(marginals) == k_rails``: structural consistency.
        * ``0 <= lower <= upper <= 1`` (up to MATHEMATICAL_TOLERANCE).
        * Each marginal p_i satisfies 0 <= p_i <= 1 and is finite.
        * confidence_level is in (0, 1).
        * If ``|upper - lower| < MATHEMATICAL_TOLERANCE`` but ``upper != lower``
          (raw inequality), a NumericalInstabilityError is raised.
    """

    lower: float
    upper: float
    marginals: Sequence[float]
    k_rails: int
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        # Cast to float and basic finiteness checks
        lower = float(self.lower)
        upper = float(self.upper)

        if math.isnan(lower) or math.isinf(lower):
            raise FHBoundViolationError(f"lower bound is not finite: {lower}")
        if math.isnan(upper) or math.isinf(upper):
            raise FHBoundViolationError(f"upper bound is not finite: {upper}")

        # Confidence level sanity
        if not (0.0 < float(self.confidence_level) < 1.0):
            raise FHBoundViolationError(
                f"confidence_level={self.confidence_level} must lie in (0, 1)"
            )

        # Marginal probabilities validation
        validate_probability_vector(self.marginals, "marginals")

        if len(self.marginals) != self.k_rails:
            raise FHBoundViolationError(
                f"len(marginals)={len(self.marginals)} inconsistent with k_rails={self.k_rails}"
            )

        # Domain constraints for the composite probability
        if not _approx_ge(lower, 0.0):
            raise FHBoundViolationError(f"lower={lower} is negative beyond tolerance")
        if not _approx_le(upper, 1.0):
            raise FHBoundViolationError(f"upper={upper} exceeds 1 beyond tolerance")
        if not _approx_le(lower, upper):
            raise FHBoundViolationError(f"lower={lower} exceeds upper={upper} beyond tolerance")

        # Numerical instability check: suspiciously tight but not equal
        if abs(upper - lower) < MATHEMATICAL_TOLERANCE and upper != lower:
            raise NumericalInstabilityError(
                f"FHBounds interval suspiciously tight but not equal: "
                f"[{lower}, {upper}] with width={abs(upper - lower)}"
            )

        # Because dataclasses are frozen, we cannot assign back; but all checks
        # are performed on the casted values which originate from the fields.

    @property
    def width(self) -> float:
        """Return ``upper - lower``."""
        return float(self.upper) - float(self.lower)


@dataclass(frozen=True, slots=True)
class RailPerformance:
    """
    Observed or measured performance of a single detection rail.

    Attributes:
        tpr:
            True positive rate (sensitivity) in [0, 1].
        fpr:
            False positive rate in [0, 1].
        j:
            Youden's J statistic = tpr - fpr, in [-1, 1].

    Invariants:
        * tpr and fpr lie in [0, 1].
        * j lies in [-1, 1].
        * j is numerically consistent with tpr - fpr.
    """

    tpr: float
    fpr: float
    j: float

    def __post_init__(self) -> None:
        tpr = float(self.tpr)
        fpr = float(self.fpr)
        j = float(self.j)

        # Basic probability checks
        if not (0.0 <= tpr <= 1.0):
            raise InvalidProbabilityError(f"tpr={tpr} not in [0, 1]")
        if not (0.0 <= fpr <= 1.0):
            raise InvalidProbabilityError(f"fpr={fpr} not in [0, 1]")

        if not (-1.0 <= j <= 1.0):
            raise InvalidProbabilityError(f"j={j} not in [-1, 1]")

        computed_j = tpr - fpr
        if abs(j - computed_j) > MATHEMATICAL_TOLERANCE:
            raise ValueError(f"j={j} inconsistent with tpr - fpr={computed_j}")


@dataclass(frozen=True, slots=True)
class ComposedJBounds:
    """
    Bounds on Youden's J statistic for a composed detection system.

    Attributes:
        j_lower:
            Lower bound on J = TPR - FPR.
        j_upper:
            Upper bound on J.
        tpr_bounds:
            FHBounds describing the true positive rate of the composed system.
        fpr_bounds:
            FHBounds describing the false positive rate of the composed system.
        k_rails:
            Number of rails contributing to this composition.
        composition_type:
            Topology of the composition: "serial_or" or "parallel_and".
        individual_j_stats:
            Optional per-rail J statistics.  Either empty (length 0) or
            length k_rails.

    Invariants (enforced in ``__post_init__``):
        * -1 - tol <= j_lower <= j_upper <= 1 + tol.
        * tpr_bounds.k_rails == k_rails and fpr_bounds.k_rails == k_rails.
        * len(individual_j_stats) in {0, k_rails}.
        * composition_type in {"serial_or", "parallel_and"}.
        * j_lower/j_upper are coherent with tpr_bounds/fpr_bounds:
              j_lower ≈ tpr_bounds.lower - fpr_bounds.upper
              j_upper ≈ tpr_bounds.upper - fpr_bounds.lower
    """

    j_lower: float
    j_upper: float
    tpr_bounds: FHBounds
    fpr_bounds: FHBounds
    k_rails: int
    composition_type: str
    individual_j_stats: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        j_lower = float(self.j_lower)
        j_upper = float(self.j_upper)

        if math.isnan(j_lower) or math.isinf(j_lower):
            raise FHBoundViolationError(f"j_lower is not finite: {j_lower}")
        if math.isnan(j_upper) or math.isinf(j_upper):
            raise FHBoundViolationError(f"j_upper is not finite: {j_upper}")

        if j_lower < -1.0 - MATHEMATICAL_TOLERANCE:
            raise FHBoundViolationError(f"j_lower={j_lower} < -1 beyond tolerance")
        if j_upper > 1.0 + MATHEMATICAL_TOLERANCE:
            raise FHBoundViolationError(f"j_upper={j_upper} > 1 beyond tolerance")
        if not _approx_le(j_lower, j_upper):
            raise FHBoundViolationError(
                f"j_lower={j_lower} exceeds j_upper={j_upper} beyond tolerance"
            )

        # k_rails consistency
        if self.tpr_bounds.k_rails != self.k_rails:
            raise FHBoundViolationError(
                "tpr_bounds.k_rails "
                f"({self.tpr_bounds.k_rails}) inconsistent with k_rails={self.k_rails}"
            )
        if self.fpr_bounds.k_rails != self.k_rails:
            raise FHBoundViolationError(
                "fpr_bounds.k_rails "
                f"({self.fpr_bounds.k_rails}) inconsistent with k_rails={self.k_rails}"
            )

        # individual_j_stats length and range
        if self.individual_j_stats:
            if len(self.individual_j_stats) != self.k_rails:
                raise FHBoundViolationError(
                    "len(individual_j_stats) must be 0 or k_rails; "
                    f"got {len(self.individual_j_stats)} and k_rails={self.k_rails}"
                )
            for i, j in enumerate(self.individual_j_stats):
                jf = float(j)
                if math.isnan(jf) or math.isinf(jf) or not (-1.0 <= jf <= 1.0):
                    raise FHBoundViolationError(f"individual_j_stats[{i}]={jf} not in [-1, 1]")

        if self.composition_type not in ("serial_or", "parallel_and"):
            raise FHBoundViolationError(
                f"composition_type={self.composition_type!r} must be 'serial_or' or 'parallel_and'"
            )

        # Coherence with TPR/FPR bounds
        expected_j_lower = float(self.tpr_bounds.lower) - float(self.fpr_bounds.upper)
        expected_j_upper = float(self.tpr_bounds.upper) - float(self.fpr_bounds.lower)

        if abs(j_lower - expected_j_lower) > MATHEMATICAL_TOLERANCE:
            raise CompositionIncoherenceError(
                "j_lower inconsistent with tpr/fpr bounds:\n"
                f"  j_lower={j_lower}\n"
                f"  expected_j_lower={expected_j_lower}\n"
                f"  tpr_bounds.lower={self.tpr_bounds.lower}\n"
                f"  fpr_bounds.upper={self.fpr_bounds.upper}"
            )
        if abs(j_upper - expected_j_upper) > MATHEMATICAL_TOLERANCE:
            raise CompositionIncoherenceError(
                "j_upper inconsistent with tpr/fpr bounds:\n"
                f"  j_upper={j_upper}\n"
                f"  expected_j_upper={expected_j_upper}\n"
                f"  tpr_bounds.upper={self.tpr_bounds.upper}\n"
                f"  fpr_bounds.lower={self.fpr_bounds.lower}"
            )


# ---------------------------------------------------------------------------
# Fréchet-Hoeffding bounds for intersections and unions
# ---------------------------------------------------------------------------


def frechet_intersection_lower_bound(
    marginals: Sequence[float],
    confidence_level: float = 0.95,
) -> FHBounds:
    """
    Fréchet-Hoeffding bounds for the intersection of k events.

    Semantics:
        Given exact marginal probabilities p_i = P(A_i), the probability of the
        intersection A = n_i A_i satisfies:

            max(0, Σ p_i - (k - 1)) <= P(A) <= min_i p_i

    Args:
        marginals:
            Sequence of marginal probabilities p_i in [0, 1].
        confidence_level:
            Optional nominal confidence level to attach to the resulting
            FHBounds (purely annotational here).

    Returns:
        FHBounds instance representing bounds on P(n A_i).

    Raises:
        InvalidProbabilityError:
            If any marginal is invalid.
        FHBoundViolationError:
            If structural invariants fail (e.g., empty marginals).

    Notes:
        This function assumes *known* marginals.  When only intervals on the
        marginals are known, you should propagate those intervals explicitly
        rather than calling this helper.
    """
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]
    k = len(p)
    if k == 0:
        raise FHBoundViolationError("frechet_intersection_lower_bound: empty marginals")

    sum_p = sum(p)
    lower = max(0.0, sum_p - (k - 1))
    upper = min(p)

    return FHBounds(
        lower=lower,
        upper=upper,
        marginals=p,
        k_rails=k,
        confidence_level=confidence_level,
    )


def hoeffding_union_upper_bound(
    marginals: Sequence[float],
    confidence_level: float = 0.95,
) -> FHBounds:
    """
    Fréchet-Hoeffding bounds for the union of k events.

    Semantics:
        Given exact marginal probabilities p_i = P(A_i), the probability of the
        union U = U_i A_i satisfies:

            max_i p_i <= P(U) <= min(1, Σ p_i)

    Args:
        marginals:
            Sequence of marginal probabilities p_i in [0, 1].
        confidence_level:
            Optional nominal confidence level to attach to the resulting
            FHBounds (purely annotational here).

    Returns:
        FHBounds instance representing bounds on P(U A_i).

    Raises:
        InvalidProbabilityError:
            If any marginal is invalid.
        FHBoundViolationError:
            If structural invariants fail (e.g., empty marginals).
    """
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]
    k = len(p)
    if k == 0:
        raise FHBoundViolationError("hoeffding_union_upper_bound: empty marginals")

    sum_p = sum(p)
    lower = max(p)
    upper = min(1.0, sum_p)

    return FHBounds(
        lower=lower,
        upper=upper,
        marginals=p,
        k_rails=k,
        confidence_level=confidence_level,
    )


def _compose_intersection_from_bounds(
    bounds: Sequence[FHBounds],
    confidence_level: float = 0.95,
) -> FHBounds:
    """
    Intersection bounds when each event probability is itself interval-valued.

    Semantics:
        Suppose each event i has probability p_i ∈ [l_i, u_i].  Among all
        choices of p_i within these intervals and all joint distributions
        consistent with those probabilities, P(n A_i) is bounded by:

            max(0, Σ l_i - (k - 1)) <= P(n A_i) <= min_i u_i

        This follows by applying the Fréchet-Hoeffding intersection bounds
        with the "worst" admissible choice of marginals for each side.

    Args:
        bounds:
            Sequence of FHBounds objects, each representing an interval for a
            single event probability.
        confidence_level:
            Nominal confidence level for the resulting FHBounds.  This is
            typically inherited from the most conservative source.

    Returns:
        FHBounds representing bounds on P(n A_i).

    Raises:
        FHBoundViolationError:
            If the input sequence is empty.
    """
    if not bounds:
        raise FHBoundViolationError("_compose_intersection_from_bounds: empty bounds")

    k = len(bounds)
    lowers = [float(b.lower) for b in bounds]
    uppers = [float(b.upper) for b in bounds]

    sum_l = sum(lowers)
    lower = max(0.0, sum_l - (k - 1))
    upper = min(uppers)

    # Use midpoints as effective marginals; if intervals are sharp, this is exact.
    marginals = [0.5 * (lower + upper) for lower, upper in zip(lowers, uppers, strict=False)]

    return FHBounds(
        lower=lower,
        upper=upper,
        marginals=marginals,
        k_rails=k,
        confidence_level=confidence_level,
    )


def _compose_union_from_bounds(
    bounds: Sequence[FHBounds],
    confidence_level: float = 0.95,
) -> FHBounds:
    """
    Union bounds when each event probability is itself interval-valued.

    Semantics:
        Suppose each event i has probability p_i ∈ [l_i, u_i].  Among all
        choices of p_i within these intervals and all joint distributions
        consistent with those probabilities, P(U A_i) is bounded by:

            max_i l_i <= P(U A_i) <= min(1, Σ u_i)

        This uses the Fréchet-Hoeffding union bounds with the worst-case
        admissible choice of marginals for each side.

    Args:
        bounds:
            Sequence of FHBounds objects, each representing an interval for a
            single event probability.
        confidence_level:
            Nominal confidence level for the resulting FHBounds.

    Returns:
        FHBounds representing bounds on P(U A_i).
    """
    if not bounds:
        raise FHBoundViolationError("_compose_union_from_bounds: empty bounds")

    k = len(bounds)
    lowers = [float(b.lower) for b in bounds]
    uppers = [float(b.upper) for b in bounds]

    lower = max(lowers)
    upper = min(1.0, sum(uppers))

    marginals = [0.5 * (lower + upper) for lower, upper in zip(lowers, uppers, strict=False)]

    return FHBounds(
        lower=lower,
        upper=upper,
        marginals=marginals,
        k_rails=k,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# Composition logic for detection systems
# ---------------------------------------------------------------------------


def serial_or_composition(
    miss_rate_bounds: Sequence[FHBounds],
    false_alarm_bounds: Sequence[FHBounds],
) -> ComposedJBounds:
    """
    Compose miss and false-alarm bounds for a serial-OR detection pipeline.

    Semantics:
        * Miss event H = n F_i  (all detectors miss).
        * False-alarm event B = U A_i  (at least one detector fires falsely).
        * TPR = 1 - P(H).
        * FPR = P(B).

    Args:
        miss_rate_bounds:
            Sequence of FHBounds, one per rail, describing bounds on the
            per-rail miss probability P(F_i).  Typically each will have
            k_rails == 1, but this is not strictly required for the math.
        false_alarm_bounds:
            Sequence of FHBounds, one per rail, describing bounds on the
            per-rail false-alarm probability P(A_i).

    Returns:
        ComposedJBounds instance representing bounds on J for the composed
        serial-OR system.

    Raises:
        ValueError:
            If the lengths of the input sequences differ.
        FHBoundViolationError:
            If invariants are violated.
    """
    k = len(miss_rate_bounds)
    if k != len(false_alarm_bounds):
        raise ValueError(
            f"serial_or_composition: len(miss_rate_bounds)={k} "
            f"!= len(false_alarm_bounds)={len(false_alarm_bounds)}"
        )
    if k == 0:
        raise ValueError("serial_or_composition: empty rail list")

    # Compose miss rates via intersection: H = n F_i
    h_bounds = _compose_intersection_from_bounds(miss_rate_bounds)

    tpr_lower = max(0.0, 1.0 - float(h_bounds.upper))
    tpr_upper = min(1.0, 1.0 - float(h_bounds.lower))
    # Effective per-rail TPRs as complements of effective miss marginals.
    tpr_marginals = [1.0 - m for m in h_bounds.marginals]

    tpr_bounds = FHBounds(
        lower=tpr_lower,
        upper=tpr_upper,
        marginals=tpr_marginals,
        k_rails=k,
        confidence_level=h_bounds.confidence_level,
    )

    # Compose false alarms via union: B = U A_i
    b_bounds = _compose_union_from_bounds(false_alarm_bounds)

    fpr_lower = max(0.0, float(b_bounds.lower))
    fpr_upper = min(1.0, float(b_bounds.upper))
    fpr_marginals = list(b_bounds.marginals)

    fpr_bounds = FHBounds(
        lower=fpr_lower,
        upper=fpr_upper,
        marginals=fpr_marginals,
        k_rails=k,
        confidence_level=b_bounds.confidence_level,
    )

    # J bounds induced strictly from TPR/FPR bounds (coherence invariant)
    j_lower = tpr_bounds.lower - fpr_bounds.upper
    j_upper = tpr_bounds.upper - fpr_bounds.lower

    return ComposedJBounds(
        j_lower=j_lower,
        j_upper=j_upper,
        tpr_bounds=tpr_bounds,
        fpr_bounds=fpr_bounds,
        k_rails=k,
        composition_type="serial_or",
        individual_j_stats=(),  # can be populated by caller if desired
    )


def parallel_and_composition(
    miss_rate_bounds: Sequence[FHBounds],
    false_alarm_bounds: Sequence[FHBounds],
) -> ComposedJBounds:
    """
    Compose miss and false-alarm bounds for a parallel-AND detection system.

    Semantics:
        The system fires only if all rails fire.

        * Miss event H = U F_i  (at least one detector misses).
        * False-alarm event B = n A_i  (all detectors fire falsely).
        * TPR = 1 - P(H).
        * FPR = P(B).

    Args:
        miss_rate_bounds:
            Sequence of FHBounds, one per rail, describing bounds on the
            per-rail miss probability P(F_i).
        false_alarm_bounds:
            Sequence of FHBounds, one per rail, describing bounds on the
            per-rail false-alarm probability P(A_i).

    Returns:
        ComposedJBounds instance representing bounds on J for the composed
        parallel-AND system.
    """
    k = len(miss_rate_bounds)
    if k != len(false_alarm_bounds):
        raise ValueError(
            f"parallel_and_composition: len(miss_rate_bounds)={k} "
            f"!= len(false_alarm_bounds)={len(false_alarm_bounds)}"
        )
    if k == 0:
        raise ValueError("parallel_and_composition: empty rail list")

    # Miss event is a union: H = U F_i
    h_bounds = _compose_union_from_bounds(miss_rate_bounds)

    tpr_lower = max(0.0, 1.0 - float(h_bounds.upper))
    tpr_upper = min(1.0, 1.0 - float(h_bounds.lower))
    tpr_marginals = [1.0 - m for m in h_bounds.marginals]

    tpr_bounds = FHBounds(
        lower=tpr_lower,
        upper=tpr_upper,
        marginals=tpr_marginals,
        k_rails=k,
        confidence_level=h_bounds.confidence_level,
    )

    # False alarms require all rails to fire falsely: B = n A_i
    b_bounds = _compose_intersection_from_bounds(false_alarm_bounds)

    fpr_lower = max(0.0, float(b_bounds.lower))
    fpr_upper = min(1.0, float(b_bounds.upper))
    fpr_marginals = list(b_bounds.marginals)

    fpr_bounds = FHBounds(
        lower=fpr_lower,
        upper=fpr_upper,
        marginals=fpr_marginals,
        k_rails=k,
        confidence_level=b_bounds.confidence_level,
    )

    j_lower = tpr_bounds.lower - fpr_bounds.upper
    j_upper = tpr_bounds.upper - fpr_bounds.lower

    return ComposedJBounds(
        j_lower=j_lower,
        j_upper=j_upper,
        tpr_bounds=tpr_bounds,
        fpr_bounds=fpr_bounds,
        k_rails=k,
        composition_type="parallel_and",
        individual_j_stats=(),
    )


# ---------------------------------------------------------------------------
# Independence assumptions (fallback models)
# ---------------------------------------------------------------------------


def independence_serial_or_rates(
    individual_tprs: Sequence[float],
    individual_fprs: Sequence[float],
) -> Tuple[float, float]:
    """
    Compute serial-OR system rates under an independence assumption.

    Semantics:
        * Serial-OR system fires if any rail fires.
        * Under independence, with per-rail (TPR_i, FPR_i):

              TPR = 1 - ∏ (1 - TPR_i)
              FPR = 1 - ∏ (1 - FPR_i)

    Args:
        individual_tprs:
            Per-rail true positive rates in [0, 1].
        individual_fprs:
            Per-rail false positive rates in [0, 1].

    Returns:
        (TPR, FPR) for the serial-OR composition under independence.

    Raises:
        InvalidProbabilityError:
            If any rate is invalid.
        ValueError:
            If lengths of the sequences mismatch or are empty.
    """
    validate_probability_vector(individual_tprs, "individual_tprs")
    validate_probability_vector(individual_fprs, "individual_fprs")

    if len(individual_tprs) != len(individual_fprs):
        raise ValueError(
            "independence_serial_or_rates: len(individual_tprs) "
            f"{len(individual_tprs)} != len(individual_fprs) {len(individual_fprs)}"
        )
    if not individual_tprs:
        raise ValueError("independence_serial_or_rates: empty rate sequences")

    prod_one_minus_tpr = 1.0
    prod_one_minus_fpr = 1.0
    for tpr, fpr in zip(individual_tprs, individual_fprs, strict=False):
        prod_one_minus_tpr *= 1.0 - float(tpr)
        prod_one_minus_fpr *= 1.0 - float(fpr)

    tpr_sys = 1.0 - prod_one_minus_tpr
    fpr_sys = 1.0 - prod_one_minus_fpr

    # Clip into [0, 1] defensively
    tpr_sys = _clip(tpr_sys, 0.0, 1.0)
    fpr_sys = _clip(fpr_sys, 0.0, 1.0)

    return tpr_sys, fpr_sys


def independence_parallel_and_rates(
    individual_tprs: Sequence[float],
    individual_fprs: Sequence[float],
) -> Tuple[float, float]:
    """
    Compute parallel-AND system rates under an independence assumption.

    Semantics:
        * Parallel-AND system fires only if all rails fire.
        * Under independence:

              TPR = ∏ TPR_i
              FPR = ∏ FPR_i

    Args:
        individual_tprs:
            Per-rail true positive rates in [0, 1].
        individual_fprs:
            Per-rail false positive rates in [0, 1].

    Returns:
        (TPR, FPR) for the parallel-AND composition under independence.
    """
    validate_probability_vector(individual_tprs, "individual_tprs")
    validate_probability_vector(individual_fprs, "individual_fprs")

    if len(individual_tprs) != len(individual_fprs):
        raise ValueError(
            "independence_parallel_and_rates: len(individual_tprs) "
            f"{len(individual_tprs)} != len(individual_fprs) {len(individual_fprs)}"
        )
    if not individual_tprs:
        raise ValueError("independence_parallel_and_rates: empty rate sequences")

    tpr_sys = 1.0
    fpr_sys = 1.0
    for tpr, fpr in zip(individual_tprs, individual_fprs, strict=False):
        tpr_sys *= float(tpr)
        fpr_sys *= float(fpr)

    tpr_sys = _clip(tpr_sys, 0.0, 1.0)
    fpr_sys = _clip(fpr_sys, 0.0, 1.0)

    return tpr_sys, fpr_sys


def independence_serial_or_j(
    individual_tprs: Sequence[float],
    individual_fprs: Sequence[float],
) -> float:
    """
    Youden's J for a serial-OR composition under independence.

    Returns:
        J = TPR - FPR, computed from independence_serial_or_rates and clipped
        into [-1, 1] for numerical robustness.
    """
    tpr_sys, fpr_sys = independence_serial_or_rates(individual_tprs, individual_fprs)
    j = tpr_sys - fpr_sys
    return _clip(j, -1.0, 1.0)


def independence_parallel_and_j(
    individual_tprs: Sequence[float],
    individual_fprs: Sequence[float],
) -> float:
    """
    Youden's J for a parallel-AND composition under independence.

    Returns:
        J = TPR - FPR, computed from independence_parallel_and_rates and
        clipped into [-1, 1] for numerical robustness.
    """
    tpr_sys, fpr_sys = independence_parallel_and_rates(individual_tprs, individual_fprs)
    j = tpr_sys - fpr_sys
    return _clip(j, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Composability Interference Index (κ)
# ---------------------------------------------------------------------------


def compute_composability_interference_index(
    observed_j: float,
    bounds: ComposedJBounds,
    use_independence_baseline: bool = True,
    *,
    individual_tprs: Sequence[float] | None = None,
    individual_fprs: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """
    Compute the Composability Interference Index κ for a composed system.

    Semantics:
        κ measures how the observed J compares to a baseline J, normalized
        against the worst-case J within the Fréchet-Hoeffding envelope.

        * κ < 0: constructive interference (better than baseline).
        * κ ≈ 0: approximately independent behavior.
        * κ > 0: destructive interference (worse than baseline).
        * κ > 1: worse than FH worst-case bound (indicates model mismatch).

    Args:
        observed_j:
            Empirical J for the composed system, assumed to lie in [-1, 1].
        bounds:
            ComposedJBounds from composition (serial_or or parallel_and).
        use_independence_baseline:
            If True and per-rail TPR/FPR are provided, use the independence
            model as the baseline J.  Otherwise, fall back to the midpoint of
            the FH J-interval.
        individual_tprs:
            Optional per-rail true positive rates.  Only used if
            use_independence_baseline is True.
        individual_fprs:
            Optional per-rail false positive rates.  Only used if
            use_independence_baseline is True.

    Returns:
        Dictionary with keys:
            * 'kappa': κ, the interference index (may be NaN in degenerate
              normalization cases).
            * 'baseline_type': "independence" or "fh_midpoint".
            * 'j_baseline': Baseline J value.
            * 'j_worst': Worst-case J (bounds.j_lower).
            * 'j_best': Best-case J (bounds.j_upper).
            * 'baseline_within_bounds': Whether baseline J lies within the FH
              envelope (up to tolerance).
            * 'observed_within_bounds': Whether observed_j lies within FH
              envelope (up to tolerance).
            * 'notes': List of textual warnings or clarifications.

    Raises:
        ValueError:
            If observed_j is outside [-1, 1] or if per-rail rates have
            inconsistent lengths.
    """
    notes: List[str] = []

    observed_j = float(observed_j)
    if not (-1.0 <= observed_j <= 1.0):
        raise ValueError(f"observed_j={observed_j} not in [-1, 1]")

    j_worst = float(bounds.j_lower)
    j_best = float(bounds.j_upper)

    # Determine baseline
    if use_independence_baseline and individual_tprs is not None and individual_fprs is not None:
        validate_probability_vector(individual_tprs, "individual_tprs")
        validate_probability_vector(individual_fprs, "individual_fprs")

        if len(individual_tprs) != bounds.k_rails or len(individual_fprs) != bounds.k_rails:
            raise ValueError(
                "Length mismatch between per-rail rates and bounds.k_rails: "
                f"k_rails={bounds.k_rails}, "
                f"len(tprs)={len(individual_tprs)}, len(fprs)={len(individual_fprs)}"
            )

        if bounds.composition_type == "serial_or":
            j_baseline = independence_serial_or_j(individual_tprs, individual_fprs)
        elif bounds.composition_type == "parallel_and":
            j_baseline = independence_parallel_and_j(individual_tprs, individual_fprs)
        else:  # pragma: no cover - guarded in ComposedJBounds already
            raise ValueError(f"Unknown composition_type: {bounds.composition_type}")

        baseline_type = "independence"
    else:
        # FH midpoint baseline (data-independent, neutral prior)
        j_baseline = 0.5 * (j_worst + j_best)
        baseline_type = "fh_midpoint"
        if use_independence_baseline and (individual_tprs is None or individual_fprs is None):
            notes.append(
                "use_independence_baseline=True but individual rates not provided; "
                "using FH midpoint as neutral baseline instead."
            )

    # κ normalization
    denom = j_worst - j_baseline
    if abs(denom) < MATHEMATICAL_TOLERANCE:
        kappa = float("nan")
        notes.append(
            "Degenerate normalization: j_baseline ≈ j_worst. "
            "κ is undefined; consider using raw (observed_j - baseline) instead."
        )
    else:
        kappa = (observed_j - j_baseline) / denom

    # Within-bounds checks
    baseline_within_bounds = _approx_ge(j_baseline, j_worst) and _approx_le(j_baseline, j_best)
    observed_within_bounds = _approx_ge(observed_j, j_worst) and _approx_le(observed_j, j_best)

    if not observed_within_bounds:
        notes.append(
            f"observed_j={observed_j} lies outside FH bounds "
            f"[{j_worst}, {j_best}]. This suggests model mismatch or "
            "numerical error."
        )

    return {
        "kappa": kappa,
        "baseline_type": baseline_type,
        "j_baseline": j_baseline,
        "j_worst": j_worst,
        "j_best": j_best,
        "baseline_within_bounds": baseline_within_bounds,
        "observed_within_bounds": observed_within_bounds,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Statistical helpers: inverse normal and Wilson score intervals
# ---------------------------------------------------------------------------


def robust_inverse_normal(p: float, name: str = "p") -> float:
    """
    Inverse normal (quantile) function Q(p) = Φ⁻¹(p).

    Args:
        p:
            Tail probability in (0, 1).
        name:
            Name to use in error messages.

    Returns:
        The z-value such that P(Z <= z) = p for Z ~ N(0, 1).

    Raises:
        ValueError:
            If p is not in (0, 1) or if all available backends fail.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"{name}={p} not in (0, 1)")

    # Clip to avoid numerical issues at extreme tails
    p_clipped = max(1e-16, min(1.0 - 1e-16, p))

    # Prefer SciPy if available
    if _scipy_stats is not None:
        try:
            return float(_scipy_stats.norm.ppf(p_clipped))
        except Exception as e:  # pragma: no cover - extremely rare
            raise ValueError(f"Failed to compute Q({p}): {e}") from e

    # Fallback: use Python's statistics.NormalDist if SciPy is unavailable
    try:
        from statistics import NormalDist  # type: ignore

        return float(NormalDist().inv_cdf(p_clipped))
    except Exception as e:  # pragma: no cover - very unlikely
        raise ValueError(f"Failed to compute Q({p}) without SciPy: {e}") from e


def _wilson_interval(p: float, n: int, z: float) -> Tuple[float, float]:
    """
    Wilson score interval for a single proportion.

    Args:
        p:
            Observed proportion in [0, 1].
        n:
            Number of Bernoulli trials.
        z:
            z-score for the desired (two-sided) confidence level.

    Returns:
        (lower, upper) Wilson score interval.

    Notes:
        If n == 0, the interval is set to (0, 1), i.e., completely uninformative.
    """
    if n <= 0:
        return (0.0, 1.0)

    p = float(p)
    n = int(n)
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def wilson_score_ci(
    tpr: float,
    fpr: float,
    n_total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Wilson score confidence interval for Youden's J.

    Semantics:
        J = TPR - FPR.  We derive a conservative CI by taking Wilson intervals
        for TPR and FPR individually (using the same n_total for each) and then
        combining via worst/best-case subtraction:

            J_lower = TPR_lower - FPR_upper
            J_upper = TPR_upper - FPR_lower

    Args:
        tpr:
            Estimated true positive rate in [0, 1].
        fpr:
            Estimated false positive rate in [0, 1].
        n_total:
            Effective sample size used for both rates.  In fully stratified
            analyses you may want to use the smaller of {n_pos, n_neg}; here
            this is treated as a generic count.
        confidence_level:
            Confidence level (e.g., 0.95 for 95%).

    Returns:
        (J_lower, J_upper) in [-1, 1].

    Raises:
        InvalidProbabilityError:
            If tpr or fpr is invalid.
        ValueError:
            If n_total is non-positive or confidence_level is invalid.
    """
    validate_probability_vector([tpr, fpr], "rates")

    if n_total <= 0:
        raise ValueError("wilson_score_ci: n_total must be positive")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("wilson_score_ci: confidence_level must lie in (0, 1)")

    alpha = 1.0 - confidence_level
    z_alpha = robust_inverse_normal(1.0 - alpha / 2.0, name="1 - alpha/2")

    tpr_lower, tpr_upper = _wilson_interval(tpr, n_total, z_alpha)
    fpr_lower, fpr_upper = _wilson_interval(fpr, n_total, z_alpha)

    j_lower = tpr_lower - fpr_upper
    j_upper = tpr_upper - fpr_lower

    # Clip to the admissible range of J
    j_lower = _clip(j_lower, -1.0, 1.0)
    j_upper = _clip(j_upper, -1.0, 1.0)

    return j_lower, j_upper


# ---------------------------------------------------------------------------
# AttackResult integration and empirical J estimation
# ---------------------------------------------------------------------------


def _is_baseline_world_bit(wb: Any) -> bool:
    """Return True if wb denotes the baseline (benign) world."""
    if wb == 0:
        return True
    if WorldBit is not None:
        try:
            if wb == getattr(WorldBit, "BASELINE", None):
                return True
        except Exception:  # pragma: no cover - defensive
            pass
    return False


def _is_protected_world_bit(wb: Any) -> bool:
    """Return True if wb denotes the protected/adversarial world."""
    if wb == 1:
        return True
    if WorldBit is not None:
        try:
            if wb == getattr(WorldBit, "PROTECTED", None):
                return True
        except Exception:  # pragma: no cover - defensive
            pass
    return False


def _partition_attack_results_by_world(results: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Partition AttackResult-like objects by world_bit.

    Returns:
        (world_0_results, world_1_results)

    Notes:
        * If world_bit is missing, a warning is issued and the result is
          treated as world_bit=0 (benign baseline), matching the documented
          default but not silently.
        * If world_bit is unrecognized, the sample is ignored with a warning.
    """
    world_0_results: List[Any] = []
    world_1_results: List[Any] = []

    for idx, r in enumerate(results):
        wb = getattr(r, "world_bit", None)
        if wb is None:
            warnings.warn(
                f"AttackResult at index {idx} missing world_bit; "
                "defaulting to world_bit=0 (benign).",
                stacklevel=2,
            )
            wb = 0

        if _is_baseline_world_bit(wb):
            world_0_results.append(r)
        elif _is_protected_world_bit(wb):
            world_1_results.append(r)
        else:
            warnings.warn(
                f"AttackResult at index {idx} has unknown world_bit={wb!r}; "
                "ignoring this sample for empirical J computation.",
                stacklevel=2,
            )

    return world_0_results, world_1_results


def _extract_success_flag(r: Any, idx: int) -> bool:
    """
    Extract a boolean success flag from an AttackResult-like object.

    Raises:
        ValueError:
            If the 'success' attribute is missing or None.
    """
    if not hasattr(r, "success"):
        raise ValueError(
            f"AttackResult at index {idx} missing 'success' attribute; "
            "cannot infer detection outcome. Provide a compute_j_statistic "
            "callback or ensure 'success' is populated."
        )
    s = r.success
    if s is None:
        raise ValueError(
            f"AttackResult at index {idx} has success=None; cannot infer detection outcome."
        )
    return bool(s)


def empirical_j_from_attack_results(
    results: Sequence[Any],
    compute_j_statistic: Callable[[Sequence[Any]], float] | None = None,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute empirical Youden's J and a Wilson confidence interval from results.

    Semantic contract for the default behavior (compute_j_statistic is None):
        * Each result has an attribute ``world_bit``:
            - 0 (or WorldBit.BASELINE) => benign world.
            - 1 (or WorldBit.PROTECTED) => adversarial world.
          If missing, a warning is issued and world_bit=0 is assumed.
        * Each result has a boolean-like attribute ``success`` whose meaning
          depends on the world:
            - world_bit == 0: success=True => false positive (detector fired).
            - world_bit == 1: success=True => true positive (detector fired).

    If your semantics differ, pass a custom ``compute_j_statistic(results)``
    that directly returns J.

    Args:
        results:
            Sequence of AttackResult-like objects.
        compute_j_statistic:
            Optional callback to compute J directly.  If provided, the Wilson
            interval is degenerate at J (no additional uncertainty modeling is
            attempted).
        confidence_level:
            Confidence level for the Wilson interval when using default
            semantics.

    Returns:
        (observed_j, ci_lower, ci_upper)

    Raises:
        ValueError:
            If inputs are invalid, world stratification fails, or the callback
            returns an out-of-range J.
    """
    if not results:
        raise ValueError("empirical_j_from_attack_results: empty results")

    if compute_j_statistic is not None:
        if not callable(compute_j_statistic):
            raise ValueError("compute_j_statistic must be callable when provided")
        try:
            observed_j = float(compute_j_statistic(results))
        except Exception as e:
            raise ValueError(f"compute_j_statistic raised {type(e).__name__}: {e}") from e
        if not (-1.0 <= observed_j <= 1.0):
            raise ValueError(
                f"compute_j_statistic returned invalid J={observed_j}; expected value in [-1, 1]"
            )
        # Without world semantics we cannot construct a meaningful Wilson CI.
        warnings.warn(
            "compute_j_statistic provided; returning a degenerate CI at "
            "observed_j (no world-stratified uncertainty modeling).",
            stacklevel=2,
        )
        return observed_j, observed_j, observed_j

    # Default: world-stratified semantics
    world_0_results, world_1_results = _partition_attack_results_by_world(results)

    if not world_0_results or not world_1_results:
        raise ValueError(
            "empirical_j_from_attack_results requires samples from both "
            "world_bit=0 and world_bit=1. "
            f"Got n0={len(world_0_results)}, n1={len(world_1_results)}."
        )

    # Count successes in each world
    fp_count = 0
    for idx, r in enumerate(world_0_results):
        if _extract_success_flag(r, idx):
            fp_count += 1

    tp_count = 0
    for idx, r in enumerate(world_1_results):
        if _extract_success_flag(r, idx):
            tp_count += 1

    n0 = len(world_0_results)
    n1 = len(world_1_results)

    fpr = fp_count / n0
    tpr = tp_count / n1
    observed_j = tpr - fpr

    ci_lower, ci_upper = wilson_score_ci(
        tpr=tpr,
        fpr=fpr,
        n_total=len(results),
        confidence_level=confidence_level,
    )

    return observed_j, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Bootstrap with world stratification
# ---------------------------------------------------------------------------


def bootstrap_composability_bounds(
    results: Sequence[Any],
    n_boots: int = 10000,
    composition_type: str = "serial_or",
    compute_j_statistic: Callable[[Sequence[Any]], float] | None = None,
    random_seed: int | None = None,
    confidence_level: float = 0.95,
) -> ComposedJBounds:
    """
    Empirical bootstrap of J bounds from AttackResult-style evaluation data.

    The bootstrap is world-stratified: samples from the benign and adversarial
    worlds are resampled separately and then recombined, preserving the class
    balance of the original dataset.

    Args:
        results:
            Sequence of AttackResult-like objects with ``world_bit`` and
            ``success`` attributes, as in empirical_j_from_attack_results.
        n_boots:
            Number of bootstrap resamples.
        composition_type:
            Tag for the resulting ComposedJBounds; for a black-box system this
            is mainly used for baseline interpretation (serial_or vs
            parallel_and).
        compute_j_statistic:
            Currently not supported in the bootstrap path.  If provided, a
            NotImplementedError is raised to avoid silently ignoring semantics.
        random_seed:
            Optional random seed to make the bootstrap reproducible.
        confidence_level:
            Nominal level for the resulting FHBounds.

    Returns:
        ComposedJBounds representing bootstrap-based bounds on J, with
        associated FHBounds for TPR and FPR.

    Raises:
        ValueError:
            If inputs are invalid or stratification fails.
        NotImplementedError:
            If compute_j_statistic is supplied (semantic mismatch).
    """
    if compute_j_statistic is not None:
        raise NotImplementedError(
            "bootstrap_composability_bounds does not yet support "
            "custom compute_j_statistic callbacks; pass results to "
            "empirical_j_from_attack_results instead."
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    world_0_results, world_1_results = _partition_attack_results_by_world(results)

    if not world_0_results or not world_1_results:
        raise ValueError(
            "bootstrap_composability_bounds requires samples from both "
            f"world_bit=0 and world_bit=1. Got n0={len(world_0_results)}, "
            f"n1={len(world_1_results)}."
        )

    n0 = len(world_0_results)
    n1 = len(world_1_results)

    tpr_samples: List[float] = []
    fpr_samples: List[float] = []

    world_0_results = list(world_0_results)
    world_1_results = list(world_1_results)

    for _ in range(n_boots):
        # Resample within each world with replacement
        boot_0_idx = np.random.randint(0, n0, size=n0)
        boot_1_idx = np.random.randint(0, n1, size=n1)

        fp_count = 0
        for _j, idx0 in enumerate(boot_0_idx):
            if _extract_success_flag(world_0_results[idx0], idx0):
                fp_count += 1

        tp_count = 0
        for _j, idx1 in enumerate(boot_1_idx):
            if _extract_success_flag(world_1_results[idx1], idx1):
                tp_count += 1

        fpr_boot = fp_count / n0
        tpr_boot = tp_count / n1

        fpr_samples.append(fpr_boot)
        tpr_samples.append(tpr_boot)

    tpr_samples_arr = np.asarray(tpr_samples, dtype=float)
    fpr_samples_arr = np.asarray(fpr_samples, dtype=float)

    lower_q = (1.0 - confidence_level) / 2.0
    upper_q = 1.0 - lower_q

    tpr_lower = float(np.quantile(tpr_samples_arr, lower_q))
    tpr_upper = float(np.quantile(tpr_samples_arr, upper_q))
    fpr_lower = float(np.quantile(fpr_samples_arr, lower_q))
    fpr_upper = float(np.quantile(fpr_samples_arr, upper_q))

    tpr_mean = float(tpr_samples_arr.mean())
    fpr_mean = float(fpr_samples_arr.mean())

    tpr_bounds = FHBounds(
        lower=_clip(tpr_lower, 0.0, 1.0),
        upper=_clip(tpr_upper, 0.0, 1.0),
        marginals=[_clip(tpr_mean, 0.0, 1.0)],
        k_rails=1,
        confidence_level=confidence_level,
    )
    fpr_bounds = FHBounds(
        lower=_clip(fpr_lower, 0.0, 1.0),
        upper=_clip(fpr_upper, 0.0, 1.0),
        marginals=[_clip(fpr_mean, 0.0, 1.0)],
        k_rails=1,
        confidence_level=confidence_level,
    )

    j_lower = tpr_bounds.lower - fpr_bounds.upper
    j_upper = tpr_bounds.upper - fpr_bounds.lower

    return ComposedJBounds(
        j_lower=j_lower,
        j_upper=j_upper,
        tpr_bounds=tpr_bounds,
        fpr_bounds=fpr_bounds,
        k_rails=1,
        composition_type=composition_type,
        individual_j_stats=(),
    )


# ---------------------------------------------------------------------------
# Self-tests for internal consistency
# ---------------------------------------------------------------------------


def verify_fh_bound_properties(verbose: bool = False) -> None:
    """
    Exhaustive internal self-tests for the fh_bounds module.

    Raises:
        AssertionError:
            If any check fails.

    Notes:
        This is intended as a quick coherence check for development and
        research use.  It is not a substitute for a full unit test suite, but
        it should catch gross inconsistencies in the mathematical structure.
    """
    TOL = MATHEMATICAL_TOLERANCE

    # Test 1: Single-event exactness in FHBounds
    p = 0.3
    bounds = FHBounds(lower=p, upper=p, marginals=[p, p, p], k_rails=3)
    assert bounds.lower == bounds.upper == p

    # Test 2: Intersection monotonicity
    m1 = [0.5, 0.6]
    m2 = [0.5, 0.6, 0.2]
    b1 = frechet_intersection_lower_bound(m1)
    b2 = frechet_intersection_lower_bound(m2)
    assert b2.lower <= b1.lower + TOL, f"Intersection not monotone: {b2.lower} > {b1.lower}"

    # Test 3: Union vs Intersection ordering
    int_b = frechet_intersection_lower_bound([0.4, 0.6])
    uni_b = hoeffding_union_upper_bound([0.4, 0.6])
    assert int_b.upper <= min(0.4, 0.6) + TOL, "Intersection upper > min(marginals)"
    assert max(0.4, 0.6) <= uni_b.lower + TOL, "Union lower < max(marginals)"

    # Test 4: J consistency in ComposedJBounds
    tpr_b = FHBounds(0.6, 0.8, [0.7], k_rails=1)
    fpr_b = FHBounds(0.1, 0.3, [0.2], k_rails=1)
    cj = ComposedJBounds(
        j_lower=0.6 - 0.3,
        j_upper=0.8 - 0.1,
        tpr_bounds=tpr_b,
        fpr_bounds=fpr_b,
        k_rails=1,
        composition_type="serial_or",
    )

    # Test 5: Independence assumptions
    individual_tprs = [0.9, 0.85]
    individual_fprs = [0.05, 0.1]
    j_indep_or = independence_serial_or_j(individual_tprs, individual_fprs)
    expected_j = (1.0 - (1.0 - 0.9) * (1.0 - 0.85)) - (1.0 - (1.0 - 0.05) * (1.0 - 0.1))
    assert abs(j_indep_or - expected_j) < TOL

    # Test 6: CII sanity
    observed_j = 0.5
    cii_result = compute_composability_interference_index(
        observed_j,
        cj,
        use_independence_baseline=False,
        individual_tprs=individual_tprs,
        individual_fprs=individual_fprs,
    )
    kappa = cii_result["kappa"]
    assert (math.isnan(kappa)) or (-2.0 <= kappa <= 2.0)
    assert cii_result["baseline_type"] in ("independence", "fh_midpoint")

    # Test 7: Wilson CI properties
    ci_lower, ci_upper = wilson_score_ci(0.8, 0.2, n_total=100)
    j_hat = 0.8 - 0.2
    assert 0.0 <= ci_lower <= j_hat <= ci_upper <= 1.0 + TOL
    assert ci_lower <= j_hat <= ci_upper

    if verbose:
        print("✓ All FH-bounds self-tests passed")
