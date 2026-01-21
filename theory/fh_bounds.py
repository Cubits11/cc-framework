# src/cc/theory/fh_bounds.py
"""
Fréchet-Hoeffding Bounds for Guardrail Composition (Research-Grade Implementation)

This module provides a mathematically rigorous implementation of
Fréchet-Hoeffding (FH) and related bounds for multi-guardrail safety
composition, together with independence baselines and diagnostic
indices for composability.

Design goals
------------
1. **Mathematical correctness**
   - Proper FH bounds for intersections and unions.
   - Correct mapping from miss/false-alarm rates to TPR/FPR and J-statistics.
   - Correct independence formulas for OR/AND gating.

2. **Explicit modeling of uncertainty**
   - Bounds are represented as immutable `FHBounds` objects with
     clear invariants: 0 <= lower <= upper <= 1.
   - `ComposedJBounds` captures how TPR/FPR uncertainty propagates
     into J-statistic uncertainty.

3. **No hidden assumptions**
   - We do NOT try to reconstruct TPR/FPR from J alone.
   - Independence baselines for composability analysis require true
     per-rail TPR/FPR, or gracefully fall back to a documented
     heuristic if those are unavailable.

4. **Robust numerics**
   - Probability-vector validation handles NaNs, infinities, and
     non-numeric types with informative errors.
   - Wilson score intervals are used for binomial CIs (instead of
     misusing DKW).
   - Inverse normal CDF is implemented with multiple fallbacks and
     defensive checks.

This module is intended to be the theoretical backbone for the CC
framework's guardrail-composition analyses, including:
- serial OR composition,
- parallel AND composition,
- independence baselines,
- composability interference index (CII),
- bootstrap-based J-statistic uncertainty,
- sensitivity analysis to rate perturbations,
- and internal mathematical self-tests.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# Optional imports from the core framework.
# The module is designed to *not* hard-depend on these at import time.
try:  # pragma: no cover - exercised in integration, not unit core
    from ..core.models import AttackResult, WorldBit  # type: ignore
    from ..core.stats import compute_j_statistic  # type: ignore
except Exception:  # pragma: no cover - standalone mode
    AttackResult = Any  # type: ignore
    WorldBit = Any  # type: ignore

    def compute_j_statistic(results: Sequence[Any]) -> Tuple[float, float, float]:
        """
        Minimal stub for standalone usage.

        Real implementation in cc.core.stats should be used in actual experiments.
        """
        # J, tpr, fpr (dummy values)
        return 0.0, 0.5, 0.5


# ---------------------------------------------------------------------
# Constants & type aliases
# ---------------------------------------------------------------------

EPSILON: float = 1e-12
MATHEMATICAL_TOLERANCE: float = 1e-10
MAX_BOOTSTRAP_ITERATIONS: int = 10_000
MIN_SAMPLE_SIZE_WARNING: int = 30
DEFAULT_CC_RELATIVE_MARGIN: float = 0.05
DEFAULT_CC_ABSOLUTE_MARGIN: float = 0.0
DEFAULT_CC_UNCERTAINTY_MULTIPLIER: float = 0.5

Probability = float  # in [0, 1]
JStatistic = float  # in [-1, 1]
Bounds = Tuple[float, float]
CopulaFamily = Literal[
    "independence",
    "comonotonic",
    "countermonotonic",
    "clayton",
    "gumbel",
    "frank",
]


@dataclass(frozen=True, slots=True)
class CCRegimeThresholds:
    """
    Threshold policy for classifying compositional regimes.

    The thresholds define a symmetric independence band around CC = 1:
        constructive: CC < (1 - margin)
        destructive: CC > (1 + margin)
        independent: CC in [1 - margin, 1 + margin]

    The default margin (5%) is a *policy placeholder* intended to absorb
    sampling noise, model misspecification, and audit tolerances. For
    enterprise deployments, this should be made policy-driven based on
    measurement uncertainty and contractual requirements.
    """

    constructive: float
    destructive: float
    margin: float
    rationale: str


def default_cc_regime_thresholds(
    relative_margin: float = DEFAULT_CC_RELATIVE_MARGIN,
    absolute_margin: float = DEFAULT_CC_ABSOLUTE_MARGIN,
    minimum_margin: float = 0.0,
    rationale: str | None = None,
) -> CCRegimeThresholds:
    """
    Construct default CC regime thresholds centered at 1.0.

    Parameters
    ----------
    relative_margin:
        Fractional slack around 1.0 used to define independence.
        The default is 5% (0.95/1.05), which provides a conservative
        buffer for empirical uncertainty.
    absolute_margin:
        Absolute minimum slack (useful when CC is near 1 and relative
        margins are too tight).
    minimum_margin:
        Hard lower bound for the margin.
    rationale:
        Optional override for the human-readable rationale.
    """
    if relative_margin < 0 or absolute_margin < 0 or minimum_margin < 0:
        raise ValueError("Margins must be non-negative.")

    margin = max(relative_margin, absolute_margin, minimum_margin)
    if rationale is None:
        rationale = (
            "Default independence band uses a symmetric margin around CC=1.0. "
            "The 5% (0.95/1.05) default is a policy placeholder that absorbs "
            "typical estimation error and modeling drift; adjust via policy or "
            "derive from empirical uncertainty for audits."
        )
    return CCRegimeThresholds(
        constructive=1.0 - margin,
        destructive=1.0 + margin,
        margin=margin,
        rationale=rationale,
    )


def derive_cc_thresholds_from_uncertainty(
    cc_interval_width: float,
    relative_floor: float = DEFAULT_CC_RELATIVE_MARGIN,
    absolute_floor: float = DEFAULT_CC_ABSOLUTE_MARGIN,
    uncertainty_multiplier: float = DEFAULT_CC_UNCERTAINTY_MULTIPLIER,
) -> CCRegimeThresholds:
    """
    Derive CC regime thresholds by expanding the independence band to
    cover uncertainty in the CC interval.

    The margin is computed as:
        margin = max(relative_floor, absolute_floor, uncertainty_multiplier * cc_interval_width)

    This makes the independence band at least as wide as a configurable
    fraction of the CC uncertainty, while never narrower than the policy
    floor. This helps avoid classifying noise-driven intervals as
    constructive/destructive.
    """
    if cc_interval_width < 0:
        raise ValueError("cc_interval_width must be non-negative.")
    if relative_floor < 0 or absolute_floor < 0 or uncertainty_multiplier < 0:
        raise ValueError("Floors and multiplier must be non-negative.")

    margin = max(relative_floor, absolute_floor, uncertainty_multiplier * cc_interval_width)
    rationale = (
        "Independence band widened to absorb CC interval uncertainty. "
        "Margin = max(relative_floor, absolute_floor, uncertainty_multiplier * cc_width)."
    )
    return CCRegimeThresholds(
        constructive=1.0 - margin,
        destructive=1.0 + margin,
        margin=margin,
        rationale=rationale,
    )


def compute_cc_bounds(
    j_lower: float,
    j_upper: float,
    max_individual_j: float,
) -> Tuple[float, float]:
    """
    Compute CC bounds from a J interval and the strongest individual J.

    CC = J_comp / max(J_single) for max(J_single) > 0.
    Interval arithmetic yields:
        cc_lower = j_lower / max_individual_j
        cc_upper = j_upper / max_individual_j
    """
    if max_individual_j <= MATHEMATICAL_TOLERANCE:
        raise ValueError("max_individual_j must be positive to define CC.")
    cc_lower = j_lower / max_individual_j
    cc_upper = j_upper / max_individual_j
    if cc_lower > cc_upper:
        cc_lower, cc_upper = cc_upper, cc_lower
    return cc_lower, cc_upper


# ---------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------


class FHBoundViolationError(Exception):
    """Raised when computed bounds violate mathematical constraints."""


class StatisticalValidationError(Exception):
    """Raised when statistical assumptions are violated."""


class NumericalInstabilityError(Exception):
    """Raised when numerical computations become unstable."""


# ---------------------------------------------------------------------
# Core bound containers
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FHBounds:
    """
    Immutable container for Fréchet-Hoeffding type bounds on a probability.

    Semantics
    ---------
    Represents an interval [lower, upper] ⊆ [0, 1] for the probability of some
    composite event (e.g., intersection/union of rails, TPR/FPR band, etc.),
    together with the per-rail marginals that were used to derive it.

    Attributes
    ----------
    lower:
        Lower bound on P(event). Must be finite and in [0, 1] up to tolerance.
    upper:
        Upper bound on P(event). Must be finite and in [0, 1] up to tolerance.
    marginals:
        Tuple of per-rail marginal probabilities used in the construction.
        Each must be a finite value in [0, 1].
    bound_type:
        Human-readable tag describing what this interval represents, e.g.:
        "intersection", "union", "tpr_serial_or", "fpr_parallel_and", etc.
    k_rails:
        Number of constituent rails/events. Must equal len(marginals).
    is_sharp:
        Whether this interval is mathematically sharp (attainable) under some
        joint construction (typically True for pure FH bounds).
    construction_proof:
        Optional textual note or reference sketch showing how sharpness
        or the inequality was derived.

    Invariants (enforced in __post_init__)
    --------------------------------------
    - lower and upper are finite floats (not NaN, not ±inf).
    - 0 <= lower <= upper <= 1 (within MATHEMATICAL_TOLERANCE).
    - len(marginals) == k_rails and k_rails ≥ 1.
    - Each marginal pᵢ is finite and lies in [0, 1].
    - If |upper - lower| < MATHEMATICAL_TOLERANCE but upper != lower,
      snap to midpoint and emit a warning (treat as near-degenerate).
    """

    lower: float
    upper: float
    marginals: Tuple[Probability, ...]
    bound_type: str
    k_rails: int
    is_sharp: bool = True
    construction_proof: str | None = None

    def __post_init__(self) -> None:
        # Cast to float and check finiteness early (avoid weird numeric types).
        lower = float(self.lower)
        upper = float(self.upper)

        if math.isnan(lower) or math.isinf(lower):
            raise FHBoundViolationError(f"lower bound is not finite: {lower!r}")
        if math.isnan(upper) or math.isinf(upper):
            raise FHBoundViolationError(f"upper bound is not finite: {upper!r}")

        # Structural consistency: marginals length vs k_rails.
        if self.k_rails <= 0:
            raise FHBoundViolationError(f"k_rails must be positive, got {self.k_rails}")
        if len(self.marginals) != self.k_rails:
            raise FHBoundViolationError(
                f"k_rails={self.k_rails} does not match len(marginals)={len(self.marginals)}"
            )

        # Validate marginals as probabilities in [0, 1].
        # This uses the module-level helper and will raise on NaN/Inf/out-of-range.
        validate_probability_vector(self.marginals, "marginals")

        # Domain constraints for the composite probability.
        if not self._approx_ge(lower, 0.0):
            raise FHBoundViolationError(
                f"Invalid lower bound {lower:.12f}; must be ≥ 0 (up to tolerance)."
            )
        if not self._approx_le(upper, 1.0):
            raise FHBoundViolationError(
                f"Invalid upper bound {upper:.12f}; must be <= 1 (up to tolerance)."
            )
        if not self._approx_le(lower, upper):
            raise FHBoundViolationError(
                f"Invalid bounds ordering: lower={lower:.12f} > upper={upper:.12f}"
            )

        # Treat suspiciously close but unequal bounds as near-degenerate.
        if abs(upper - lower) < MATHEMATICAL_TOLERANCE and upper != lower:
            midpoint = 0.5 * (upper + lower)
            warnings.warn(
                "FHBounds interval is within MATHEMATICAL_TOLERANCE but lower != upper; "
                "snapping to midpoint to avoid numerical instability.",
                RuntimeWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "lower", float(midpoint))
            object.__setattr__(self, "upper", float(midpoint))

        # Because the dataclass is frozen, we don't assign back to self.lower/upper,
        # but all subsequent properties read from the original fields, which have
        # been validated via the local 'lower'/'upper' checks above.

    @staticmethod
    def _approx_le(a: float, b: float) -> bool:
        """Return True if a <= b within numerical tolerance."""
        return a <= b + MATHEMATICAL_TOLERANCE

    @staticmethod
    def _approx_ge(a: float, b: float) -> bool:
        """Return True if a ≥ b within numerical tolerance."""
        return a + MATHEMATICAL_TOLERANCE >= b

    @property
    def width(self) -> float:
        """Width of the interval [lower, upper]."""
        return float(self.upper) - float(self.lower)

    @property
    def is_degenerate(self) -> bool:
        """True if bounds collapse to a single point (within tolerance)."""
        return abs(float(self.upper) - float(self.lower)) < MATHEMATICAL_TOLERANCE

    def contains(self, value: float, strict: bool = False) -> bool:
        """
        Test if value lies within bounds, with numerical tolerance.

        Parameters
        ----------
        value:
            Value to test.
        strict:
            If True, require strict inclusion (excluding endpoints up to EPSILON).
            If False, include endpoints with EPSILON slack.
        """
        v = float(value)
        if strict:
            return (float(self.lower) + EPSILON) < v < (float(self.upper) - EPSILON)
        return (float(self.lower) - EPSILON) <= v <= (float(self.upper) + EPSILON)


@dataclass(frozen=True, slots=True)
class ComposedJBounds:
    """
    J-statistic bounds for a composed multi-rail system.

    Semantics
    ---------
    The composition is defined in terms of:

    - A miss-event H (e.g., "attack slips through"), with FH bounds `miss_bounds`.
    - A false-alarm event B (e.g., "benign input flagged"), with FH bounds `alarm_bounds`.
    - TPR bounds derived as TPR = 1 - P(H).
    - FPR bounds taken as P(B).
    - J = TPR - FPR.

    This dataclass stores:
    - J bounds (j_lower, j_upper),
    - the TPR/FPR FH bounds,
    - the underlying miss/false-alarm FH bounds,
    - individual-rail J-statistics for comparison,
    - composition type (e.g., "serial_or" or "parallel_and"),
    - number of rails.
    """

    j_lower: JStatistic
    j_upper: JStatistic
    tpr_bounds: FHBounds
    fpr_bounds: FHBounds
    miss_bounds: FHBounds
    alarm_bounds: FHBounds
    individual_j_stats: Tuple[JStatistic, ...]
    composition_type: str
    k_rails: int

    def __post_init__(self) -> None:
        # Check basic J ordering and range
        if not (
            -1.0 - MATHEMATICAL_TOLERANCE
            <= self.j_lower
            <= self.j_upper
            <= 1.0 + MATHEMATICAL_TOLERANCE
        ):
            raise FHBoundViolationError(
                f"Invalid J bounds: [{self.j_lower:.12f}, {self.j_upper:.12f}]. "
                f"Must satisfy -1 <= lower <= upper <= 1 (up to tolerance)."
            )

        # Enforce k_rails consistency
        if self.tpr_bounds.k_rails != self.k_rails or self.fpr_bounds.k_rails != self.k_rails:
            raise FHBoundViolationError(
                "k_rails mismatch between composed bounds and TPR/FPR bounds."
            )

        if len(self.individual_j_stats) not in (0, self.k_rails):
            raise FHBoundViolationError(
                f"len(individual_j_stats)={len(self.individual_j_stats)} must be 0 or k_rails={self.k_rails}."
            )

        # Consistency check: J = TPR - FPR at the interval level
        expected_j_lower = self.tpr_bounds.lower - self.fpr_bounds.upper
        expected_j_upper = self.tpr_bounds.upper - self.fpr_bounds.lower

        if abs(self.j_lower - expected_j_lower) > MATHEMATICAL_TOLERANCE:
            raise FHBoundViolationError(
                f"J lower bound inconsistency: j_lower={self.j_lower:.12f} "
                f"!= tpr_lower - fpr_upper={expected_j_lower:.12f}"
            )

        if abs(self.j_upper - expected_j_upper) > MATHEMATICAL_TOLERANCE:
            raise FHBoundViolationError(
                f"J upper bound inconsistency: j_upper={self.j_upper:.12f} "
                f"!= tpr_upper - fpr_lower={expected_j_upper:.12f}"
            )

    @property
    def width(self) -> float:
        """Width of the J-statistic interval."""
        return self.j_upper - self.j_lower

    def classify_regime(
        self,
        threshold_constructive: float = 0.95,
        threshold_destructive: float = 1.05,
        threshold_policy: CCRegimeThresholds | None = None,
    ) -> Dict[str, Union[str, float, bool, Tuple[float, float]]]:
        """
        Classify compositional regime based on J/CC bounds.

        We define an interval CC = J_comp / max(J_single), using:
        - cc_lower = j_lower / max_individual_j
        - cc_upper = j_upper / max_individual_j

        Regime proof sketch
        -------------------
        1) FH + composition yields J_comp ∈ [j_lower, j_upper].
        2) Let J_max = max(J_single) with J_max > 0.
        3) Divide the entire interval by J_max (interval arithmetic) to get
           CC ∈ [j_lower / J_max, j_upper / J_max].
        4) Independence is defined as CC ≈ 1. A policy margin ε defines the
           independence band [1 - ε, 1 + ε].
        5) If cc_upper < 1 - ε, then all admissible CC values are strictly
           below the independence band → constructive interference.
        6) If cc_lower > 1 + ε, then all admissible CC values are above the
           independence band → destructive interference.
        7) If CC interval lies entirely within the band, we classify as
           independent; otherwise, the regime is uncertain.

        The default ε is 0.05 (0.95/1.05), which is a policy placeholder that
        should be tightened or expanded based on uncertainty analysis.

        and classify:
        - "constructive" if cc_upper < threshold_constructive
        - "destructive" if cc_lower > threshold_destructive
        - "independent" if [cc_lower, cc_upper] ⊆ [threshold_constructive, threshold_destructive]
        - "uncertain" otherwise

        Confidence is heuristically derived from how much the CC interval
        overlaps the "independent" band.
        """
        if not self.individual_j_stats:
            return {
                "regime": "undefined",
                "confidence": 0.0,
                "reason": "no individual stats",
            }

        max_individual = max(self.individual_j_stats)
        if max_individual < MATHEMATICAL_TOLERANCE:
            return {
                "regime": "degenerate",
                "confidence": 1.0,
                "reason": "no individual rail effectiveness (max J ≈ 0)",
            }

        cc_lower, cc_upper = compute_cc_bounds(self.j_lower, self.j_upper, max_individual)

        if threshold_policy is None:
            threshold_policy = default_cc_regime_thresholds(
                relative_margin=DEFAULT_CC_RELATIVE_MARGIN,
                absolute_margin=DEFAULT_CC_ABSOLUTE_MARGIN,
            )

        threshold_constructive = threshold_policy.constructive
        threshold_destructive = threshold_policy.destructive
        margin = threshold_policy.margin

        # Regime classification
        if cc_upper < threshold_constructive:
            regime = "constructive"
            confidence = 1.0
        elif cc_lower > threshold_destructive:
            regime = "destructive"
            confidence = 1.0
        elif (cc_lower >= threshold_constructive) and (cc_upper <= threshold_destructive):
            regime = "independent"
            confidence = 1.0
        else:
            # Uncertain regime: CC interval spans multiple bands
            regime = "uncertain"
            total_span = threshold_destructive - threshold_constructive
            if total_span <= 0:
                confidence = 0.5
            else:
                # Overlap of [cc_lower, cc_upper] with the independent band
                overlap_left = max(cc_lower, threshold_constructive)
                overlap_right = min(cc_upper, threshold_destructive)
                overlap_span = max(0.0, overlap_right - overlap_left)
                # More overlap => more "independent" confidence, but we invert for uncertainty.
                confidence = max(0.0, min(1.0, 1.0 - overlap_span / total_span))

        return {
            "regime": regime,
            "confidence": confidence,
            "cc_bounds": (cc_lower, cc_upper),
            "cc_width": cc_upper - cc_lower,
            "max_individual_j": max_individual,
            "thresholds": (threshold_constructive, threshold_destructive),
            "margin": margin,
            "threshold_rationale": threshold_policy.rationale,
        }


# ---------------------------------------------------------------------
# Probability-vector validation
# ---------------------------------------------------------------------


def validate_probability_vector(probs: Sequence[float], name: str) -> None:
    """
    Validate that a probability vector is non-empty and each element is a
    finite number in [0, 1].

    This is a defensive, research-grade validator:
    - Rejects empty sequences.
    - Casts each element to float and rejects NaN / ±inf.
    - Enforces 0.0 <= p <= 1.0 for every entry.
    - On failure, reports *all* offending indices with rich detail.

    Parameters
    ----------
    probs:
        Sequence of probability-like values (ints, floats, numpy scalars, etc.).
    name:
        Human-readable name used in error messages (e.g. "miss_rates").

    Raises
    ------
    ValueError
        If `probs` is empty or contains any invalid probability. The exception
        message includes the indices and representations of all offending
        entries, plus the cast float value or the casting error message.
    """
    if len(probs) == 0:
        raise ValueError(f"{name} cannot be empty")

    # Collect all invalid entries instead of failing on the first one, so the
    # caller gets a complete picture of what's wrong.
    invalid_values: List[Tuple[int, Any, Any]] = []

    for i, p in enumerate(probs):
        try:
            fp = float(p)
        except (TypeError, ValueError) as exc:
            # Non-numeric or non-castable type
            invalid_values.append((i, p, f"type_error: {exc}"))
            continue

        # Reject NaN, infinities, and values outside [0, 1]
        if math.isnan(fp) or math.isinf(fp) or not (0.0 <= fp <= 1.0):
            invalid_values.append((i, p, fp))

    if invalid_values:
        # Build a detailed multi-line error message for debugging
        details_lines = []
        for idx, original, info in invalid_values:
            details_lines.append(f"  - index {idx}: value={original!r}, info={info!r}")
        details = "\n".join(details_lines)

        raise ValueError(
            f"{name} contains {len(invalid_values)} invalid probabilities.\n"
            f"Each probability must be a finite real number in [0, 1].\n"
            f"Offending entries:\n{details}"
        )


# ---------------------------------------------------------------------
# Core FH / Hoeffding bounds
# ---------------------------------------------------------------------


def frechet_intersection_lower_bound(marginals: Sequence[Probability]) -> Probability:
    """
    Fréchet-Hoeffding lower bound for the intersection probability.

    For events A₁, …, A_k with P(Aᵢ) = pᵢ ∈ [0, 1], the Fréchet-Hoeffding
    inequality gives

        P(n Aᵢ) ≥ max{0, ∑ pᵢ - (k - 1)}.

    This lower bound is *sharp*: there exist joint distributions over
    (A₁, …, A_k) that attain it.

    Parameters
    ----------
    marginals:
        Sequence of marginal probabilities pᵢ in [0, 1]. Must be non-empty.

    Returns
    -------
    Probability
        The Fréchet-Hoeffding lower bound on P(n Aᵢ), clamped into [0, 1].

    Raises
    ------
    ValueError
        If `marginals` is empty or contains invalid probabilities.
    FHBoundViolationError
        If, after numerical clamping, the bound exceeds min(marginals) by more
        than MATHEMATICAL_TOLERANCE (indicating a logical or numerical bug).
    """
    # 1) Validate and normalize marginals to plain floats in [0, 1].
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]
    k = len(p)

    if k == 1:
        # For a single event, the intersection is just the event itself.
        return p[0]

    # 2) Compute the raw Fréchet lower bound: max{0, Σ pᵢ - (k - 1)}.
    total = sum(p)
    raw_bound = total - (k - 1)

    # Clamp into [0, 1] defensively (float roundoff can push tiny amounts out).
    bound = max(0.0, min(1.0, raw_bound))

    # 3) Mathematical consistency: bound must not exceed min(pᵢ) in exact math.
    min_p = min(p)
    if bound > min_p + MATHEMATICAL_TOLERANCE:
        # This should never happen if inputs are valid probabilities and the
        # formula is implemented correctly; treat as a hard invariant failure.
        raise FHBoundViolationError(
            "frechet_intersection_lower_bound produced a value that exceeds "
            f"min(marginals) beyond tolerance.\n"
            f"  k          = {k}\n"
            f"  marginals  = {p}\n"
            f"  sum(p)     = {total:.12f}\n"
            f"  raw_bound  = {raw_bound:.12f}\n"
            f"  bound      = {bound:.12f}\n"
            f"  min(p)     = {min_p:.12f}\n"
            f"  tolerance  = {MATHEMATICAL_TOLERANCE:.3e}"
        )

    # 4) Final safety clamp to enforce <= min(pᵢ) numerically.
    return float(min(bound, min_p))


def hoeffding_intersection_upper_bound(marginals: Sequence[Probability]) -> Probability:
    """
    Hoeffding upper bound for the intersection probability.

    For events A₁, …, A_k with P(Aᵢ) = pᵢ ∈ [0, 1], the intersection satisfies

        P(n Aᵢ) <= P(Aⱼ)  for every j,
        => P(n Aᵢ) <= minᵢ pᵢ.

    This upper bound is *sharp*: attainable by, for example, choosing a nested
    set construction A₁ ⊆ A₂ ⊆ ⋯ ⊆ A_k with P(Aᵢ) = pᵢ, in which case

        P(n Aᵢ) = P(A₁) = minᵢ pᵢ.

    Parameters
    ----------
    marginals:
        Sequence of marginal probabilities pᵢ in [0, 1]. Must be non-empty.

    Returns
    -------
    Probability
        The Hoeffding upper bound on P(n Aᵢ), guaranteed to lie in [0, 1]
        (up to numerical tolerance).

    Raises
    ------
    ValueError
        If `marginals` is empty or contains invalid probabilities.
    """
    # Validate and normalize marginals.
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]

    # With valid probabilities, min(p) is already in [0, 1]; we still clamp
    # defensively in case of exotic numeric types / roundoff.
    upper = min(p)
    upper = max(0.0, min(1.0, upper))

    return float(upper)


def frechet_union_lower_bound(marginals: Sequence[Probability]) -> Probability:
    """
    Fréchet lower bound for the union probability.

    For events A₁, …, A_k with P(Aᵢ) = pᵢ ∈ [0, 1], the union satisfies

        P(U Aᵢ) ≥ P(Aⱼ)  for every j,
        => P(U Aᵢ) ≥ maxᵢ pᵢ.

    This lower bound is *sharp*: attainable by, for example, choosing a nested
    set construction A₁ ⊆ A₂ ⊆ ⋯ ⊆ A_k with P(Aᵢ) = pᵢ, in which case

        P(U Aᵢ) = P(A_k) = maxᵢ pᵢ.

    Parameters
    ----------
    marginals:
        Sequence of marginal probabilities pᵢ in [0, 1]. Must be non-empty.

    Returns
    -------
    Probability
        The Fréchet lower bound on P(U Aᵢ), guaranteed to lie in [0, 1]
        (up to numerical tolerance).

    Raises
    ------
    ValueError
        If `marginals` is empty or contains invalid probabilities.
    """
    # Validate and normalize marginals.
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]

    # With valid probabilities, max(p) is already in [0, 1]; we still clamp
    # defensively in case of exotic numeric types / roundoff.
    lower = max(p)
    lower = max(0.0, min(1.0, lower))

    return float(lower)


def hoeffding_union_upper_bound(marginals: Sequence[Probability]) -> Probability:
    """
    Hoeffding (Boole) upper bound for the union probability.

    For events A₁, …, A_k with P(Aᵢ) = pᵢ ∈ [0, 1], the union U = U Aᵢ satisfies

        P(U) <= ∑ pᵢ        (Boole's inequality)
        P(U) <= 1           (by definition of probability)

    Hence

        P(U) <= min(1, ∑ pᵢ).

    Sharpness:
        * If the events are pairwise disjoint and ∑ pᵢ <= 1, then
              P(U) = ∑ pᵢ.
        * If ∑ pᵢ ≥ 1, one can construct events whose union covers the entire
          sample space, achieving P(U) = 1 while preserving the marginals.

    Parameters
    ----------
    marginals:
        Sequence of marginal probabilities pᵢ in [0, 1]. Must be non-empty.

    Returns
    -------
    Probability
        Hoeffding upper bound on P(U Aᵢ), guaranteed to lie in [0, 1]
        (up to numerical tolerance).

    Raises
    ------
    ValueError
        If `marginals` is empty or contains invalid probabilities.
    """
    # Validate and normalize marginals
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]

    # Boole + trivial <= 1
    s = float(sum(p))
    upper = min(1.0, s)

    # Defensive clamp into [0, 1] in case of exotic numeric types / rounding
    upper = max(0.0, min(1.0, upper))

    return upper


def intersection_bounds(marginals: Sequence[Probability]) -> FHBounds:
    """
    Compute Fréchet-Hoeffding bounds for the intersection probability P(n Aᵢ).

    Semantics
    ---------
    For events A₁, …, A_k with marginals P(Aᵢ) = pᵢ ∈ [0, 1], the probability of
    the intersection H = n Aᵢ is bounded by the classical Fréchet-Hoeffding
    envelope:

        max(0, Σ pᵢ - (k - 1)) <= P(H) <= minᵢ pᵢ.

    Both bounds are *sharp*:
      * The lower bound is achievable via extremal Fréchet couplings
        (Bonferroni/Frechet construction).
      * The upper bound is achievable by nested sets A₁ ⊆ A₂ ⊆ … ⊆ A_k.

    Parameters
    ----------
    marginals:
        Sequence of marginal probabilities pᵢ in [0, 1]. Must be non-empty and
        each entry finite and within [0, 1].

    Returns
    -------
    FHBounds
        An FHBounds instance with:
            - lower = max(0, Σ pᵢ - (k - 1)) clamped into [0, 1],
            - upper = minᵢ pᵢ clamped into [0, 1],
            - marginals = tuple of the input probabilities (as floats),
            - bound_type = "intersection",
            - k_rails = len(marginals),
            - is_sharp = True.

    Raises
    ------
    ValueError
        If `marginals` is empty or contains invalid probabilities (via
        validate_probability_vector).
    FHBoundViolationError
        If the derived lower/upper are inconsistent (lower > upper beyond
        MATHEMATICAL_TOLERANCE).
    """
    # Validate and normalize marginals.
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]
    k = len(p)

    # Core FH intersection bounds.
    lower = float(frechet_intersection_lower_bound(p))
    upper = float(hoeffding_intersection_upper_bound(p))

    # Defensive clamp into [0, 1] to guard against tiny numerical drift.
    lower = max(0.0, min(1.0, lower))
    upper = max(0.0, min(1.0, upper))

    # Internal consistency check: lower must not exceed upper beyond tolerance.
    if lower > upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            f"Inconsistent intersection bounds: lower={lower:.12f} > upper={upper:.12f}"
        )

    return FHBounds(
        lower=lower,
        upper=upper,
        marginals=tuple(p),
        bound_type="intersection",
        k_rails=k,
        construction_proof=(
            "Fréchet-Hoeffding intersection bounds: "
            "lower = max(0, Σ p_i - (k - 1)); upper = min_i p_i."
        ),
    )


def union_bounds(marginals: Sequence[Probability]) -> FHBounds:
    """
    Compute Fréchet-Hoeffding bounds for the union probability P(U Aᵢ).

    Semantics
    ---------
    For events A₁, …, A_k with marginals P(Aᵢ) = pᵢ ∈ [0, 1], the probability of
    the union U = U Aᵢ is bounded by:

        maxᵢ pᵢ <= P(U) <= min(1, Σ pᵢ).

    Both bounds are *sharp*:
      * The lower bound is achievable by nested sets A₁ ⊆ A₂ ⊆ … ⊆ A_k.
      * The upper bound is achievable by disjoint events when Σ pᵢ <= 1, and by
        suitable overlapping constructions otherwise.

    Parameters
    ----------
    marginals:
        Sequence of marginal probabilities pᵢ in [0, 1]. Must be non-empty and
        each entry finite and within [0, 1].

    Returns
    -------
    FHBounds
        An FHBounds instance with:
            - lower = maxᵢ pᵢ clamped into [0, 1],
            - upper = min(1, Σ pᵢ) clamped into [0, 1],
            - marginals = tuple of the input probabilities (as floats),
            - bound_type = "union",
            - k_rails = len(marginals),
            - is_sharp = True.

    Raises
    ------
    ValueError
        If `marginals` is empty or contains invalid probabilities (via
        validate_probability_vector).
    FHBoundViolationError
        If the derived lower/upper are inconsistent (lower > upper beyond
        MATHEMATICAL_TOLERANCE).
    """
    # Validate and normalize marginals.
    validate_probability_vector(marginals, "marginals")
    p = [float(x) for x in marginals]

    # Core FH union bounds.
    lower = float(frechet_union_lower_bound(p))
    upper = float(hoeffding_union_upper_bound(p))

    # Defensive clamp into [0, 1] to guard against tiny numerical drift.
    lower = max(0.0, min(1.0, lower))
    upper = max(0.0, min(1.0, upper))

    # Internal consistency check: lower must not exceed upper beyond tolerance.
    if lower > upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            f"Inconsistent union bounds: lower={lower:.12f} > upper={upper:.12f}"
        )

    return FHBounds(
        lower=lower,
        upper=upper,
        marginals=tuple(p),
        bound_type="union",
        k_rails=len(p),
        construction_proof=(
            "Fréchet-Hoeffding union bounds: lower = max_i p_i; upper = min(1, Σ p_i)."
        ),
    )


# ---------------------------------------------------------------------
# Composition: serial OR / parallel AND
# ---------------------------------------------------------------------


def serial_or_composition_bounds(
    miss_rates: Sequence[Probability],
    false_alarm_rates: Sequence[Probability],
) -> ComposedJBounds:
    """
    Serial OR composition (system fires if ANY rail fires).

    Semantics
    ---------
    We consider k detection rails. For a fixed attack distribution:

    - On an adversarial input (world 1):
        * Rail i misses with probability m_i = miss_rates[i].
        * Let F_i be the “rail i missed” event.
        * Joint miss event: H = n_i F_i  (all rails miss).
        * Thus  TPR = 1 - P(H).

    - On a benign input (world 0):
        * Rail i false-alarms with probability f_i = false_alarm_rates[i].
        * Let A_i be the “rail i false-alarms” event.
        * Joint false-alarm event: B = U_i A_i  (at least one rail false-alarms).
        * Thus  FPR = P(B).

    We first apply Fréchet-Hoeffding bounds to P(H) and P(B):

        H_bounds = intersection_bounds(miss_rates)   # P(H) = P(n F_i)
        B_bounds = union_bounds(false_alarm_rates)  # P(B) = P(U A_i)

    and then push them through:

        TPR ∈ [1 - H_upper, 1 - H_lower]
        FPR ∈ [B_lower,      B_upper]

    Finally:

        J = TPR - FPR  =>
        J_lower = TPR_lower - FPR_upper
        J_upper = TPR_upper - FPR_lower

    All bounds are guaranteed to respect probability constraints up to
    MATHEMATICAL_TOLERANCE and are mathematically sharp at the level of FH.
    """
    # -------------------------------
    # 0. Dimension and validity checks
    # -------------------------------
    if not miss_rates or not false_alarm_rates:
        raise ValueError("serial_or_composition_bounds: rate vectors must be non-empty")

    if len(miss_rates) != len(false_alarm_rates):
        raise ValueError(
            "serial_or_composition_bounds: dimension mismatch: "
            f"{len(miss_rates)} miss rates vs {len(false_alarm_rates)} false alarm rates"
        )

    validate_probability_vector(miss_rates, "miss_rates")
    validate_probability_vector(false_alarm_rates, "false_alarm_rates")

    k = len(miss_rates)

    # Cast to floats once, to avoid surprises with numpy/Decimal/etc.
    miss = [float(m) for m in miss_rates]
    fa = [float(f) for f in false_alarm_rates]

    # --------------------------------------
    # 1. FH bounds for joint miss P(H) = P(n F_i)
    # --------------------------------------
    miss_b = intersection_bounds(miss)  # FHBounds for P(H)

    # Sanity: ensure intersection_bounds' k_rails agrees with k
    if miss_b.k_rails != k:
        raise FHBoundViolationError(
            "serial_or_composition_bounds: miss_bounds.k_rails "
            f"({miss_b.k_rails}) inconsistent with k={k}"
        )

    # ----------------------------------------------
    # 2. FH bounds for joint false alarm P(B) = P(U A_i)
    # ----------------------------------------------
    alarm_b = union_bounds(fa)  # FHBounds for P(B)

    if alarm_b.k_rails != k:
        raise FHBoundViolationError(
            "serial_or_composition_bounds: alarm_bounds.k_rails "
            f"({alarm_b.k_rails}) inconsistent with k={k}"
        )

    # -----------------------------
    # 3. TPR bounds via P(H) bounds
    # -----------------------------
    # Worst-case TPR: largest joint miss
    tpr_lower = 1.0 - float(miss_b.upper)
    # Best-case TPR: smallest joint miss
    tpr_upper = 1.0 - float(miss_b.lower)

    # Clamp defensively into [0, 1] to avoid tiny negative/ >1 drift.
    tpr_lower = max(0.0, min(1.0, tpr_lower))
    tpr_upper = max(0.0, min(1.0, tpr_upper))

    if tpr_lower > tpr_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            "serial_or_composition_bounds: TPR bounds inconsistent: "
            f"tpr_lower={tpr_lower:.12f} > tpr_upper={tpr_upper:.12f}"
        )

    # Per-rail TPR marginals: complements of miss rates.
    tpr_marginals = tuple(max(0.0, min(1.0, 1.0 - m)) for m in miss)

    tpr_b = FHBounds(
        lower=tpr_lower,
        upper=tpr_upper,
        marginals=tpr_marginals,
        bound_type="tpr_serial_or",
        k_rails=k,
        construction_proof="TPR = 1 - P(H) with FH bounds on H = n F_i.",
    )

    # -----------------------------
    # 4. FPR bounds via P(B) bounds
    # -----------------------------
    fpr_lower = float(alarm_b.lower)
    fpr_upper = float(alarm_b.upper)

    # Clamp into [0, 1] defensively (should already hold from union_bounds).
    fpr_lower = max(0.0, min(1.0, fpr_lower))
    fpr_upper = max(0.0, min(1.0, fpr_upper))

    if fpr_lower > fpr_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            "serial_or_composition_bounds: FPR bounds inconsistent: "
            f"fpr_lower={fpr_lower:.12f} > fpr_upper={fpr_upper:.12f}"
        )

    fpr_b = FHBounds(
        lower=fpr_lower,
        upper=fpr_upper,
        marginals=tuple(fa),
        bound_type="fpr_serial_or",
        k_rails=k,
        construction_proof="FPR = P(B) with FH bounds on B = U A_i.",
    )

    # -----------------------------
    # 5. J-statistic bounds
    # -----------------------------
    j_lower = tpr_b.lower - fpr_b.upper
    j_upper = tpr_b.upper - fpr_b.lower

    # In principle J ∈ [-1, 1] automatically since TPR,FPR ∈ [0,1], but we
    # clamp defensively in case of tiny numerical overshoot.
    j_lower = max(-1.0, min(1.0, j_lower))
    j_upper = max(-1.0, min(1.0, j_upper))

    if j_lower > j_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            "serial_or_composition_bounds: J bounds inconsistent: "
            f"j_lower={j_lower:.12f} > j_upper={j_upper:.12f}"
        )

    # Per-rail J statistics for comparison / CC metrics
    individual_j_stats = tuple(
        (max(0.0, min(1.0, 1.0 - m))) - f for m, f in zip(miss, fa, strict=False)
    )

    return ComposedJBounds(
        j_lower=j_lower,
        j_upper=j_upper,
        tpr_bounds=tpr_b,
        fpr_bounds=fpr_b,
        miss_bounds=miss_b,
        alarm_bounds=alarm_b,
        individual_j_stats=individual_j_stats,
        composition_type="serial_or",
        k_rails=k,
    )


def parallel_and_composition_bounds(
    miss_rates: Sequence[Probability],
    false_alarm_rates: Sequence[Probability],
) -> ComposedJBounds:
    """
    Parallel AND composition (system fires only if ALL rails fire).

    Semantics
    ---------
    We consider k detection rails. For a fixed attack distribution:

    - On an adversarial input (world 1):
        * Rail i misses with probability m_i = miss_rates[i].
        * Let F_i be the “rail i missed” event.
        * Joint miss event for AND system: H = U_i F_i  (at least one misses).
        * Thus  TPR = 1 - P(H).

    - On a benign input (world 0):
        * Rail i false-alarms with probability f_i = false_alarm_rates[i].
        * Let A_i be the “rail i false-alarms” event.
        * Joint false-alarm event for AND system: B = n_i A_i (all rails false-alarm).
        * Thus  FPR = P(B).

    We first apply Fréchet-Hoeffding bounds to P(H) and P(B):

        H_bounds = union_bounds(miss_rates)         # P(H) = P(U F_i)
        B_bounds = intersection_bounds(false_alarm_rates)  # P(B) = P(n A_i)

    and then push them through:

        TPR ∈ [1 - H_upper, 1 - H_lower]
        FPR ∈ [B_lower,      B_upper]

    Finally:

        J = TPR - FPR  =>
        J_lower = TPR_lower - FPR_upper
        J_upper = TPR_upper - FPR_lower

    All bounds are guaranteed to respect probability constraints up to
    MATHEMATICAL_TOLERANCE and are mathematically sharp at the FH level.
    """
    # -------------------------------
    # 0. Dimension and validity checks
    # -------------------------------
    if not miss_rates or not false_alarm_rates:
        raise ValueError("parallel_and_composition_bounds: rate vectors must be non-empty")

    if len(miss_rates) != len(false_alarm_rates):
        raise ValueError(
            "parallel_and_composition_bounds: dimension mismatch: "
            f"{len(miss_rates)} miss rates vs {len(false_alarm_rates)} false alarm rates"
        )

    validate_probability_vector(miss_rates, "miss_rates")
    validate_probability_vector(false_alarm_rates, "false_alarm_rates")

    k = len(miss_rates)

    # Cast to floats once for safety with numpy/Decimal/etc.
    miss = [float(m) for m in miss_rates]
    fa = [float(f) for f in false_alarm_rates]

    # --------------------------------------
    # 1. FH bounds for joint miss P(H) = P(U F_i)
    # --------------------------------------
    miss_b = union_bounds(miss)  # FHBounds for P(H)

    if miss_b.k_rails != k:
        raise FHBoundViolationError(
            "parallel_and_composition_bounds: miss_bounds.k_rails "
            f"({miss_b.k_rails}) inconsistent with k={k}"
        )

    # ----------------------------------------------
    # 2. FH bounds for joint false alarm P(B) = P(n A_i)
    # ----------------------------------------------
    alarm_b = intersection_bounds(fa)  # FHBounds for P(B)

    if alarm_b.k_rails != k:
        raise FHBoundViolationError(
            "parallel_and_composition_bounds: alarm_bounds.k_rails "
            f"({alarm_b.k_rails}) inconsistent with k={k}"
        )

    # -----------------------------
    # 3. TPR bounds via P(H) bounds
    # -----------------------------
    tpr_lower = 1.0 - float(miss_b.upper)
    tpr_upper = 1.0 - float(miss_b.lower)

    # Clamp into [0, 1] defensively.
    tpr_lower = max(0.0, min(1.0, tpr_lower))
    tpr_upper = max(0.0, min(1.0, tpr_upper))

    if tpr_lower > tpr_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            "parallel_and_composition_bounds: TPR bounds inconsistent: "
            f"tpr_lower={tpr_lower:.12f} > tpr_upper={tpr_upper:.12f}"
        )

    tpr_marginals = tuple(max(0.0, min(1.0, 1.0 - m)) for m in miss)

    tpr_b = FHBounds(
        lower=tpr_lower,
        upper=tpr_upper,
        marginals=tpr_marginals,
        bound_type="tpr_parallel_and",
        k_rails=k,
        construction_proof="TPR = 1 - P(H) with FH bounds on H = U F_i.",
    )

    # -----------------------------
    # 4. FPR bounds via P(B) bounds
    # -----------------------------
    fpr_lower = float(alarm_b.lower)
    fpr_upper = float(alarm_b.upper)

    fpr_lower = max(0.0, min(1.0, fpr_lower))
    fpr_upper = max(0.0, min(1.0, fpr_upper))

    if fpr_lower > fpr_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            "parallel_and_composition_bounds: FPR bounds inconsistent: "
            f"fpr_lower={fpr_lower:.12f} > fpr_upper={fpr_upper:.12f}"
        )

    fpr_b = FHBounds(
        lower=fpr_lower,
        upper=fpr_upper,
        marginals=tuple(fa),
        bound_type="fpr_parallel_and",
        k_rails=k,
        construction_proof="FPR = P(B) with FH bounds on B = n A_i.",
    )

    # -----------------------------
    # 5. J-statistic bounds
    # -----------------------------
    j_lower = tpr_b.lower - fpr_b.upper
    j_upper = tpr_b.upper - fpr_b.lower

    # Clamp into [-1, 1] for numerical robustness.
    j_lower = max(-1.0, min(1.0, j_lower))
    j_upper = max(-1.0, min(1.0, j_upper))

    if j_lower > j_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            "parallel_and_composition_bounds: J bounds inconsistent: "
            f"j_lower={j_lower:.12f} > j_upper={j_upper:.12f}"
        )

    individual_j_stats = tuple(
        (max(0.0, min(1.0, 1.0 - m))) - f for m, f in zip(miss, fa, strict=False)
    )

    return ComposedJBounds(
        j_lower=j_lower,
        j_upper=j_upper,
        tpr_bounds=tpr_b,
        fpr_bounds=fpr_b,
        miss_bounds=miss_b,
        alarm_bounds=alarm_b,
        individual_j_stats=individual_j_stats,
        composition_type="parallel_and",
        k_rails=k,
    )


# ---------------------------------------------------------------------
# Independence-based composition
# ---------------------------------------------------------------------


def _stable_product(values: Sequence[float]) -> float:
    """Compute ∏ values in log space to avoid underflow."""
    log_sum = 0.0
    for value in values:
        if value <= 0.0:
            return 0.0
        log_sum += math.log(value)
    return float(math.exp(log_sum))


def _stable_product_complement(values: Sequence[float]) -> float:
    """Compute ∏ (1 - value) in log space to avoid underflow."""
    log_sum = 0.0
    for value in values:
        complement = 1.0 - value
        if complement <= 0.0:
            return 0.0
        log_sum += math.log1p(-value)
    return float(math.exp(log_sum))


def independence_serial_or_j(
    tprs: Sequence[Probability],
    fprs: Sequence[Probability],
) -> JStatistic:
    """
    Youden's J under an *independence* model for a serial-OR composition.

    Semantics
    ---------
    We have k rails, each with per-rail performance (TPR_i, FPR_i). The
    serial-OR system fires if *any* rail fires.

    Under the assumption that rails are conditionally independent given the
    world (benign/adversarial):

        - On adversarial inputs:
            * Rail i detects with probability TPR_i.
            * Rail i misses with probability (1 - TPR_i).
            * Joint miss event H ("system fails to detect") occurs iff all
              rails miss independently:
                    P(H) = ∏_i (1 - TPR_i)

        - On benign inputs:
            * Rail i false-alarms with probability FPR_i.
            * Rail i is quiet with probability (1 - FPR_i).
            * Joint false-alarm event B ("system fires spuriously") occurs iff
              at least one rail false-alarms:
                    P(B) = 1 - ∏_i (1 - FPR_i)

        Hence:
            TPR_sys = 1 - P(H) = 1 - ∏_i (1 - TPR_i)
            FPR_sys = P(B)     = 1 - ∏_i (1 - FPR_i)
            J       = TPR_sys - FPR_sys

    This helper computes that J and clamps it into [-1, 1] for numerical
    robustness.

    Args
    ----
    tprs:
        Sequence of per-rail true positive rates TPR_i in [0, 1].
    fprs:
        Sequence of per-rail false positive rates FPR_i in [0, 1].

    Returns
    -------
    JStatistic
        The independence-model Youden's J for the serial-OR composition,
        clipped into [-1, 1].

    Raises
    ------
    ValueError
        If the sequences are empty, have mismatched lengths, or contain
        invalid probabilities.
    """
    # Basic structural checks
    if not tprs or not fprs:
        raise ValueError("independence_serial_or_j: tprs and fprs must be non-empty")

    if len(tprs) != len(fprs):
        raise ValueError(
            "independence_serial_or_j: TPR and FPR vectors must have the same length "
            f"(got len(tprs)={len(tprs)}, len(fprs)={len(fprs)})"
        )

    # Validate and coerce to floats once, to avoid surprises from numpy/Decimal.
    validate_probability_vector(tprs, "tprs")
    validate_probability_vector(fprs, "fprs")

    tprs_f = [float(t) for t in tprs]
    fprs_f = [float(f) for f in fprs]

    miss_product = _stable_product_complement(tprs_f)
    no_false_alarm_product = _stable_product_complement(fprs_f)

    # System-level rates under independence
    tpr_sys = 1.0 - miss_product
    fpr_sys = 1.0 - no_false_alarm_product

    # Clamp into [0, 1] defensively (floating error can push slightly outside).
    tpr_sys = max(0.0, min(1.0, tpr_sys))
    fpr_sys = max(0.0, min(1.0, fpr_sys))

    j = tpr_sys - fpr_sys

    # Final clamp into [-1, 1] to protect downstream code from tiny drift.
    return max(-1.0, min(1.0, float(j)))


def independence_parallel_and_j(
    tprs: Sequence[Probability],
    fprs: Sequence[Probability],
) -> JStatistic:
    """
    Youden's J under an *independence* model for a parallel-AND composition.

    Semantics
    ---------
    We have k rails, each with per-rail performance (TPR_i, FPR_i). The
    parallel-AND system fires only if *all* rails fire.

    Under the assumption that rails are conditionally independent given the
    world (benign/adversarial):

        - On adversarial inputs:
            * Rail i detects with probability TPR_i.
            * Joint detection requires all rails detect:
                  TPR_sys = ∏_i TPR_i

        - On benign inputs:
            * Rail i false-alarms with probability FPR_i.
            * Joint false alarm requires all rails false-alarm:
                  FPR_sys = ∏_i FPR_i

        Hence:
            J = TPR_sys - FPR_sys

    This helper computes that J and clamps it into [-1, 1] for numerical
    robustness.

    Args
    ----
    tprs:
        Sequence of per-rail true positive rates TPR_i in [0, 1].
    fprs:
        Sequence of per-rail false positive rates FPR_i in [0, 1].

    Returns
    -------
    JStatistic
        The independence-model Youden's J for the parallel-AND composition,
        clipped into [-1, 1].

    Raises
    ------
    ValueError
        If the sequences are empty, have mismatched lengths, or contain
        invalid probabilities.
    """
    # Structural checks
    if not tprs or not fprs:
        raise ValueError("independence_parallel_and_j: tprs and fprs must be non-empty")

    if len(tprs) != len(fprs):
        raise ValueError(
            "independence_parallel_and_j: TPR and FPR vectors must have the same length "
            f"(got len(tprs)={len(tprs)}, len(fprs)={len(fprs)})"
        )

    # Validate and coerce to floats (handles numpy scalars, Decimal, etc.).
    validate_probability_vector(tprs, "tprs")
    validate_probability_vector(fprs, "fprs")

    tprs_f = [float(t) for t in tprs]
    fprs_f = [float(f) for f in fprs]

    tpr_sys = _stable_product(tprs_f)
    fpr_sys = _stable_product(fprs_f)

    # Clamp into [0, 1] defensively in case of tiny floating drift.
    tpr_sys = max(0.0, min(1.0, tpr_sys))
    fpr_sys = max(0.0, min(1.0, fpr_sys))

    j = tpr_sys - fpr_sys

    # Final clamp into [-1, 1] for robustness.
    return max(-1.0, min(1.0, float(j)))


# ---------------------------------------------------------------------
# Copula-based visualization helpers
# ---------------------------------------------------------------------


def copula_cdf(
    family: CopulaFamily,
    u: np.ndarray,
    v: np.ndarray,
    theta: float | None = None,
) -> np.ndarray:
    """
    Compute the bivariate copula CDF on a grid.

    This helper is intentionally lightweight (numpy only) and designed for
    generating *plot-ready* surfaces that visualize dependence structures.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.clip(u, EPSILON, 1.0 - EPSILON)
    v = np.clip(v, EPSILON, 1.0 - EPSILON)

    if family == "independence":
        return u * v
    if family == "comonotonic":
        return np.minimum(u, v)
    if family == "countermonotonic":
        return np.maximum(u + v - 1.0, 0.0)

    if family == "clayton":
        if theta is None or theta <= 0:
            raise ValueError("Clayton copula requires theta > 0.")
        inner = np.maximum(u ** (-theta) + v ** (-theta) - 1.0, EPSILON)
        return inner ** (-1.0 / theta)

    if family == "gumbel":
        if theta is None or theta < 1:
            raise ValueError("Gumbel copula requires theta >= 1.")
        log_u = -np.log(u)
        log_v = -np.log(v)
        return np.exp(-((log_u**theta + log_v**theta) ** (1.0 / theta)))

    if family == "frank":
        if theta is None or abs(theta) < EPSILON:
            raise ValueError("Frank copula requires theta != 0.")
        num = np.expm1(-theta * u) * np.expm1(-theta * v)
        denom = np.expm1(-theta)
        inner = 1.0 + num / denom
        inner = np.maximum(inner, EPSILON)
        return -(1.0 / theta) * np.log(inner)

    raise ValueError(f"Unknown copula family: {family!r}")


def copula_grid(
    family: CopulaFamily,
    theta: float | None = None,
    grid_size: int = 51,
) -> Dict[str, np.ndarray]:
    """
    Generate a (U, V, C) grid for copula visualization.

    Returns a dict with keys:
        - "u": grid of U coordinates
        - "v": grid of V coordinates
        - "cdf": copula CDF values

    Example:
        grid = copula_grid("clayton", theta=2.0, grid_size=51)
        # Use matplotlib to plot grid["cdf"] as a surface.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2.")

    axis = np.linspace(0.0, 1.0, grid_size)
    u, v = np.meshgrid(axis, axis)
    cdf = copula_cdf(family, u, v, theta=theta)
    return {"u": u, "v": v, "cdf": cdf}


def copula_experiment_plan() -> List[Dict[str, str]]:
    """
    Provide a structured brainstorm for copula-based visualization experiments.

    The returned list is designed to seed dossier appendix visuals and
    sensitivity analysis of CC/CII under dependence shifts.
    """
    return [
        {
            "experiment": "Independence baseline surface",
            "family": "independence",
            "goal": "Visualize the CC=1 reference case.",
            "risk_gap": "Confirms that independence is a *band*, not a point.",
        },
        {
            "experiment": "Comonotonic extreme",
            "family": "comonotonic",
            "goal": "Show maximal positive dependence (upper FH sharpness).",
            "risk_gap": "Highlights worst-case false alarms in serial OR.",
        },
        {
            "experiment": "Countermonotonic extreme",
            "family": "countermonotonic",
            "goal": "Show maximal negative dependence (lower FH sharpness).",
            "risk_gap": "Highlights best-case misses for parallel AND.",
        },
        {
            "experiment": "Clayton tail dependence sweep",
            "family": "clayton",
            "goal": "Plot theta ∈ {0.5, 1, 2, 5} to expose lower-tail coupling.",
            "risk_gap": "Reveals hidden joint failures (miss clustering).",
        },
        {
            "experiment": "Gumbel tail dependence sweep",
            "family": "gumbel",
            "goal": "Plot theta ∈ {1, 1.5, 2, 4} to expose upper-tail coupling.",
            "risk_gap": "Reveals joint alarm inflation under correlation.",
        },
        {
            "experiment": "Frank symmetric dependence sweep",
            "family": "frank",
            "goal": "Plot theta ∈ {-5, -2, 2, 5} for symmetric dependence shifts.",
            "risk_gap": "Demonstrates both constructive and destructive shifts.",
        },
    ]


# ---------------------------------------------------------------------
# Composability Interference Index (CII)
# ---------------------------------------------------------------------


def compute_composability_interference_index(
    observed_j: float,
    bounds: ComposedJBounds,
    individual_tprs: Sequence[Probability] | None = None,
    individual_fprs: Sequence[Probability] | None = None,
    use_independence_baseline: bool = True,
) -> Dict[str, Union[float, str, bool, Tuple[float, float]]]:
    """
    Composability Interference Index (CII).

    Definition
    ----------
    CII = (J_obs - J_baseline) / (J_worst - J_baseline),

    where:
    - J_obs     = observed composed J (point estimate from experiment).
    - J_baseline:
        * If `use_independence_baseline` and per-rail TPR/FPR are provided,
          J_baseline is the independence-model composition J.
        * Otherwise, J_baseline falls back to the midpoint of [j_lower, j_upper].
    - J_worst   = bounds.j_lower (pessimistic FH J bound).
    - J_best    = bounds.j_upper (optimistic FH J bound).

    Interpretation (when J_worst <= J_baseline <= J_best)
    ---------------------------------------------------
    - κ < 0: constructive (better than baseline).
    - κ ≈ 0: independent (matches baseline).
    - κ > 0: destructive (worse than baseline).
    - |κ| large: strong deviation from baseline.

    Notes
    -----
    - If per-rail TPR/FPR are not available, the baseline is a heuristic
      (midpoint of FH J interval), and this is reflected in `baseline_type`.
    - We do NOT attempt to reconstruct TPR/FPR from J alone.
    """
    # Sanity check on the bounds width (should be non-negative by construction).
    if bounds.width < -MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(f"ComposedJBounds has negative width: {bounds.width}")

    # 0. Validate observed_j as a finite J in [-1, 1] (up to tolerance).
    try:
        observed_j_f = float(observed_j)
    except (TypeError, ValueError) as e:
        raise ValueError(f"observed_j={observed_j!r} cannot be cast to float: {e}") from e

    if math.isnan(observed_j_f) or math.isinf(observed_j_f):
        raise ValueError(f"observed_j must be finite, got {observed_j_f!r}")

    if not (-1.0 - MATHEMATICAL_TOLERANCE <= observed_j_f <= 1.0 + MATHEMATICAL_TOLERANCE):
        raise ValueError(
            f"observed_j={observed_j_f:.6f} is outside the admissible J range [-1, 1] "
            "(up to numerical tolerance)"
        )

    # 1. Choose J_baseline
    baseline_type: str
    if use_independence_baseline and individual_tprs is not None and individual_fprs is not None:
        # Validate and check lengths against each other and k_rails.
        validate_probability_vector(individual_tprs, "individual_tprs")
        validate_probability_vector(individual_fprs, "individual_fprs")

        if len(individual_tprs) != len(individual_fprs):
            raise ValueError(
                "individual_tprs and individual_fprs must have the same length "
                f"(got {len(individual_tprs)} and {len(individual_fprs)})"
            )

        if len(individual_tprs) != bounds.k_rails:
            raise ValueError(
                "Length of per-rail rates must match bounds.k_rails: "
                f"k_rails={bounds.k_rails}, len(individual_tprs)={len(individual_tprs)}"
            )

        if bounds.individual_j_stats and len(bounds.individual_j_stats) != len(individual_tprs):
            raise ValueError(
                "len(individual_j_stats) must match len(individual_tprs) "
                f"when both are provided (got {len(bounds.individual_j_stats)} "
                f"and {len(individual_tprs)})."
            )

        # Independence baseline based on composition topology.
        if bounds.composition_type == "serial_or":
            j_baseline = independence_serial_or_j(individual_tprs, individual_fprs)
        elif bounds.composition_type == "parallel_and":
            j_baseline = independence_parallel_and_j(individual_tprs, individual_fprs)
        else:
            # Extremely defensive fallback (should not occur if ComposedJBounds
            # enforces composition_type).
            j_baseline = (
                float(np.mean(list(bounds.individual_j_stats)))
                if bounds.individual_j_stats
                else 0.0
            )

        baseline_type = "independence"
    else:
        # Heuristic: midpoint of FH interval as a neutral, topology-agnostic baseline.
        j_baseline = 0.5 * (bounds.j_lower + bounds.j_upper)
        baseline_type = "fh_midpoint"

    j_baseline = float(j_baseline)

    # 2. Check baseline / observation within FH interval (up to tolerance)
    j_worst = float(bounds.j_lower)
    j_best = float(bounds.j_upper)

    baseline_within_bounds = (
        j_worst - MATHEMATICAL_TOLERANCE <= j_baseline <= j_best + MATHEMATICAL_TOLERANCE
    )

    observed_within_bounds = (
        j_worst - MATHEMATICAL_TOLERANCE <= observed_j_f <= j_best + MATHEMATICAL_TOLERANCE
    )

    # 3. Compute κ with careful handling of degeneracy
    denom = j_worst - j_baseline

    if abs(denom) < MATHEMATICAL_TOLERANCE:
        # Baseline essentially coincides with worst-case bound: normalization breaks.
        kappa = 0.0
        interpretation = "degenerate"
        reliability = "low"
    else:
        kappa = (observed_j_f - j_baseline) / denom

        # Interpretation thresholds
        if kappa < -0.1:
            interpretation = "constructive"
        elif kappa > 0.1:
            interpretation = "destructive"
        else:
            interpretation = "independent"

        # Reliability heuristic based on how far baseline is from worst-case.
        spread = abs(denom)
        if spread > 0.2:
            reliability = "high"
        elif spread > 0.05:
            reliability = "moderate"
        else:
            reliability = "low"

        # If either baseline or observation fall outside FH bounds, downgrade reliability.
        if not baseline_within_bounds or not observed_within_bounds:
            reliability = "questionable" if reliability == "high" else "low"

        # Flag extreme κ values that likely indicate model mismatch / sampling noise.
        if abs(kappa) > 2.0:
            warnings.warn(
                f"CII = {kappa:.3f} is very large in magnitude; "
                "this may indicate model mismatch, data drift, or extreme sampling noise.",
                UserWarning,
                stacklevel=2,
            )
            if reliability == "high":
                reliability = "questionable"

    return {
        "cii": float(kappa),
        "interpretation": interpretation,
        "reliability": reliability,
        "baseline_type": baseline_type,
        "baseline_within_bounds": baseline_within_bounds,
        "observed_within_bounds": observed_within_bounds,
        "j_observed": float(observed_j_f),
        "j_baseline": float(j_baseline),
        "j_theoretical_bounds": (float(j_worst), float(j_best)),
        "interference_strength": abs(float(kappa)),
        "bounds_width": bounds.width,
        "composition_type": bounds.composition_type,
    }


# ---------------------------------------------------------------------
# Statistical helpers: Wilson CI & inverse normal
# ---------------------------------------------------------------------


def robust_inverse_normal(p: float) -> float:
    """
    Numerically stable inverse normal CDF Φ⁻¹(p) with multiple fallbacks.

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        z such that Φ(z) ≈ p.

    Raises
    ------
    ValueError
        If p is not in (0, 1) or cannot be interpreted as a finite float.
    """
    # 0. Cast & basic validation
    try:
        p_f = float(p)
    except (TypeError, ValueError) as e:
        raise ValueError(f"p={p!r} cannot be cast to float: {e}") from e

    if not (0.0 < p_f < 1.0):
        raise ValueError(f"p = {p_f} must be in (0, 1)")

    # Lightweight clipping to avoid underflow/overflow in extreme tails.
    p_clipped = min(max(p_f, 1e-16), 1.0 - 1e-16)

    # 1. Try math.erfcinv if available (fast, no heavy deps)
    try:  # pragma: no cover - platform dependent
        from math import erfcinv

        return math.sqrt(2.0) * erfcinv(2.0 * (1.0 - p_clipped))
    except Exception:
        pass

    # 2. Try SciPy if available
    try:  # pragma: no cover - optional dependency
        import scipy.stats  # type: ignore

        return float(scipy.stats.norm.ppf(p_clipped))
    except Exception:
        pass

    # 3. Try Python's statistics.NormalDist (stdlib, 3.8+)
    try:  # pragma: no cover - environment dependent
        from statistics import NormalDist  # type: ignore

        return float(NormalDist().inv_cdf(p_clipped))
    except Exception:
        pass

    # 4. Beasley-Springer-Moro-style approximation as final fallback.
    # Non-recursive, with central symmetry handled explicitly.
    def _bsm_inverse_normal(pc: float) -> float:
        # Exploit symmetry: Φ⁻¹(p) = -Φ⁻¹(1 - p)
        if pc > 0.5:
            pc = 1.0 - pc
            sign = -1.0
        else:
            sign = 1.0

        # Coefficients for tail approximation
        a0 = -3.969683028665376e01
        a1 = 2.209460984245205e02
        a2 = -2.759285104469687e02
        a3 = 1.383577518672690e02
        b1 = -5.447609879822406e01
        b2 = 1.615858368580409e02
        b3 = -1.556989798598866e02
        b4 = 6.680131188771972e01

        y = math.sqrt(-2.0 * math.log(pc))
        x = y + (
            (((y * a3 + a2) * y + a1) * y + a0) / ((((y * b4 + b3) * y + b2) * y + b1) * y + 1.0)
        )
        return sign * x

    return _bsm_inverse_normal(p_clipped)


def wilson_score_interval(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> Bounds:
    """
    Wilson score confidence interval for a binomial proportion.

    This is the correct tool for binomial CIs (two-sided, 1 - alpha), in
    contrast to DKW which applies to empirical CDFs, not binomial
    proportions.

    Parameters
    ----------
    successes : int
        Number of successes. Must be an integer in [0, trials].
    trials : int
        Number of Bernoulli trials. Must be a positive integer.
    alpha : float
        Significance level for a two-sided (1 - alpha) interval,
        with 0 < alpha < 1.

    Returns
    -------
    (lower, upper) : tuple of float
        Wilson score interval, clipped to [0, 1].

    Raises
    ------
    ValueError
        If inputs are invalid (non-integer counts, impossible ranges,
        or alpha outside (0, 1)).
    """
    # --- 0. Coerce and validate counts as integers -------------------------
    try:
        s_float = float(successes)
        t_float = float(trials)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"successes={successes!r} or trials={trials!r} cannot be interpreted as numeric: {e}"
        ) from e

    if not t_float.is_integer() or not s_float.is_integer():
        raise ValueError(
            f"successes and trials must be integer counts; "
            f"got successes={successes!r}, trials={trials!r}"
        )

    successes_i = int(s_float)
    trials_i = int(t_float)

    if trials_i <= 0:
        raise ValueError("trials must be a positive integer")
    if not (0 <= successes_i <= trials_i):
        raise ValueError(f"successes {successes_i} must be in [0, {trials_i}]")

    # --- 1. Validate alpha --------------------------------------------------
    try:
        alpha_f = float(alpha)
    except (TypeError, ValueError) as e:
        raise ValueError(f"alpha={alpha!r} cannot be cast to float: {e}") from e

    if not (0.0 < alpha_f < 1.0):
        raise ValueError(f"alpha {alpha_f} must be in (0, 1)")

    # --- 2. Core Wilson computation ----------------------------------------
    p_hat = successes_i / trials_i
    n = float(trials_i)

    # Two-sided critical value z_{1 - alpha/2}
    z = robust_inverse_normal(1.0 - alpha_f / 2.0)

    z2 = z * z
    denominator = 1.0 + z2 / n
    if denominator < EPSILON:
        # Extremely degenerate edge-case; fall back to point estimate.
        return float(p_hat), float(p_hat)

    center = (p_hat + z2 / (2.0 * n)) / denominator
    variance_term = (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))

    # Numerical safety: variance_term should be >= 0, but clamp in case of
    # tiny negative values from floating point noise.
    if variance_term < 0.0:
        variance_term = 0.0

    half_width = (z / denominator) * math.sqrt(variance_term)

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)

    # Final sanity: enforce lower <= upper within tolerance.
    if lower > upper + MATHEMATICAL_TOLERANCE:
        raise NumericalInstabilityError(
            f"Wilson interval numerically inverted: lower={lower:.12f}, upper={upper:.12f}"
        )

    return float(lower), float(upper)


def propagate_marginal_uncertainty_to_composed_bounds(
    tp_counts: Sequence[int],
    fn_counts: Sequence[int],
    fp_counts: Sequence[int],
    tn_counts: Sequence[int],
    composition_type: str,
    alpha: float = 0.05,
    policy_threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Propagate per-rail Wilson intervals into worst-/best-case composed J bounds.

    Parameters
    ----------
    tp_counts, fn_counts, fp_counts, tn_counts:
        Per-rail counts for world-1 (tp, fn) and world-0 (fp, tn).
    composition_type:
        "serial_or" or "parallel_and".
    alpha:
        Two-sided significance level for Wilson intervals.
    policy_threshold:
        Optional pass/fail threshold for certified_j_lower.
    """
    if composition_type not in {"serial_or", "parallel_and"}:
        raise ValueError("composition_type must be 'serial_or' or 'parallel_and'")

    counts = (tp_counts, fn_counts, fp_counts, tn_counts)
    if len({len(seq) for seq in counts}) != 1:
        raise ValueError("All count sequences must have the same length")

    k_rails = len(tp_counts)
    if k_rails == 0:
        raise ValueError("Count sequences must be non-empty")

    miss_intervals: List[Bounds] = []
    fpr_intervals: List[Bounds] = []
    for tp, fn, fp, tn in zip(tp_counts, fn_counts, fp_counts, tn_counts, strict=False):
        if tp < 0 or fn < 0 or fp < 0 or tn < 0:
            raise ValueError("Counts must be non-negative")
        n_attack = tp + fn
        n_benign = fp + tn
        if n_attack <= 0 or n_benign <= 0:
            raise ValueError("Each rail must have positive attack and benign sample counts")

        miss_ci = wilson_score_interval(successes=fn, trials=n_attack, alpha=alpha)
        fpr_ci = wilson_score_interval(successes=fp, trials=n_benign, alpha=alpha)
        miss_intervals.append(miss_ci)
        fpr_intervals.append(fpr_ci)

    miss_lower = [ci[0] for ci in miss_intervals]
    miss_upper = [ci[1] for ci in miss_intervals]
    fpr_lower = [ci[0] for ci in fpr_intervals]
    fpr_upper = [ci[1] for ci in fpr_intervals]

    if composition_type == "serial_or":
        miss_bounds_worst = intersection_bounds(miss_upper)
        miss_bounds_best = intersection_bounds(miss_lower)
        alarm_bounds_worst = union_bounds(fpr_upper)
        alarm_bounds_best = union_bounds(fpr_lower)
    else:
        miss_bounds_worst = union_bounds(miss_upper)
        miss_bounds_best = union_bounds(miss_lower)
        alarm_bounds_worst = intersection_bounds(fpr_upper)
        alarm_bounds_best = intersection_bounds(fpr_lower)

    tpr_lower = 1.0 - miss_bounds_worst.upper
    fpr_upper = alarm_bounds_worst.upper
    certified_j_lower = tpr_lower - fpr_upper

    tpr_upper = 1.0 - miss_bounds_best.lower
    fpr_lower = alarm_bounds_best.lower
    certified_j_upper = tpr_upper - fpr_lower

    result: Dict[str, Any] = {
        "composition_type": composition_type,
        "k_rails": k_rails,
        "alpha": alpha,
        "miss_intervals": miss_intervals,
        "fpr_intervals": fpr_intervals,
        "certified_j_lower": certified_j_lower,
        "certified_j_upper": certified_j_upper,
        "assumptions": "Unknown dependence bounded by FH envelopes; Wilson CIs per rail.",
    }

    if policy_threshold is not None:
        result["policy_threshold"] = policy_threshold
        result["pass"] = certified_j_lower >= policy_threshold

    return result


# ---------------------------------------------------------------------
# Stratified bootstrap for J-statistic
# ---------------------------------------------------------------------


def stratified_bootstrap_j_statistic(
    results_world_0: Sequence[Any],
    results_world_1: Sequence[Any],
    n_bootstrap: int = 2000,
    random_seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[List[float], Bounds]:
    """
    Stratified bootstrap for the Youden J-statistic, preserving world balance.

    Semantics
    ---------
    We maintain the original class balance by resampling *within* each world:

        - world 0 (benign): results_world_0
        - world 1 (adversarial/protected): results_world_1

    For each bootstrap iteration b:

        1. Sample n0 results with replacement from world 0.
        2. Sample n1 results with replacement from world 1.
        3. Concatenate and feed to `compute_j_statistic`, which must return
           a triple (J, TPR, FPR) or at least have its first element be J.
        4. Store J_b.

    The final CI is a percentile bootstrap interval on the J_b samples.

    Parameters
    ----------
    results_world_0, results_world_1 :
        Sequences of AttackResult-like objects for worlds 0 and 1.
        The only hard requirement is that the global `compute_j_statistic`
        can consume their concatenation.
    n_bootstrap : int
        Number of bootstrap samples (must be a positive integer).
    random_seed : int
        Deterministic seed for the Python RNG used for resampling.
    alpha : float
        Significance level (0 < alpha < 1) for a two-sided bootstrap CI.
        For example, alpha=0.05 => 95% percentile CI.

    Returns
    -------
    (bootstrap_j_stats, (ci_lower, ci_upper))
        - bootstrap_j_stats : list of bootstrap J estimates (possibly shorter
          than n_bootstrap if some iterations fail).
        - (ci_lower, ci_upper) : percentile CI for J, clipped to [-1, 1].

    Raises
    ------
    ValueError
        If either world has no samples, n_bootstrap/alpha are invalid, or
        `compute_j_statistic` is not callable.
    UserWarning
        If a large fraction of bootstrap iterations fail or sample sizes are small.
    """
    # --- 0. Basic input validation -----------------------------------------
    if not results_world_0 or not results_world_1:
        raise ValueError("Both worlds must have non-empty results")

    try:
        n_boots_float = float(n_bootstrap)
    except (TypeError, ValueError) as e:
        raise ValueError(f"n_bootstrap={n_bootstrap!r} is not numeric: {e}") from e

    if not n_boots_float.is_integer():
        raise ValueError(f"n_bootstrap must be an integer; got {n_bootstrap!r}")
    n_bootstrap_i = int(n_boots_float)
    if n_bootstrap_i <= 0:
        raise ValueError("n_bootstrap must be a positive integer")

    try:
        alpha_f = float(alpha)
    except (TypeError, ValueError) as e:
        raise ValueError(f"alpha={alpha!r} cannot be cast to float: {e}") from e
    if not (0.0 < alpha_f < 1.0):
        raise ValueError(f"alpha {alpha_f} must be in (0, 1)")

    if n_bootstrap_i > MAX_BOOTSTRAP_ITERATIONS:
        warnings.warn(
            f"n_bootstrap={n_bootstrap_i} exceeds MAX_BOOTSTRAP_ITERATIONS={MAX_BOOTSTRAP_ITERATIONS}; "
            f"this may be slow.",
            UserWarning,
            stacklevel=2,
        )

    if len(results_world_0) < MIN_SAMPLE_SIZE_WARNING:
        warnings.warn(
            f"Small sample size for world 0: n={len(results_world_0)}",
            UserWarning,
            stacklevel=2,
        )
    if len(results_world_1) < MIN_SAMPLE_SIZE_WARNING:
        warnings.warn(
            f"Small sample size for world 1: n={len(results_world_1)}",
            UserWarning,
            stacklevel=2,
        )

    # Ensure we actually have a usable compute_j_statistic.
    if not callable(compute_j_statistic):
        raise ValueError(
            "compute_j_statistic is not callable; ensure it is imported "
            "correctly from core.stats or provide a valid implementation."
        )

    # --- 1. Core bootstrap loop --------------------------------------------
    rng = random.Random(random_seed)
    n0, n1 = len(results_world_0), len(results_world_1)
    bootstrap_j_stats: List[float] = []

    for i in range(n_bootstrap_i):
        # Sample with replacement within each world
        try:
            bootstrap_w0 = [results_world_0[rng.randint(0, n0 - 1)] for _ in range(n0)]
            bootstrap_w1 = [results_world_1[rng.randint(0, n1 - 1)] for _ in range(n1)]
        except ValueError as exc:  # pragma: no cover - extremely defensive
            warnings.warn(
                f"Bootstrap iteration {i} failed during resampling: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue

        combined = bootstrap_w0 + bootstrap_w1

        try:
            # Expect compute_j_statistic to return (J, TPR, FPR) or similar.
            j_boot, *_ = compute_j_statistic(combined)
            j_val = float(j_boot)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Bootstrap iteration {i} failed in compute_j_statistic: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Validate J value: finite and within [-1, 1] up to a small tolerance.
        if math.isnan(j_val) or math.isinf(j_val):
            warnings.warn(
                f"Bootstrap iteration {i} produced non-finite J={j_val!r}; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if j_val < -1.0 - MATHEMATICAL_TOLERANCE or j_val > 1.0 + MATHEMATICAL_TOLERANCE:
            warnings.warn(
                f"Bootstrap iteration {i} produced out-of-range J={j_val:.6f}; "
                "clipping and including.",
                UserWarning,
                stacklevel=2,
            )
        # Clip into admissible range before storing.
        j_val = max(-1.0, min(1.0, j_val))
        bootstrap_j_stats.append(j_val)

    # --- 2. Post-processing and CI computation -----------------------------
    n_success = len(bootstrap_j_stats)
    if n_success == 0:
        warnings.warn(
            "All bootstrap iterations failed; returning empty samples and "
            "degenerate CI (0.0, 0.0).",
            UserWarning,
            stacklevel=2,
        )
        return [], (0.0, 0.0)

    if n_success < 0.9 * n_bootstrap_i:
        warnings.warn(
            f"Only {n_success}/{n_bootstrap_i} bootstrap samples succeeded "
            "(< 90%); CI may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Convert to array and drop any remaining non-finite values (should be none).
    j_array = np.asarray(bootstrap_j_stats, dtype=float)
    finite_mask = np.isfinite(j_array)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        warnings.warn(
            f"{n_bad} non-finite J values in bootstrap_j_stats; removing before CI.",
            UserWarning,
            stacklevel=2,
        )
        j_array = j_array[finite_mask]

    if j_array.size == 0:
        warnings.warn(
            "No finite J values after filtering; returning degenerate CI (0.0, 0.0).",
            UserWarning,
            stacklevel=2,
        )
        return [], (0.0, 0.0)

    # Percentile bootstrap CI
    lower_q = 100.0 * (alpha_f / 2.0)
    upper_q = 100.0 * (1.0 - alpha_f / 2.0)

    ci_lower = float(np.percentile(j_array, lower_q))
    ci_upper = float(np.percentile(j_array, upper_q))

    # Clip CI into [-1, 1] and sanity-check ordering.
    ci_lower = max(-1.0, min(1.0, ci_lower))
    ci_upper = max(-1.0, min(1.0, ci_upper))

    if ci_lower > ci_upper + MATHEMATICAL_TOLERANCE:
        raise NumericalInstabilityError(
            f"Bootstrap CI inverted: lower={ci_lower:.12f}, upper={ci_upper:.12f}"
        )

    return bootstrap_j_stats, (ci_lower, ci_upper)


# ---------------------------------------------------------------------
# Integration helpers with AttackResult
# ---------------------------------------------------------------------


def extract_rates_from_attack_results(
    results: Sequence[Any],
) -> Tuple[List[Probability], List[Probability]]:
    """
    Extract a single pair of (miss_rate, false_positive_rate) from AttackResult-like
    objects, aggregating *all* results into one effective rail.

    This is a deliberately conservative bridge from empirical data to a
    single-rail FH analysis: you get one miss rate and one FPR for the whole
    system.

    Interpretation of `world_bit`
    -----------------------------
    We use the following mapping:

    - world_bit == 0, or WorldBit.BASELINE  → benign world (world 0)
    - world_bit == 1, or WorldBit.PROTECTED → adversarial/attack world (world 1)

    If `world_bit` is missing (None), we emit a warning and *default to world 0*
    (benign). If `world_bit` has any other value, we emit a warning and ignore
    that sample.

    Interpretation of `success`
    ---------------------------
    For this helper we assume:

    - world 0 (benign):
        success == True  => false positive (rail fired on benign input)
    - world 1 (adversarial):
        success == True  => true positive (rail fired on attack input)

    If your `AttackResult.success` uses opposite semantics (e.g., success=True
    means "attack got through"), you MUST invert before calling this helper.

    We require that every *kept* sample has a non-None `success` attribute.
    Missing or None `success` is treated as a hard error rather than silently
    assuming False.

    Returns
    -------
    (miss_rates, fpr_rates)
        Each is a list with a single element (float in [0, 1]), suitable for
        feeding into composition bounds when you treat your current system as
        one effective rail.

    Raises
    ------
    ValueError
        If `results` is empty, if no usable samples remain after filtering, or
        if any kept sample is missing a valid `success` attribute.
    """
    if not results:
        raise ValueError("extract_rates_from_attack_results: results list is empty")

    world_0_results: List[Any] = []
    world_1_results: List[Any] = []

    warned_unknown_world = False

    for idx, r in enumerate(results):
        wb = getattr(r, "world_bit", None)

        # Missing world_bit → default to baseline (benign) but warn.
        if wb is None:
            warnings.warn(
                f"Result at index {idx} missing world_bit; defaulting to world_bit=0 (benign).",
                UserWarning,
                stacklevel=2,
            )
            wb = 0

        # Map onto world 0 / world 1
        if wb == 0 or wb == getattr(WorldBit, "BASELINE", 0):
            world_0_results.append(r)
        elif wb == 1 or wb == getattr(WorldBit, "PROTECTED", 1):
            world_1_results.append(r)
        else:
            if not warned_unknown_world:
                warnings.warn(
                    "Encountered result(s) with unknown world_bit values; "
                    "such samples will be ignored. First offending value: "
                    f"{wb!r}",
                    UserWarning,
                    stacklevel=2,
                )
                warned_unknown_world = True
            # Skip this sample
            continue

    n0 = len(world_0_results)
    n1 = len(world_1_results)

    if n0 == 0 or n1 == 0:
        raise ValueError(
            "extract_rates_from_attack_results requires usable samples from both "
            f"world 0 and world 1 after filtering unknown world_bits. "
            f"Got n0={n0}, n1={n1}."
        )

    # Helper for strict success extraction
    def _extract_success_flag(r: Any, idx: int, world_label: str) -> bool:
        if not hasattr(r, "success"):
            raise ValueError(
                f"Result in {world_label} at index {idx} missing 'success' attribute; "
                "cannot infer detection outcome. Either fix the data or adapt "
                "extract_rates_from_attack_results to your schema."
            )
        s = r.success
        if s is None:
            raise ValueError(
                f"Result in {world_label} at index {idx} has success=None; "
                "this is ambiguous and not allowed."
            )
        return bool(s)

    # False positives in world 0 (benign)
    fp_count = 0
    for idx, r in enumerate(world_0_results):
        if _extract_success_flag(r, idx, "world_0"):
            fp_count += 1
    fpr = fp_count / n0

    # True positives in world 1 (adversarial)
    tp_count = 0
    for idx, r in enumerate(world_1_results):
        if _extract_success_flag(r, idx, "world_1"):
            tp_count += 1
    tpr = tp_count / n1
    miss_rate = 1.0 - tpr

    # Numerical safety (should already be in [0, 1], but we clip defensively)
    fpr = max(0.0, min(1.0, float(fpr)))
    miss_rate = max(0.0, min(1.0, float(miss_rate)))

    return [miss_rate], [fpr]


def validate_fh_bounds_against_empirical(
    bounds: ComposedJBounds,
    observed_j: float,
    confidence_interval: Bounds | None = None,
) -> Dict[str, Union[bool, float, str, Bounds]]:
    """
    Compare FH J-bounds against an empirical J estimate (with optional CI).

    This is a diagnostic helper, not a hypothesis test. It answers:
    - Does observed J lie inside the FH envelope (up to numerical tolerance)?
    - Where inside the envelope does it sit (relative position in [0,1])?
    - Are the FH bounds tight/moderate/loose?
    - If an empirical CI is provided, how much does it overlap the FH envelope?

    Parameters
    ----------
    bounds : ComposedJBounds
        Theoretical FH bounds on J (with j_lower, j_upper, width).
    observed_j : float
        Empirical point estimate of J (should lie in [-1, 1]).
    confidence_interval : Optional[Bounds]
        Optional empirical CI on J as (ci_lower, ci_upper). If provided,
        overlap and consistency diagnostics are added.

    Returns
    -------
    report : dict
        A dictionary with keys including:
        - 'bounds_contain_observation' : bool
        - 'observed_j' : float
        - 'theoretical_bounds' : (j_lower, j_upper)
        - 'bound_width' : float
        - 'relative_position' : float in [0,1]
        - 'position_interpretation' : str
        - 'bound_quality' : {'tight', 'moderate', 'loose'}
        - 'timestamp' : str (ISO-like from numpy datetime64)
        And, if CI is provided:
        - 'ci_bounds_overlap' : bool
        - 'overlap_width' : float
        - 'overlap_fraction_of_ci' : float
        - 'overlap_fraction_of_bounds' : float
        - 'statistical_consistency' : {'good', 'moderate', 'poor'}
        - optionally 'discrepancy' : str when overlap is empty.
    """
    # --- Basic validation ---
    j_obs = float(observed_j)
    if math.isnan(j_obs) or math.isinf(j_obs):
        raise ValueError(f"observed_j must be finite; got {j_obs!r}")
    if not (-1.0 - MATHEMATICAL_TOLERANCE <= j_obs <= 1.0 + MATHEMATICAL_TOLERANCE):
        raise ValueError(
            f"observed_j={j_obs} is outside the admissible range [-1, 1] "
            "beyond numerical tolerance."
        )

    j_lower = float(bounds.j_lower)
    j_upper = float(bounds.j_upper)
    width = float(bounds.width)

    if j_lower > j_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            f"ComposedJBounds has inverted J-interval: "
            f"j_lower={j_lower:.12f} > j_upper={j_upper:.12f}"
        )

    # --- Core report scaffold ---
    # Use tolerance-aware containment.
    contains = (j_lower - MATHEMATICAL_TOLERANCE) <= j_obs <= (j_upper + MATHEMATICAL_TOLERANCE)

    report: Dict[str, Union[bool, float, str, Bounds]] = {
        "bounds_contain_observation": contains,
        "observed_j": j_obs,
        "theoretical_bounds": (j_lower, j_upper),
        "bound_width": width,
        "timestamp": str(np.datetime64("now")),
    }

    # --- Relative position of observation within FH envelope ---
    if width > MATHEMATICAL_TOLERANCE:
        # Normalize and clip into [0, 1] defensively.
        rel = (j_obs - j_lower) / width
        rel = max(0.0, min(1.0, float(rel)))
        report["relative_position"] = rel

        if rel < 0.1:
            report["position_interpretation"] = "near_lower_bound"
        elif rel > 0.9:
            report["position_interpretation"] = "near_upper_bound"
        else:
            report["position_interpretation"] = "central"
    else:
        # Degenerate or near-degenerate FH interval.
        report["relative_position"] = 0.5
        report["position_interpretation"] = "degenerate_bounds"

    # --- Qualitative bound tightness ---
    if width < 0.05:
        report["bound_quality"] = "tight"
    elif width < 0.2:
        report["bound_quality"] = "moderate"
    else:
        report["bound_quality"] = "loose"

    # --- Optional CI comparison ---
    if confidence_interval is not None:
        ci_lower_raw, ci_upper_raw = confidence_interval
        ci_lower = float(ci_lower_raw)
        ci_upper = float(ci_upper_raw)

        if (
            math.isnan(ci_lower)
            or math.isnan(ci_upper)
            or math.isinf(ci_lower)
            or math.isinf(ci_upper)
        ):
            raise ValueError(
                f"confidence_interval must contain finite values; got ({ci_lower_raw!r}, {ci_upper_raw!r})"
            )

        # Allow slightly inverted CI due to upstream quirks, but fix and warn.
        if ci_lower > ci_upper:
            warnings.warn(
                f"validate_fh_bounds_against_empirical: confidence_interval is inverted "
                f"({ci_lower:.6f}, {ci_upper:.6f}); swapping endpoints.",
                UserWarning,
                stacklevel=2,
            )
            ci_lower, ci_upper = ci_upper, ci_lower

        # Overlap computations with tolerance.
        overlap_lower = max(j_lower, ci_lower)
        overlap_upper = min(j_upper, ci_upper)
        has_overlap = overlap_lower <= overlap_upper + MATHEMATICAL_TOLERANCE

        if has_overlap:
            overlap_width = max(0.0, overlap_upper - overlap_lower)
            ci_width = max(0.0, ci_upper - ci_lower)
            bounds_width = max(0.0, width)

            overlap_fraction_ci = (
                overlap_width / ci_width if ci_width > MATHEMATICAL_TOLERANCE else 0.0
            )
            overlap_fraction_bounds = (
                overlap_width / bounds_width if bounds_width > MATHEMATICAL_TOLERANCE else 0.0
            )

            if overlap_fraction_ci > 0.8:
                consistency = "good"
            elif overlap_fraction_ci > 0.3:
                consistency = "moderate"
            else:
                consistency = "poor"

            report.update(
                {
                    "ci_bounds_overlap": True,
                    "overlap_width": float(overlap_width),
                    "overlap_fraction_of_ci": float(overlap_fraction_ci),
                    "overlap_fraction_of_bounds": float(overlap_fraction_bounds),
                    "statistical_consistency": consistency,
                }
            )
        else:
            report.update(
                {
                    "ci_bounds_overlap": False,
                    "statistical_consistency": "poor",
                    "discrepancy": (
                        "Empirical CI and FH J-bounds do not overlap "
                        "(beyond numerical tolerance) - potential model "
                        "violation, mis-specification, or under-estimated "
                        "uncertainty."
                    ),
                }
            )

    return report


# ---------------------------------------------------------------------
# Advanced analyses
# ---------------------------------------------------------------------


def sensitivity_analysis_fh_bounds(
    nominal_miss_rates: Sequence[Probability],
    nominal_fpr_rates: Sequence[Probability],
    perturbation_size: float = 0.01,
    n_perturbations: int = 100,
    random_seed: int = 42,
) -> Dict[str, Union[float, str, int]]:
    """
    Sensitivity analysis: perturb rates slightly and observe FH J-bound variability.

    Semantics
    ---------
    We take the nominal per-rail miss and false positive rates, compute a baseline
    serial-OR FH J-interval, then repeatedly:
      - add uniform noise in [-perturbation_size, +perturbation_size] to each rate,
      - clamp back into [0, 1],
      - recompute the serial-OR FH J-bounds.

    We then examine how the J-interval width and endpoints fluctuate under these
    small perturbations as a measure of *local sensitivity* of the FH envelope.

    Parameters
    ----------
    nominal_miss_rates, nominal_fpr_rates :
        Baseline miss/FPR rates per rail (probabilities in [0, 1]).
    perturbation_size : float
        Uniform perturbation magnitude in [-perturbation_size, +perturbation_size].
        Must lie in (0, 0.1] to keep the “local perturbation” interpretation.
    n_perturbations : int
        Number of perturbation samples to draw.
    random_seed : int
        Seed for the RNG used to generate perturbations (for reproducibility).

    Returns
    -------
    dict
        Keys include:
        - 'baseline_width' : float
        - 'width_std' : float
        - 'width_sensitivity' : float (std / baseline_width or inf)
        - 'j_lower_std' : float
        - 'j_upper_std' : float
        - 'perturbation_size_tested' : float
        - 'n_successful_perturbations' : int
        - 'n_failed_perturbations' : int
        - 'sensitivity_interpretation' : {'low', 'moderate', 'high'}
        - 'status' : {'ok', 'failed'}
        - 'reason' : str (only when status != 'ok')
    """
    # --- Input validation ----------------------------------------------------
    validate_probability_vector(nominal_miss_rates, "nominal_miss_rates")
    validate_probability_vector(nominal_fpr_rates, "nominal_fpr_rates")

    if len(nominal_miss_rates) != len(nominal_fpr_rates):
        raise ValueError("nominal_miss_rates and nominal_fpr_rates must have the same length")

    if not (0.0 < perturbation_size <= 0.1):
        raise ValueError("perturbation_size should be in (0, 0.1]")

    if n_perturbations <= 0:
        raise ValueError("n_perturbations must be positive")

    # Reuse bootstrap-scale guardrails if available.
    try:  # pragma: no cover - constants may be defined elsewhere in module
        max_iters = MAX_BOOTSTRAP_ITERATIONS  # type: ignore[name-defined]
    except NameError:  # fallback if not defined in this module
        max_iters = 10_000

    if n_perturbations > max_iters:
        warnings.warn(
            f"n_perturbations={n_perturbations} is large; this may be slow.",
            UserWarning,
            stacklevel=2,
        )

    # --- Baseline FH bounds --------------------------------------------------
    baseline = serial_or_composition_bounds(nominal_miss_rates, nominal_fpr_rates)
    baseline_width = float(baseline.width)

    # --- Perturbation loop ---------------------------------------------------
    width_variations: List[float] = []
    j_lower_variations: List[float] = []
    j_upper_variations: List[float] = []

    rng = random.Random(random_seed)
    n_success = 0
    n_fail = 0

    for _ in range(n_perturbations):
        perturbed_miss: List[float] = []
        perturbed_fpr: List[float] = []

        # Perturb miss rates
        for m in nominal_miss_rates:
            delta = rng.uniform(-perturbation_size, perturbation_size)
            perturbed_miss.append(max(0.0, min(1.0, float(m) + delta)))

        # Perturb FPR rates
        for f in nominal_fpr_rates:
            delta = rng.uniform(-perturbation_size, perturbation_size)
            perturbed_fpr.append(max(0.0, min(1.0, float(f) + delta)))

        try:
            perturbed_bounds = serial_or_composition_bounds(perturbed_miss, perturbed_fpr)
        except Exception as exc:  # pragma: no cover - defensive
            n_fail += 1
            warnings.warn(
                f"sensitivity_analysis_fh_bounds: perturbation failed with {exc!r}",
                UserWarning,
                stacklevel=2,
            )
            continue

        n_success += 1
        width_variations.append(float(perturbed_bounds.width))
        j_lower_variations.append(float(perturbed_bounds.j_lower))
        j_upper_variations.append(float(perturbed_bounds.j_upper))

    if n_success == 0:
        return {
            "status": "failed",
            "reason": "all perturbations failed",
            "baseline_width": float(baseline_width),
            "perturbation_size_tested": float(perturbation_size),
            "n_successful_perturbations": 0,
            "n_failed_perturbations": int(n_fail),
        }

    # Warn if a substantial fraction of perturbations failed.
    if n_success < 0.9 * n_perturbations:
        warnings.warn(
            f"Only {n_success}/{n_perturbations} perturbations succeeded "
            "(>10% failures). Sensitivity summary may be unstable.",
            UserWarning,
            stacklevel=2,
        )

    # --- Aggregate variation statistics --------------------------------------
    width_std = float(np.std(width_variations))
    j_lower_std = float(np.std(j_lower_variations))
    j_upper_std = float(np.std(j_upper_variations))

    if baseline_width > MATHEMATICAL_TOLERANCE:
        width_sensitivity = width_std / baseline_width
    else:
        # Degenerate baseline bounds: any variation is "infinite" relative change.
        width_sensitivity = float("inf")

    # Qualitative interpretation of sensitivity on width
    if not math.isfinite(width_sensitivity):
        interpretation = "high"
    elif width_sensitivity < 0.1:
        interpretation = "low"
    elif width_sensitivity < 0.5:
        interpretation = "moderate"
    else:
        interpretation = "high"

    return {
        "status": "ok",
        "baseline_width": float(baseline_width),
        "width_std": width_std,
        "width_sensitivity": float(width_sensitivity),
        "j_lower_std": j_lower_std,
        "j_upper_std": j_upper_std,
        "perturbation_size_tested": float(perturbation_size),
        "n_successful_perturbations": int(n_success),
        "n_failed_perturbations": int(n_fail),
        "sensitivity_interpretation": interpretation,
    }


# ---------------------------------------------------------------------
# Internal mathematical self-tests
# ---------------------------------------------------------------------


def verify_fh_bound_properties() -> Dict[str, bool]:
    """
    Internal sanity checks for FH bounds and related functionality.

    Returns
    -------
    Dict[str, bool]
        Mapping from test label to boolean. Intended for developer /
        research sanity, not for end-user use.

    Notes
    -----
    This is NOT a replacement for a full unit test suite, but it should
    catch gross inconsistencies in the mathematical structure:
    - FH intersection/union formulas
    - Monotonicity properties
    - J-bounds coherence
    - Independence calculations
    - CII basic sanity
    - Wilson CI basic sanity
    - FHBounds invariant enforcement
    """
    tests_passed: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Test 1: Single-event bounds are exact for both intersection & union
    # ------------------------------------------------------------------
    try:
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            int_b = intersection_bounds([p])
            uni_b = union_bounds([p])
            assert abs(int_b.lower - p) < MATHEMATICAL_TOLERANCE
            assert abs(int_b.upper - p) < MATHEMATICAL_TOLERANCE
            assert abs(uni_b.lower - p) < MATHEMATICAL_TOLERANCE
            assert abs(uni_b.upper - p) < MATHEMATICAL_TOLERANCE
        tests_passed["single_event_exactness"] = True
    except Exception:
        tests_passed["single_event_exactness"] = False

    # ------------------------------------------------------------------
    # Test 2: Two-event intersection and union sharpness
    # ------------------------------------------------------------------
    try:
        p1, p2 = 0.9, 0.8
        int_b = intersection_bounds([p1, p2])
        uni_b = union_bounds([p1, p2])

        expected_int_lower = max(0.0, p1 + p2 - 1.0)
        expected_int_upper = min(p1, p2)
        expected_uni_lower = max(p1, p2)
        expected_uni_upper = min(1.0, p1 + p2)

        assert abs(int_b.lower - expected_int_lower) < MATHEMATICAL_TOLERANCE
        assert abs(int_b.upper - expected_int_upper) < MATHEMATICAL_TOLERANCE
        assert abs(uni_b.lower - expected_uni_lower) < MATHEMATICAL_TOLERANCE
        assert abs(uni_b.upper - expected_uni_upper) < MATHEMATICAL_TOLERANCE

        tests_passed["two_event_sharpness"] = True
    except Exception:
        tests_passed["two_event_sharpness"] = False

    # ------------------------------------------------------------------
    # Test 3: Monotonicity - adding events
    #   - intersection upper bound should not increase
    #   - union lower bound should not decrease
    # ------------------------------------------------------------------
    try:
        # Intersection monotonicity
        b2_int = intersection_bounds([0.8, 0.7])
        b3_int = intersection_bounds([0.8, 0.7, 0.6])
        assert b3_int.upper <= b2_int.upper + MATHEMATICAL_TOLERANCE

        # Union monotonicity
        b2_uni = union_bounds([0.3, 0.4])
        b3_uni = union_bounds([0.3, 0.4, 0.5])
        assert b3_uni.lower + MATHEMATICAL_TOLERANCE >= b2_uni.lower

        tests_passed["monotonicity"] = True
    except Exception:
        tests_passed["monotonicity"] = False

    # ------------------------------------------------------------------
    # Test 4: Intersection vs union consistency
    # For any joint law, P(n A_i) <= P(U A_i)
    # => FH bounds should respect this ordering.
    # ------------------------------------------------------------------
    try:
        marginals = [0.3, 0.4, 0.5]
        int_b = intersection_bounds(marginals)
        uni_b = union_bounds(marginals)

        assert uni_b.lower >= int_b.lower - MATHEMATICAL_TOLERANCE
        assert uni_b.upper >= int_b.upper - MATHEMATICAL_TOLERANCE

        tests_passed["intersection_union_consistency"] = True
    except Exception:
        tests_passed["intersection_union_consistency"] = False

    # ------------------------------------------------------------------
    # Test 5: FH vs independence - intersection/union probabilities
    # Independence law should lie within FH envelopes.
    # ------------------------------------------------------------------
    try:
        p = [0.2, 0.5, 0.7]
        int_b = intersection_bounds(p)
        uni_b = union_bounds(p)

        # Under independence:
        p_int_indep = 1.0
        for pi in p:
            p_int_indep *= float(pi)
        p_uni_indep = 1.0
        for pi in p:
            p_uni_indep *= 1.0 - float(pi)
        p_uni_indep = 1.0 - p_uni_indep

        assert (
            int_b.lower - MATHEMATICAL_TOLERANCE
            <= p_int_indep
            <= int_b.upper + MATHEMATICAL_TOLERANCE
        )
        assert (
            uni_b.lower - MATHEMATICAL_TOLERANCE
            <= p_uni_indep
            <= uni_b.upper + MATHEMATICAL_TOLERANCE
        )

        tests_passed["fh_vs_independence_envelope"] = True
    except Exception:
        tests_passed["fh_vs_independence_envelope"] = False

    # ------------------------------------------------------------------
    # Test 6: J-statistic bounds consistency in serial OR
    # ------------------------------------------------------------------
    try:
        miss_rates = [0.2, 0.3]
        fpr_rates = [0.05, 0.1]
        cb = serial_or_composition_bounds(miss_rates, fpr_rates)

        expected_j_lower = cb.tpr_bounds.lower - cb.fpr_bounds.upper
        expected_j_upper = cb.tpr_bounds.upper - cb.fpr_bounds.lower

        assert abs(cb.j_lower - expected_j_lower) < MATHEMATICAL_TOLERANCE
        assert abs(cb.j_upper - expected_j_upper) < MATHEMATICAL_TOLERANCE

        tests_passed["j_statistic_serial_or_consistency"] = True
    except Exception:
        tests_passed["j_statistic_serial_or_consistency"] = False

    # ------------------------------------------------------------------
    # Test 7: J-statistic bounds consistency in parallel AND
    # ------------------------------------------------------------------
    try:
        miss_rates = [0.1, 0.15]
        fpr_rates = [0.02, 0.03]
        cb = parallel_and_composition_bounds(miss_rates, fpr_rates)

        expected_j_lower = cb.tpr_bounds.lower - cb.fpr_bounds.upper
        expected_j_upper = cb.tpr_bounds.upper - cb.fpr_bounds.lower

        assert abs(cb.j_lower - expected_j_lower) < MATHEMATICAL_TOLERANCE
        assert abs(cb.j_upper - expected_j_upper) < MATHEMATICAL_TOLERANCE

        tests_passed["j_statistic_parallel_and_consistency"] = True
    except Exception:
        tests_passed["j_statistic_parallel_and_consistency"] = False

    # ------------------------------------------------------------------
    # Test 8: Independence calculations - finite and in [-1, 1]
    # ------------------------------------------------------------------
    try:
        tprs = [0.7, 0.8]
        fprs = [0.05, 0.1]

        j_indep_or = independence_serial_or_j(tprs, fprs)
        j_indep_and = independence_parallel_and_j(tprs, fprs)

        assert math.isfinite(j_indep_or) and -1.0 <= j_indep_or <= 1.0
        assert math.isfinite(j_indep_and) and -1.0 <= j_indep_and <= 1.0

        tests_passed["independence_calculation_range"] = True
    except Exception:
        tests_passed["independence_calculation_range"] = False

    # ------------------------------------------------------------------
    # Test 9: Independence J lies within FH J-bounds (serial OR)
    # ------------------------------------------------------------------
    try:
        miss_rates = [0.3, 0.2]  # TPR ~ [0.7, 0.8]
        fpr_rates = [0.05, 0.1]
        cb = serial_or_composition_bounds(miss_rates, fpr_rates)

        individual_tprs = [1.0 - m for m in miss_rates]
        individual_fprs = list(fpr_rates)
        j_indep = independence_serial_or_j(individual_tprs, individual_fprs)

        assert cb.j_lower - MATHEMATICAL_TOLERANCE <= j_indep <= cb.j_upper + MATHEMATICAL_TOLERANCE

        tests_passed["independence_within_fh_j_bounds_serial_or"] = True
    except Exception:
        tests_passed["independence_within_fh_j_bounds_serial_or"] = False

    # ------------------------------------------------------------------
    # Test 10: CII computation basic sanity
    # ------------------------------------------------------------------
    try:
        miss_rates = [0.3, 0.2]
        fpr_rates = [0.05, 0.1]
        cb = serial_or_composition_bounds(miss_rates, fpr_rates)
        individual_tprs = [1.0 - m for m in miss_rates]
        individual_fprs = list(fpr_rates)

        cii_result = compute_composability_interference_index(
            observed_j=0.6,
            bounds=cb,
            individual_tprs=individual_tprs,
            individual_fprs=individual_fprs,
            use_independence_baseline=True,
        )

        assert "cii" in cii_result
        cii_value = float(cii_result["cii"])
        assert math.isfinite(cii_value)
        tests_passed["cii_computation"] = True
    except Exception:
        tests_passed["cii_computation"] = False

    # ------------------------------------------------------------------
    # Test 11: Wilson CI basic sanity
    # ------------------------------------------------------------------
    try:
        ci = wilson_score_interval(successes=50, trials=100, alpha=0.05)
        assert len(ci) == 2
        assert 0.0 <= ci[0] <= ci[1] <= 1.0
        tests_passed["wilson_ci"] = True
    except Exception:
        tests_passed["wilson_ci"] = False

    # ------------------------------------------------------------------
    # Test 12: FHBounds invariants are enforced
    # ------------------------------------------------------------------
    try:
        # This should be fine
        _ = FHBounds(
            lower=0.2,
            upper=0.4,
            marginals=(0.3, 0.5),
            bound_type="test",
            k_rails=2,
        )

        # This should fail (lower > upper)
        failed = False
        try:
            _ = FHBounds(
                lower=0.8,
                upper=0.2,
                marginals=(0.5,),
                bound_type="test",
                k_rails=1,
            )
            failed = True  # if no exception, test fails
        except FHBoundViolationError:
            pass

        assert not failed
        tests_passed["fhbounds_invariant_enforcement"] = True
    except Exception:
        tests_passed["fhbounds_invariant_enforcement"] = False

    # ------------------------------------------------------------------
    # Optional: robust_inverse_normal monotonicity sanity
    # ------------------------------------------------------------------
    try:
        z1 = robust_inverse_normal(0.1)
        z2 = robust_inverse_normal(0.5)
        z3 = robust_inverse_normal(0.9)
        # Should be strictly increasing in p
        assert z1 < z2 < z3
        tests_passed["robust_inverse_normal_monotonicity"] = True
    except Exception:
        tests_passed["robust_inverse_normal_monotonicity"] = False

    # Overall summary
    try:
        overall = all(tests_passed.values())
    except Exception:
        overall = False
    tests_passed["all_passed"] = overall

    return tests_passed


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "DEFAULT_CC_ABSOLUTE_MARGIN",
    "DEFAULT_CC_RELATIVE_MARGIN",
    "DEFAULT_CC_UNCERTAINTY_MULTIPLIER",
    # Constants
    "EPSILON",
    "MATHEMATICAL_TOLERANCE",
    "CCRegimeThresholds",
    "ComposedJBounds",
    # Exceptions
    "FHBoundViolationError",
    # Data structures
    "FHBounds",
    "NumericalInstabilityError",
    "StatisticalValidationError",
    "compute_cc_bounds",
    # CII analysis
    "compute_composability_interference_index",
    # Copula visualization helpers
    "copula_cdf",
    "copula_experiment_plan",
    "copula_grid",
    # CC regime thresholds
    "default_cc_regime_thresholds",
    "derive_cc_thresholds_from_uncertainty",
    # Integration functions
    "extract_rates_from_attack_results",
    # Core mathematical functions
    "frechet_intersection_lower_bound",
    "frechet_union_lower_bound",
    "hoeffding_intersection_upper_bound",
    "hoeffding_union_upper_bound",
    "independence_parallel_and_j",
    # Independence calculations
    "independence_serial_or_j",
    "intersection_bounds",
    "parallel_and_composition_bounds",
    "propagate_marginal_uncertainty_to_composed_bounds",
    "robust_inverse_normal",
    # Advanced analysis
    "sensitivity_analysis_fh_bounds",
    # Composition analysis
    "serial_or_composition_bounds",
    "stratified_bootstrap_j_statistic",
    "union_bounds",
    "validate_fh_bounds_against_empirical",
    # Verification
    "verify_fh_bound_properties",
    # Statistical functions
    "wilson_score_interval",
]


if __name__ == "__main__":  # pragma: no cover - manual verification
    """
    Manual self-check entrypoint.

    Run:

        python -m cc_framework.fh_bounds

    (or whatever your module path is) to execute the internal mathematical
    self-tests and get a human-readable summary.
    """
    print("Running internal FH-bound property verification...\n")
    results = verify_fh_bound_properties()

    print("Verification Results")
    print("=" * 40)

    # Treat any aggregate key (like "all_passed") specially.
    aggregate_keys = {"all_passed"}
    detailed_items = [(k, v) for k, v in results.items() if k not in aggregate_keys]

    # Sort for deterministic, diff-friendly output.
    detailed_items.sort(key=lambda kv: kv[0])

    total = len(detailed_items)
    passed = sum(1 for _, ok in detailed_items if ok)

    for name, ok in detailed_items:
        status = "✓ PASSED" if ok else "✗ FAILED"
        print(f"{name:35s}: {status}")

    print("\nSummary")
    print("-" * 40)
    print(f"{passed}/{total} tests passed")

    # Prefer explicit aggregate flag if present; otherwise derive from counts.
    all_passed = results.get("all_passed", passed == total)

    if all_passed:
        print("\n🎉 ALL MATHEMATICAL PROPERTIES VERIFIED - implementation is research-ready.")
    else:
        print("\n⚠️  Some tests failed - investigate before relying on these bounds in production.")
