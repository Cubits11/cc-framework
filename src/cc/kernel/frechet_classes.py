"""Sharp Frechet-class bounds for finite binary guardrail events.

The module works on the finite sample space ``Omega = {0, 1}^n``.  A joint
law is a probability vector ``x`` over the ``2**n`` atoms.  Marginal
information and pairwise dependence information are linear moment constraints:

```
sum_w x_w = 1
sum_w w_i x_w = p_i
sum_w w_i w_j x_w = p_ij              (if a pairwise constraint is known)
x_w >= 0
```

Classical n-way Frechet-Hoeffding bounds are the special case with only the
first two kinds of constraints:

```
P(all_i A_i) in [max(0, sum_i p_i - (n - 1)), min_i p_i]
P(any_i A_i) in [max_i p_i, min(1, sum_i p_i)]
```

When side information is supplied, the sharp lower and upper bounds are the
minimum and maximum of a linear event functional over the smaller feasible
polytope.  This is the finite Bernoulli form of the Frechet-class/Ruschendorf
duality viewpoint: bounds are primal linear programs over all couplings with
the prescribed marginals and side moments, and their duals are best linear
minorants/majorants built from the known moments.

Monotonicity proof.  Let ``C_S`` be the feasible set produced by a collection
``S`` of known constraints, and let ``f(x)`` be the probability of the queried
event.  If ``T`` adds constraints to ``S``, then every law feasible for ``T`` is
also feasible for ``S``; hence ``C_T subset C_S``.  Therefore

```
inf_{x in C_T} f(x) >= inf_{x in C_S} f(x)
sup_{x in C_T} f(x) <= sup_{x in C_S} f(x)
```

provided ``C_T`` is nonempty.  Thus every added valid piece of side information
can only raise the lower bound, lower the upper bound, or leave an endpoint
unchanged.  It can never loosen the interval.  If the added constraints are
inconsistent, the Frechet class is empty and no probability law satisfies all
inputs.

For binary indicators, Spearman's rho computed from midranks and Kendall's
tie-corrected tau-b are both equal to Pearson's phi coefficient.  This module
therefore converts ``spearman_rho`` and ``kendall_tau`` side information into
the corresponding pairwise joint probability ``p_ij``:

```
p_ij = p_i p_j + rho * sqrt(p_i (1 - p_i) p_j (1 - p_j)).
```

If a caller uses a different Kendall convention, such as untied tau-a, convert
that value to ``p_ij`` first and pass ``kind="joint_probability"``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isclose, sqrt
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
EventKind: TypeAlias = Literal["and", "or"]
DependenceKind: TypeAlias = Literal[
    "joint",
    "joint_probability",
    "pearson",
    "pearson_phi",
    "spearman",
    "spearman_rho",
    "kendall",
    "kendall_tau",
    "kendall_tau_b",
]
CanonicalDependenceKind: TypeAlias = Literal["joint_probability", "rank_or_phi"]

_TOL = 1.0e-10

__all__ = [
    "DependenceKind",
    "EventKind",
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


@dataclass(frozen=True)
class PairwiseDependence:
    """Known pairwise dependence between two binary event indicators.

    ``kind`` accepts ``joint_probability`` directly or one of ``pearson_phi``,
    ``spearman_rho``, and ``kendall_tau``.  For binary indicators the latter
    three are interpreted as the same normalized pairwise association.
    """

    i: int
    j: int
    kind: DependenceKind
    value: float


@dataclass(frozen=True)
class PairwiseJointConstraint:
    """Canonical pairwise moment constraint ``P(A_i and A_j) = value``."""

    i: int
    j: int
    joint_probability: float


@dataclass(frozen=True)
class FrechetBoundResult:
    """Sharp lower and upper event-probability bounds.

    ``lower_distribution`` and ``upper_distribution`` are returned only when
    requested by ``frechet_bounds(..., return_distributions=True)`` or when the
    side-constrained LP path is used.  Their atom ordering is the one returned by
    :func:`atom_matrix`.
    """

    lower: float
    upper: float
    event: EventKind
    marginals: FloatArray
    pairwise: tuple[PairwiseJointConstraint, ...]
    lower_distribution: FloatArray | None = None
    upper_distribution: FloatArray | None = None

    @property
    def width(self) -> float:
        """Return ``upper - lower`` with tiny numerical negatives clipped."""

        return max(0.0, self.upper - self.lower)


class FrechetClassInfeasibleError(ValueError):
    """Raised when the supplied marginals and side constraints are inconsistent."""


def atom_matrix(n_events: int) -> IntArray:
    """Return the ``2**n_events`` binary atoms in little-endian bit order."""

    if not isinstance(n_events, (int, np.integer)):
        raise TypeError("n_events must be an integer.")
    n_int = int(n_events)
    if n_int < 1:
        raise ValueError("n_events must be at least 1.")
    if n_int > 24:
        raise ValueError("n_events is too large for an explicit 2**n atom matrix.")

    atom_ids = np.arange(1 << n_int, dtype=np.uint64)
    shifts = np.arange(n_int, dtype=np.uint64)
    return ((atom_ids[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.int_)


def classical_frechet_bounds(
    marginals: ArrayLike,
    *,
    event: Literal["and", "or", "AND", "OR"] = "and",
) -> tuple[float, float]:
    """Return classical n-way Frechet-Hoeffding bounds from marginals alone."""

    p = _as_probability_vector(marginals)
    event_kind = _canonical_event(event)
    if event_kind == "and":
        lower = max(0.0, float(np.sum(p)) - (p.size - 1))
        upper = float(np.min(p))
    else:
        lower = float(np.max(p))
        upper = min(1.0, float(np.sum(p)))
    return _clip01(lower), _clip01(upper)


def frechet_bounds(
    marginals: ArrayLike,
    *,
    pairwise: Iterable[PairwiseDependence] | None = None,
    event: Literal["and", "or", "AND", "OR"] = "and",
    return_distributions: bool = False,
    feasibility_tol: float = _TOL,
) -> FrechetBoundResult:
    """Compute sharp Frechet bounds for an n-way binary event.

    With no pairwise side information, the function returns the classical
    closed-form n-way bounds.  With side information, it solves the exact
    finite Bernoulli linear program over ``2**n`` atoms.
    """

    p = _as_probability_vector(marginals)
    event_kind = _canonical_event(event)
    raw_pairwise = tuple(pairwise or ())
    constraints = _canonical_pairwise_constraints(p, raw_pairwise, feasibility_tol)

    if not constraints and not return_distributions:
        lower, upper = classical_frechet_bounds(p, event=event_kind)
        return FrechetBoundResult(
            lower=lower,
            upper=upper,
            event=event_kind,
            marginals=p.copy(),
            pairwise=(),
        )

    states = atom_matrix(p.size).astype(float)
    objective = _event_vector(states, event_kind)
    a_eq, b_eq = _constraint_system(states, p, constraints)

    lower_value, lower_dist = _solve_event_lp(
        objective,
        a_eq,
        b_eq,
        maximize=False,
        feasibility_tol=feasibility_tol,
    )
    upper_value, upper_dist = _solve_event_lp(
        objective,
        a_eq,
        b_eq,
        maximize=True,
        feasibility_tol=feasibility_tol,
    )

    lower_value = _clip01(lower_value)
    upper_value = _clip01(upper_value)
    if upper_value + feasibility_tol < lower_value:
        raise FrechetClassInfeasibleError(
            f"Numerical LP returned an empty interval [{lower_value}, {upper_value}]."
        )
    lower_value, upper_value = _snap_interval(lower_value, upper_value, feasibility_tol)

    return FrechetBoundResult(
        lower=lower_value,
        upper=upper_value,
        event=event_kind,
        marginals=p.copy(),
        pairwise=constraints,
        lower_distribution=lower_dist,
        upper_distribution=upper_dist,
    )


def improved_frechet_bounds(
    marginals: ArrayLike,
    *,
    pairwise: Iterable[PairwiseDependence],
    event: Literal["and", "or", "AND", "OR"] = "and",
    return_distributions: bool = False,
    feasibility_tol: float = _TOL,
) -> FrechetBoundResult:
    """Alias emphasizing the side-information use case."""

    return frechet_bounds(
        marginals,
        pairwise=pairwise,
        event=event,
        return_distributions=return_distributions,
        feasibility_tol=feasibility_tol,
    )


def dependence_to_joint_probability(
    p_i: float,
    p_j: float,
    value: float,
    kind: DependenceKind,
    *,
    tol: float = _TOL,
) -> float:
    """Convert a pairwise dependence value to ``P(A_i and A_j)``.

    For binary indicators, ``pearson_phi``, ``spearman_rho``, and
    ``kendall_tau`` are all mapped through the normalized covariance formula.
    """

    _validate_probability("p_i", p_i, tol)
    _validate_probability("p_j", p_j, tol)
    if not np.isfinite(value):
        raise ValueError("dependence value must be finite.")

    canonical = _canonical_dependence_kind(kind)
    if canonical == "joint_probability":
        joint = float(value)
    else:
        if value < -1.0 - tol or value > 1.0 + tol:
            raise ValueError(f"{kind} must lie in [-1, 1]. Got {value}.")
        scale = _bernoulli_pair_scale(p_i, p_j)
        joint = p_i * p_j + float(value) * scale

    lower, upper = _pairwise_joint_bounds(p_i, p_j)
    if joint < lower - tol or joint > upper + tol:
        corr_bounds = pairwise_correlation_bounds(p_i, p_j) if _has_pair_variance(p_i, p_j) else None
        extra = "" if corr_bounds is None else f" admissible correlation range is {corr_bounds}."
        raise ValueError(
            f"{kind}={value} implies P(A_i and A_j)={joint}, outside "
            f"[{lower}, {upper}] for marginals ({p_i}, {p_j}).{extra}"
        )
    return float(np.clip(joint, lower, upper))


def joint_probability_to_dependence(
    p_i: float,
    p_j: float,
    joint_probability: float,
    *,
    kind: DependenceKind = "spearman_rho",
    tol: float = _TOL,
) -> float:
    """Convert ``P(A_i and A_j)`` to a binary rank/phi dependence value."""

    joint = dependence_to_joint_probability(
        p_i,
        p_j,
        joint_probability,
        "joint_probability",
        tol=tol,
    )
    canonical = _canonical_dependence_kind(kind)
    if canonical == "joint_probability":
        return joint
    scale = _bernoulli_pair_scale(p_i, p_j)
    value = (joint - p_i * p_j) / scale
    return float(np.clip(value, -1.0, 1.0))


def pairwise_correlation_bounds(p_i: float, p_j: float, *, tol: float = _TOL) -> tuple[float, float]:
    """Return admissible bounds for binary phi/rho/tau-b at fixed margins."""

    _validate_probability("p_i", p_i, tol)
    _validate_probability("p_j", p_j, tol)
    lower_joint, upper_joint = _pairwise_joint_bounds(p_i, p_j)
    scale = _bernoulli_pair_scale(p_i, p_j)
    lower = (lower_joint - p_i * p_j) / scale
    upper = (upper_joint - p_i * p_j) / scale
    return float(np.clip(lower, -1.0, 1.0)), float(np.clip(upper, -1.0, 1.0))


def distribution_moments(distribution: ArrayLike, n_events: int | None = None) -> tuple[FloatArray, FloatArray]:
    """Return marginals and pairwise joint-probability matrix for an atom pmf."""

    pmf, n_int = _as_distribution(distribution, n_events)
    states = atom_matrix(n_int).astype(float)
    marginals = states.T @ pmf
    pairwise = (states.T * pmf[None, :]) @ states
    return marginals.astype(float), pairwise.astype(float)


def event_probability(
    distribution: ArrayLike,
    *,
    event: Literal["and", "or", "AND", "OR"] = "and",
    n_events: int | None = None,
) -> float:
    """Evaluate an AND/OR event probability under an atom pmf."""

    pmf, n_int = _as_distribution(distribution, n_events)
    states = atom_matrix(n_int).astype(float)
    objective = _event_vector(states, _canonical_event(event))
    return _clip01(float(objective @ pmf))


def random_feasible_distribution(
    marginals: ArrayLike,
    *,
    pairwise: Iterable[PairwiseDependence] | None = None,
    rng: np.random.Generator | int | None = None,
    feasibility_tol: float = _TOL,
) -> FloatArray:
    """Sample an exposed feasible vertex by optimizing a random linear functional."""

    p = _as_probability_vector(marginals)
    generator = np.random.default_rng(rng)
    constraints = _canonical_pairwise_constraints(p, tuple(pairwise or ()), feasibility_tol)
    states = atom_matrix(p.size).astype(float)
    a_eq, b_eq = _constraint_system(states, p, constraints)
    direction = np.asarray(generator.normal(size=states.shape[0]), dtype=float)
    _, distribution = _solve_event_lp(
        direction,
        a_eq,
        b_eq,
        maximize=False,
        feasibility_tol=feasibility_tol,
    )
    return distribution


def sample_binary_vectors(
    distribution: ArrayLike,
    size: int,
    *,
    rng: np.random.Generator | int | None = None,
    n_events: int | None = None,
) -> IntArray:
    """Draw binary vectors from an atom pmf using the :func:`atom_matrix` ordering."""

    if not isinstance(size, (int, np.integer)) or int(size) < 0:
        raise ValueError("size must be a nonnegative integer.")
    pmf, n_int = _as_distribution(distribution, n_events)
    generator = np.random.default_rng(rng)
    indices = generator.choice(pmf.size, size=int(size), replace=True, p=pmf)
    return atom_matrix(n_int)[indices]


def _as_probability_vector(values: ArrayLike, *, tol: float = _TOL) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("marginals must be a one-dimensional probability vector.")
    if arr.size < 2:
        raise ValueError("at least two binary events are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("marginals must be finite.")
    if np.any(arr < -tol) or np.any(arr > 1.0 + tol):
        raise ValueError("marginals must lie in [0, 1].")
    clipped = np.clip(arr, 0.0, 1.0).astype(np.float64, copy=False)
    return cast(FloatArray, clipped)


def _as_distribution(distribution: ArrayLike, n_events: int | None) -> tuple[FloatArray, int]:
    pmf = np.asarray(distribution, dtype=float)
    if pmf.ndim != 1:
        raise ValueError("distribution must be a one-dimensional atom probability vector.")
    if not np.all(np.isfinite(pmf)):
        raise ValueError("distribution must be finite.")
    if np.any(pmf < -_TOL):
        raise ValueError("distribution entries must be nonnegative.")
    total = float(np.sum(pmf))
    if not isclose(total, 1.0, abs_tol=1.0e-8):
        raise ValueError(f"distribution must sum to 1. Got {total}.")
    pmf = np.clip(pmf, 0.0, 1.0)
    pmf = pmf / float(np.sum(pmf))

    if n_events is None:
        if pmf.size == 0 or pmf.size & (pmf.size - 1):
            raise ValueError("distribution length must be a positive power of two.")
        n_events = int(np.log2(pmf.size))
    if not isinstance(n_events, (int, np.integer)) or int(n_events) < 1:
        raise ValueError("n_events must be a positive integer.")
    n_int = int(n_events)
    if pmf.size != 1 << n_int:
        raise ValueError(
            f"distribution length {pmf.size} does not equal 2**n_events ({1 << n_int})."
        )
    return pmf.astype(float, copy=False), n_int


def _validate_probability(name: str, value: float, tol: float) -> None:
    if not np.isfinite(value) or value < -tol or value > 1.0 + tol:
        raise ValueError(f"{name} must be a finite probability in [0, 1]. Got {value}.")


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _canonical_event(event: Literal["and", "or", "AND", "OR"]) -> EventKind:
    lowered = str(event).lower()
    if lowered == "and":
        return "and"
    if lowered == "or":
        return "or"
    raise ValueError('event must be "and" or "or".')


def _canonical_dependence_kind(kind: DependenceKind) -> CanonicalDependenceKind:
    lowered = str(kind).lower()
    if lowered in {"joint", "joint_probability", "pij", "p_ij"}:
        return "joint_probability"
    if lowered in {
        "pearson",
        "pearson_phi",
        "phi",
        "spearman",
        "spearman_rho",
        "rho",
        "kendall",
        "kendall_tau",
        "kendall_tau_b",
        "tau",
        "tau_b",
    }:
        return "rank_or_phi"
    raise ValueError(f"Unknown pairwise dependence kind: {kind!r}.")


def _pairwise_joint_bounds(p_i: float, p_j: float) -> tuple[float, float]:
    return max(0.0, p_i + p_j - 1.0), min(p_i, p_j)


def _has_pair_variance(p_i: float, p_j: float) -> bool:
    return p_i > _TOL and p_i < 1.0 - _TOL and p_j > _TOL and p_j < 1.0 - _TOL


def _bernoulli_pair_scale(p_i: float, p_j: float) -> float:
    scale = sqrt(max(0.0, p_i * (1.0 - p_i) * p_j * (1.0 - p_j)))
    if scale <= _TOL:
        raise ValueError(
            "rank/phi dependence is undefined for degenerate Bernoulli margins; "
            'pass kind="joint_probability" instead.'
        )
    return scale


def _canonical_pairwise_constraints(
    marginals: FloatArray,
    pairwise: Sequence[PairwiseDependence],
    tol: float,
) -> tuple[PairwiseJointConstraint, ...]:
    seen: dict[tuple[int, int], float] = {}
    for dep in pairwise:
        if not isinstance(dep.i, (int, np.integer)) or not isinstance(dep.j, (int, np.integer)):
            raise TypeError("pairwise indices must be integers.")
        i = int(dep.i)
        j = int(dep.j)
        if i == j:
            raise ValueError("pairwise constraints require two distinct indices.")
        if i < 0 or j < 0 or i >= marginals.size or j >= marginals.size:
            raise ValueError(
                f"pairwise index ({i}, {j}) is outside the marginal vector of size {marginals.size}."
            )
        if j < i:
            i, j = j, i
        joint = dependence_to_joint_probability(
            float(marginals[i]),
            float(marginals[j]),
            float(dep.value),
            dep.kind,
            tol=tol,
        )
        key = (i, j)
        previous = seen.get(key)
        if previous is not None and not isclose(previous, joint, abs_tol=tol):
            raise ValueError(
                f"Duplicate pairwise constraints for ({i}, {j}) disagree: {previous} vs {joint}."
            )
        seen[key] = joint
    return tuple(
        PairwiseJointConstraint(i=i, j=j, joint_probability=joint)
        for (i, j), joint in sorted(seen.items())
    )


def _constraint_system(
    states: FloatArray,
    marginals: FloatArray,
    pairwise: Sequence[PairwiseJointConstraint],
) -> tuple[FloatArray, FloatArray]:
    rows = [np.ones(states.shape[0], dtype=float)]
    rhs = [1.0]
    for i in range(marginals.size):
        rows.append(states[:, i])
        rhs.append(float(marginals[i]))
    for constraint in pairwise:
        rows.append(states[:, constraint.i] * states[:, constraint.j])
        rhs.append(float(constraint.joint_probability))
    return np.vstack(rows).astype(float), np.asarray(rhs, dtype=float)


def _event_vector(states: FloatArray, event: EventKind) -> FloatArray:
    if event == "and":
        return np.all(states == 1.0, axis=1).astype(float)
    return np.any(states == 1.0, axis=1).astype(float)


def _solve_event_lp(
    objective: FloatArray,
    a_eq: FloatArray,
    b_eq: FloatArray,
    *,
    maximize: bool,
    feasibility_tol: float,
) -> tuple[float, FloatArray]:
    c = -objective if maximize else objective
    result = linprog(
        c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * objective.size,
        method="highs",
    )
    if not result.success or result.x is None:
        message = result.message if result.message else "linear program failed"
        raise FrechetClassInfeasibleError(
            f"No joint distribution satisfies the supplied Frechet-class constraints: {message}"
        )
    distribution = _clean_distribution(cast(FloatArray, result.x), feasibility_tol)
    residual = np.max(np.abs(a_eq @ distribution - b_eq))
    if residual > max(1.0e-7, 100.0 * feasibility_tol):
        raise FrechetClassInfeasibleError(
            f"LP solution violates equality constraints by {residual:.3e}."
        )
    value = float(objective @ distribution)
    return value, distribution


def _clean_distribution(distribution: FloatArray, tol: float) -> FloatArray:
    cleaned = np.asarray(distribution, dtype=float).copy()
    cleaned[np.abs(cleaned) <= tol] = 0.0
    if np.any(cleaned < -tol):
        raise FrechetClassInfeasibleError("LP returned negative atom probabilities.")
    cleaned = np.clip(cleaned, 0.0, 1.0)
    total = float(np.sum(cleaned))
    if total <= tol:
        raise FrechetClassInfeasibleError("LP returned a zero-mass distribution.")
    normalized = (cleaned / total).astype(np.float64)
    return cast(FloatArray, normalized)


def _snap_interval(lower: float, upper: float, tol: float) -> tuple[float, float]:
    if abs(lower - upper) <= tol:
        midpoint = 0.5 * (lower + upper)
        return midpoint, midpoint
    return lower, upper
