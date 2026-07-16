"""Nonparametric Frechet sensitivity bounds for binary atom spaces.

This module is an explicit Phase 1 assurance kernel for bounded AI evidence.
It models the latent joint law over ``Omega = {0, 1}**n`` as a probability
vector over all binary atoms. Marginal and pairwise dependence declarations are
linear equality constraints. A caller supplies a linear query over the atoms,
and the module computes the sharp lower and upper identified values by solving
two linear programs.

Receipt integrity, human review, and policy approval do not enter this
calculation. The returned interval is only the consequence of the declared
probabilistic constraints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from cc.kernel.frechet_classes import atom_matrix
from cc.kernel.sensitivity import IdentificationInfeasibleError

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]

DEFAULT_TOLERANCE = 1.0e-9
MAX_EXPLICIT_EVENTS = 24

__all__ = [
    "DEFAULT_TOLERANCE",
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
]


@dataclass(frozen=True)
class MarginalEquality:
    """Exact marginal equality ``P(A_index = 1) = probability``."""

    index: int
    probability: float


@dataclass(frozen=True)
class PairwiseJointEquality:
    """Exact pairwise dependence equality ``P(A_left = 1, A_right = 1)``."""

    left: int
    right: int
    probability: float


@dataclass(frozen=True)
class LinearAtomQuery:
    """A named linear functional over the atom probability vector."""

    name: str
    coefficients: FloatArray
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("query name must be a non-empty string.")
        coeffs = _as_float_vector(self.coefficients, label=f"query {self.name!r} coefficients")
        object.__setattr__(self, "coefficients", coeffs)

    @classmethod
    def from_coefficients(
        cls,
        n_events: int,
        name: str,
        coefficients: ArrayLike,
        *,
        description: str | None = None,
    ) -> LinearAtomQuery:
        """Build a query from explicit coefficients and validate length ``2**n``."""

        n_int = _validate_n_events(n_events)
        coeffs = _as_float_vector(
            coefficients,
            expected_length=1 << n_int,
            label=f"query {name!r} coefficients",
        )
        return cls(name=name, coefficients=coeffs, description=description)


@dataclass(frozen=True)
class FrechetSensitivityResult:
    """Sharp lower and upper bounds for one atom-linear query."""

    query_name: str
    lower_bound: float
    upper_bound: float
    lower_distribution: FloatArray
    upper_distribution: FloatArray
    marginals: tuple[MarginalEquality, ...]
    pairwise: tuple[PairwiseJointEquality, ...]
    tolerance: float
    atom_order: str = "little-endian bits; atom_matrix(n_events)"

    def __post_init__(self) -> None:
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        tol = _validate_tolerance(self.tolerance)
        if upper + tol < lower:
            raise IdentificationInfeasibleError(f"sharp interval is empty: [{lower}, {upper}]")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(
            self,
            "lower_distribution",
            _as_probability_distribution(self.lower_distribution, "lower_distribution", tol),
        )
        object.__setattr__(
            self,
            "upper_distribution",
            _as_probability_distribution(self.upper_distribution, "upper_distribution", tol),
        )
        if self.lower_distribution.size != self.upper_distribution.size:
            raise ValueError("lower_distribution and upper_distribution lengths differ.")

    @property
    def width(self) -> float:
        """Return ``upper_bound - lower_bound`` with numerical negatives clipped."""

        return max(0.0, self.upper_bound - self.lower_bound)


def binary_atoms(n_events: int) -> IntArray:
    """Return the deterministic ``2**n_events x n_events`` binary atom matrix."""

    return atom_matrix(_validate_n_events(n_events))


def all_events_query(n_events: int, *, name: str = "P(all_events)") -> LinearAtomQuery:
    """Return the query for the probability that all events occur."""

    return event_query(
        n_events,
        name=name,
        predicate=lambda states: np.all(states == 1, axis=1),
        description="Probability of the all-ones atom set.",
    )


def any_event_query(n_events: int, *, name: str = "P(any_event)") -> LinearAtomQuery:
    """Return the query for the probability that at least one event occurs."""

    return event_query(
        n_events,
        name=name,
        predicate=lambda states: np.any(states == 1, axis=1),
        description="Probability of the nonzero atom set.",
    )


def event_query(
    n_events: int,
    *,
    name: str,
    predicate: CallablePredicate,
    description: str | None = None,
) -> LinearAtomQuery:
    """Build a ``0/1`` event-probability query from an atom predicate."""

    states = binary_atoms(n_events)
    mask = np.asarray(predicate(states), dtype=bool)
    if mask.ndim != 1 or mask.size != states.shape[0]:
        raise ValueError("predicate must return a one-dimensional mask over all atoms.")
    return LinearAtomQuery.from_coefficients(
        n_events,
        name,
        mask.astype(float),
        description=description,
    )


def pairwise_joint_from_phi(
    p_left: float,
    p_right: float,
    phi: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Convert binary Pearson phi into ``P(A_left and A_right)``.

    The formula is ``p_l p_r + phi * sqrt(p_l(1-p_l)p_r(1-p_r))``. The result
    must still lie inside the two-event Frechet-Hoeffding interval.
    """

    tol = _validate_tolerance(tolerance)
    _validate_probability("p_left", p_left, tol)
    _validate_probability("p_right", p_right, tol)
    if not np.isfinite(phi) or phi < -1.0 - tol or phi > 1.0 + tol:
        raise ValueError(f"phi must be finite and lie in [-1, 1]. Got {phi}.")
    variance_scale = p_left * (1.0 - p_left) * p_right * (1.0 - p_right)
    if variance_scale <= tol:
        raise ValueError("phi dependence is undefined for degenerate Bernoulli margins.")
    joint = p_left * p_right + float(phi) * float(np.sqrt(variance_scale))
    lower, upper = _pairwise_frechet_bounds(p_left, p_right)
    if joint < lower - tol or joint > upper + tol:
        raise ValueError(f"phi={phi} implies pairwise joint {joint}, outside [{lower}, {upper}].")
    return float(np.clip(joint, lower, upper))


def sharp_frechet_bounds(
    marginals: Sequence[float | MarginalEquality] | Mapping[int, float],
    query: LinearAtomQuery | ArrayLike,
    *,
    pairwise: Sequence[PairwiseJointEquality] | Mapping[tuple[int, int], float] = (),
    query_name: str = "linear_query",
    tolerance: float = DEFAULT_TOLERANCE,
) -> FrechetSensitivityResult:
    """Compute sharp lower and upper bounds for an atom-linear query.

    Parameters
    ----------
    marginals:
        Complete marginal equality declarations. A sequence of floats declares
        event indices ``0..n-1``. A mapping or sequence of ``MarginalEquality``
        must still cover every event exactly once.
    query:
        A ``LinearAtomQuery`` or explicit coefficient vector of length ``2**n``.
    pairwise:
        Optional exact pairwise joint probability constraints. These are
        dependence declarations, not independence assumptions.
    query_name:
        Name used when ``query`` is passed as raw coefficients.
    tolerance:
        Positive feasibility tolerance for solver checks and endpoint snapping.
    """

    tol = _validate_tolerance(tolerance)
    marginal_tuple = _canonical_marginals(marginals, tol)
    n_events = len(marginal_tuple)
    n_atoms = 1 << n_events
    if isinstance(query, LinearAtomQuery):
        atom_query = query
        _validate_vector_length(
            atom_query.coefficients,
            n_atoms,
            label=f"query {atom_query.name!r} coefficients",
        )
    else:
        atom_query = LinearAtomQuery.from_coefficients(n_events, query_name, query)

    pairwise_tuple = _canonical_pairwise(pairwise, n_events, marginal_tuple, tol)
    atoms = binary_atoms(n_events).astype(float)
    a_eq, b_eq = _equality_system(atoms, marginal_tuple, pairwise_tuple)
    lower_value, lower_distribution = _solve_lp(
        atom_query.coefficients,
        a_eq,
        b_eq,
        maximize=False,
        tolerance=tol,
    )
    upper_value, upper_distribution = _solve_lp(
        atom_query.coefficients,
        a_eq,
        b_eq,
        maximize=True,
        tolerance=tol,
    )

    lower_value = _snap_to_coefficient_range(lower_value, atom_query.coefficients, tol)
    upper_value = _snap_to_coefficient_range(upper_value, atom_query.coefficients, tol)
    if upper_value + tol < lower_value:
        raise IdentificationInfeasibleError(
            f"LP returned an empty sharp interval [{lower_value}, {upper_value}]."
        )
    lower_value, upper_value = _snap_interval(lower_value, upper_value, tol)

    return FrechetSensitivityResult(
        query_name=atom_query.name,
        lower_bound=lower_value,
        upper_bound=upper_value,
        lower_distribution=lower_distribution,
        upper_distribution=upper_distribution,
        marginals=marginal_tuple,
        pairwise=pairwise_tuple,
        tolerance=tol,
    )


CallablePredicate: TypeAlias = Callable[[IntArray], ArrayLike]


def _validate_n_events(n_events: int) -> int:
    if not isinstance(n_events, (int, np.integer)):
        raise TypeError("n_events must be an integer.")
    n_int = int(n_events)
    if n_int < 1:
        raise ValueError("n_events must be at least 1.")
    if n_int > MAX_EXPLICIT_EVENTS:
        raise ValueError(f"n_events={n_int} is too large for an explicit 2**n atom LP.")
    return n_int


def _validate_tolerance(tolerance: float) -> float:
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be a positive finite number.")
    return tol


def _as_float_vector(
    values: ArrayLike,
    *,
    label: str,
    expected_length: int | None = None,
) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional vector.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    if expected_length is not None:
        _validate_vector_length(arr, expected_length, label=label)
    out = arr.astype(np.float64, copy=True)
    out.setflags(write=False)
    return out


def _validate_vector_length(values: ArrayLike, expected_length: int, *, label: str) -> None:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size != expected_length:
        raise ValueError(f"{label} length must equal 2**n_events ({expected_length}).")


def _as_probability_distribution(values: ArrayLike, label: str, tol: float) -> FloatArray:
    arr = _as_float_vector(values, label=label)
    if np.any(arr < -tol):
        raise IdentificationInfeasibleError(f"{label} contains negative atom mass.")
    total = float(np.sum(arr))
    if abs(total - 1.0) > _effective_tol(tol):
        raise IdentificationInfeasibleError(f"{label} sums to {total}, not 1.")
    cleaned = np.clip(arr, 0.0, 1.0)
    cleaned = cleaned / float(np.sum(cleaned))
    cleaned.setflags(write=False)
    return cleaned


def _validate_probability(name: str, value: float, tol: float) -> None:
    if not np.isfinite(value) or value < -tol or value > 1.0 + tol:
        raise ValueError(f"{name} must be a finite probability in [0, 1]. Got {value}.")


def _canonical_marginals(
    marginals: Sequence[float | MarginalEquality] | Mapping[int, float],
    tol: float,
) -> tuple[MarginalEquality, ...]:
    if isinstance(marginals, Mapping):
        items = tuple(
            MarginalEquality(int(index), float(value)) for index, value in marginals.items()
        )
    else:
        if len(marginals) == 0:
            raise ValueError("at least one marginal equality is required.")
        first = marginals[0]
        if isinstance(first, MarginalEquality):
            items = tuple(cast(Sequence[MarginalEquality], marginals))
        else:
            items = tuple(
                MarginalEquality(index=index, probability=float(value))
                for index, value in enumerate(cast(Sequence[float], marginals))
            )

    by_index: dict[int, float] = {}
    for item in items:
        if not isinstance(item.index, (int, np.integer)):
            raise TypeError("marginal index must be an integer.")
        index = int(item.index)
        probability = float(item.probability)
        _validate_probability(f"marginal {index}", probability, tol)
        if index in by_index:
            raise ValueError(f"duplicate marginal equality for index {index}.")
        by_index[index] = float(np.clip(probability, 0.0, 1.0))
    expected = set(range(len(by_index)))
    actual = set(by_index)
    if actual != expected:
        raise ValueError("marginal equalities must cover contiguous indices 0..n-1 exactly.")
    return tuple(
        MarginalEquality(index=index, probability=by_index[index]) for index in sorted(by_index)
    )


def _canonical_pairwise(
    pairwise: Sequence[PairwiseJointEquality] | Mapping[tuple[int, int], float],
    n_events: int,
    marginals: tuple[MarginalEquality, ...],
    tol: float,
) -> tuple[PairwiseJointEquality, ...]:
    if isinstance(pairwise, Mapping):
        raw = tuple(
            PairwiseJointEquality(int(left), int(right), float(value))
            for (left, right), value in pairwise.items()
        )
    else:
        raw = tuple(pairwise)
    marginal_values = tuple(item.probability for item in marginals)
    seen: dict[tuple[int, int], float] = {}
    for item in raw:
        if not isinstance(item.left, (int, np.integer)) or not isinstance(
            item.right, (int, np.integer)
        ):
            raise TypeError("pairwise indices must be integers.")
        left = int(item.left)
        right = int(item.right)
        if left == right:
            raise ValueError("pairwise constraints require distinct event indices.")
        if left < 0 or right < 0 or left >= n_events or right >= n_events:
            raise ValueError(f"pairwise index ({left}, {right}) is outside 0..{n_events - 1}.")
        if right < left:
            left, right = right, left
        probability = float(item.probability)
        _validate_probability(f"pairwise joint ({left}, {right})", probability, tol)
        fh_lower, fh_upper = _pairwise_frechet_bounds(
            marginal_values[left],
            marginal_values[right],
        )
        if probability < fh_lower - tol or probability > fh_upper + tol:
            raise ValueError(
                f"pairwise joint ({left}, {right})={probability} is outside "
                f"[{fh_lower}, {fh_upper}] for its marginals."
            )
        key = (left, right)
        previous = seen.get(key)
        if previous is not None and abs(previous - probability) > tol:
            raise ValueError(
                f"duplicate pairwise constraints for {key} disagree: {previous} vs {probability}."
            )
        seen[key] = float(np.clip(probability, fh_lower, fh_upper))
    return tuple(
        PairwiseJointEquality(left=left, right=right, probability=probability)
        for (left, right), probability in sorted(seen.items())
    )


def _pairwise_frechet_bounds(p_left: float, p_right: float) -> tuple[float, float]:
    return max(0.0, p_left + p_right - 1.0), min(p_left, p_right)


def _equality_system(
    atoms: FloatArray,
    marginals: tuple[MarginalEquality, ...],
    pairwise: tuple[PairwiseJointEquality, ...],
) -> tuple[FloatArray, FloatArray]:
    rows: list[FloatArray] = [np.ones(atoms.shape[0], dtype=float)]
    rhs: list[float] = [1.0]
    for marginal in marginals:
        rows.append(atoms[:, marginal.index])
        rhs.append(marginal.probability)
    for constraint in pairwise:
        rows.append(atoms[:, constraint.left] * atoms[:, constraint.right])
        rhs.append(constraint.probability)
    return np.vstack(rows).astype(float), np.asarray(rhs, dtype=float)


def _solve_lp(
    coefficients: FloatArray,
    a_eq: FloatArray,
    b_eq: FloatArray,
    *,
    maximize: bool,
    tolerance: float,
) -> tuple[float, FloatArray]:
    objective = -coefficients if maximize else coefficients
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * coefficients.size,
        method="highs",
        options={
            "dual_feasibility_tolerance": tolerance,
            "primal_feasibility_tolerance": tolerance,
        },
    )
    if not result.success or result.x is None:
        message = str(result.message) if result.message else "linear program failed"
        raise IdentificationInfeasibleError(
            f"Declared marginal and pairwise constraints empty the Frechet class: {message}"
        )
    solution = _clean_solution(cast(FloatArray, result.x), tolerance)
    residual = float(np.max(np.abs(a_eq @ solution - b_eq)))
    if residual > _effective_tol(tolerance):
        raise IdentificationInfeasibleError(
            f"LP solution violates equality constraints by {residual:.3e}."
        )
    return float(coefficients @ solution), solution


def _clean_solution(solution: FloatArray, tol: float) -> FloatArray:
    effective_tol = _effective_tol(tol)
    cleaned = np.asarray(solution, dtype=float).copy()
    cleaned[(cleaned < 0.0) & (cleaned >= -effective_tol)] = 0.0
    cleaned[(cleaned > 1.0) & (cleaned <= 1.0 + effective_tol)] = 1.0
    if np.any(cleaned < -effective_tol):
        raise IdentificationInfeasibleError("LP returned negative atom mass.")
    if np.any(cleaned > 1.0 + effective_tol):
        raise IdentificationInfeasibleError("LP returned atom mass above 1.")
    total = float(np.sum(cleaned))
    correction = 1.0 - total
    if abs(correction) <= effective_tol:
        pivot = int(np.argmax(cleaned))
        cleaned[pivot] += correction
    cleaned = np.clip(cleaned, 0.0, 1.0)
    total = float(np.sum(cleaned))
    if abs(total - 1.0) > effective_tol:
        raise IdentificationInfeasibleError(f"LP atom probabilities sum to {total}, not 1.")
    cleaned = (cleaned / total).astype(np.float64, copy=False)
    cleaned.setflags(write=False)
    return cast(FloatArray, cleaned)


def _snap_to_coefficient_range(value: float, coefficients: FloatArray, tol: float) -> float:
    lower = float(np.min(coefficients))
    upper = float(np.max(coefficients))
    if value < lower - _effective_tol(tol) or value > upper + _effective_tol(tol):
        raise IdentificationInfeasibleError(
            f"objective value {value} is outside coefficient range [{lower}, {upper}]."
        )
    if abs(value - lower) <= tol:
        return lower
    if abs(value - upper) <= tol:
        return upper
    return float(value)


def _snap_interval(lower: float, upper: float, tol: float) -> tuple[float, float]:
    if abs(lower - upper) <= tol:
        midpoint = 0.5 * (lower + upper)
        return midpoint, midpoint
    return lower, upper


def _effective_tol(tol: float) -> float:
    return float(tol + 1000.0 * np.finfo(np.float64).eps)
