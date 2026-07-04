"""Finite binary partial-identification bounds by atom linear programming.

For named binary guardrail events ``G_0, ..., G_{K-1}``, the latent joint law is
a probability vector over the atom space ``Omega = {0, 1}**K``.  Any linear
query ``c.T @ p`` has sharp identified bounds from two linear programs over the
atom simplex, restricted by the declared linear equality and inequality
assumptions.

This module deliberately reuses the atom ordering from
``cc.kernel.frechet_classes.atom_matrix``: little-endian atoms
``00..0, 10..0, 01..0, ...``.  Classical Frechet-Hoeffding bounds are the
special case where the only declared assumptions are exact marginal moments.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from cc.kernel.frechet_classes import FrechetClassInfeasibleError, atom_matrix

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
BoolArray: TypeAlias = NDArray[np.bool_]
ConstraintSense: TypeAlias = Literal["==", "<=", ">="]
AtomPredicate: TypeAlias = Callable[[Mapping[str, int]], bool]
JsonScalar: TypeAlias = str | int | float | bool | None

DEFAULT_TOL = 1.0e-9

__all__ = [
    "AssumptionSet",
    "IdentificationInfeasibleError",
    "IdentificationResult",
    "LinearConstraint",
    "LinearQuery",
    "identified_region",
]


class IdentificationInfeasibleError(FrechetClassInfeasibleError):
    """Raised when declared atom-space assumptions admit no joint law."""


PartialIdentificationInfeasibleError = IdentificationInfeasibleError


@dataclass(frozen=True)
class LinearQuery:
    """A linear functional over the atom probability vector."""

    name: str
    coefficients: FloatArray
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_name("query name", self.name)
        object.__setattr__(
            self,
            "coefficients",
            _as_float_vector(self.coefficients, label=f"query {self.name!r} coefficients"),
        )

    @classmethod
    def from_coefficients(
        cls,
        guardrails: Sequence[str],
        name: str,
        coefficients: ArrayLike,
        *,
        description: str | None = None,
    ) -> LinearQuery:
        """Build a query from explicit coefficients, validating atom length."""

        n_atoms = _n_atoms_for_guardrails(guardrails)
        coeffs = _as_float_vector(
            coefficients,
            expected_length=n_atoms,
            label=f"query {name!r} coefficients",
        )
        return cls(name=name, coefficients=coeffs, description=description)

    @classmethod
    def marginal(
        cls,
        guardrails: Sequence[str],
        guardrail: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> LinearQuery:
        """Return the query ``P(guardrail)``."""

        guardrail_names = _guardrails_tuple(guardrails)
        coeffs = event_mask(guardrail_names, guardrail).astype(float)
        query_name = name if name is not None else f"P({guardrail})"
        return cls(name=query_name, coefficients=coeffs, description=description)

    @classmethod
    def joint(
        cls,
        guardrails: Sequence[str],
        left: str,
        right: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> LinearQuery:
        """Return the query ``P(left and right)``."""

        query_name = name if name is not None else f"P({left} and {right})"
        return cls.intersection(
            guardrails,
            (left, right),
            name=query_name,
            description=description,
        )

    @classmethod
    def intersection(
        cls,
        guardrails: Sequence[str],
        events: Sequence[str],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> LinearQuery:
        """Return the query that all listed events occur."""

        guardrail_names = _guardrails_tuple(guardrails)
        event_names = _event_tuple(events, guardrail_names)
        states = enumerate_atoms(guardrail_names)
        indices = [_guardrail_index(guardrail_names, event) for event in event_names]
        coeffs = np.all(states[:, indices] == 1, axis=1).astype(float)
        query_name = name if name is not None else f"P({' and '.join(event_names)})"
        return cls(name=query_name, coefficients=coeffs, description=description)

    @classmethod
    def union(
        cls,
        guardrails: Sequence[str],
        events: Sequence[str],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> LinearQuery:
        """Return the query that at least one listed event occurs."""

        guardrail_names = _guardrails_tuple(guardrails)
        event_names = _event_tuple(events, guardrail_names)
        states = enumerate_atoms(guardrail_names)
        indices = [_guardrail_index(guardrail_names, event) for event in event_names]
        coeffs = np.any(states[:, indices] == 1, axis=1).astype(float)
        query_name = name if name is not None else f"P({' or '.join(event_names)})"
        return cls(name=query_name, coefficients=coeffs, description=description)

    @classmethod
    def from_predicate(
        cls,
        guardrails: Sequence[str],
        name: str,
        predicate: AtomPredicate,
        *,
        description: str | None = None,
    ) -> LinearQuery:
        """Return an event-probability query from an arbitrary atom predicate."""

        coeffs = atom_predicate_mask(guardrails, predicate).astype(float)
        return cls(name=name, coefficients=coeffs, description=description)


@dataclass(frozen=True)
class LinearConstraint:
    """A named linear equality or inequality over the atom probability vector."""

    name: str
    coefficients: FloatArray
    sense: ConstraintSense
    rhs: float
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_name("constraint name", self.name)
        if self.sense not in {"==", "<=", ">="}:
            raise ValueError('constraint sense must be one of "==", "<=", ">=".')
        rhs = float(self.rhs)
        if not np.isfinite(rhs):
            raise ValueError(f"constraint {self.name!r} rhs must be finite.")
        object.__setattr__(
            self,
            "coefficients",
            _as_float_vector(
                self.coefficients,
                label=f"constraint {self.name!r} coefficients",
            ),
        )
        object.__setattr__(self, "rhs", rhs)

    @classmethod
    def from_coefficients(
        cls,
        guardrails: Sequence[str],
        name: str,
        coefficients: ArrayLike,
        sense: ConstraintSense,
        rhs: float,
        *,
        description: str | None = None,
    ) -> LinearConstraint:
        """Build a constraint from explicit coefficients, validating atom length."""

        n_atoms = _n_atoms_for_guardrails(guardrails)
        coeffs = _as_float_vector(
            coefficients,
            expected_length=n_atoms,
            label=f"constraint {name!r} coefficients",
        )
        return cls(
            name=name,
            coefficients=coeffs,
            sense=sense,
            rhs=rhs,
            description=description,
        )

    @classmethod
    def equality(
        cls,
        guardrails: Sequence[str],
        name: str,
        coefficients: ArrayLike,
        rhs: float,
        *,
        description: str | None = None,
    ) -> LinearConstraint:
        """Build an equality constraint."""

        return cls.from_coefficients(
            guardrails,
            name,
            coefficients,
            "==",
            rhs,
            description=description,
        )

    @classmethod
    def upper_bound(
        cls,
        guardrails: Sequence[str],
        name: str,
        coefficients: ArrayLike,
        rhs: float,
        *,
        description: str | None = None,
    ) -> LinearConstraint:
        """Build an upper-bound inequality ``coefficients @ p <= rhs``."""

        return cls.from_coefficients(
            guardrails,
            name,
            coefficients,
            "<=",
            rhs,
            description=description,
        )

    @classmethod
    def lower_bound(
        cls,
        guardrails: Sequence[str],
        name: str,
        coefficients: ArrayLike,
        rhs: float,
        *,
        description: str | None = None,
    ) -> LinearConstraint:
        """Build a lower-bound inequality ``coefficients @ p >= rhs``."""

        return cls.from_coefficients(
            guardrails,
            name,
            coefficients,
            ">=",
            rhs,
            description=description,
        )


@dataclass(frozen=True)
class AssumptionSet:
    """Declared linear assumption surface for a named binary atom space."""

    guardrails: tuple[str, ...]
    constraints: tuple[LinearConstraint, ...] = ()
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        guardrail_names = _guardrails_tuple(self.guardrails)
        n_atoms = 1 << len(guardrail_names)
        checked_constraints = tuple(self.constraints)
        for constraint in checked_constraints:
            if not isinstance(constraint, LinearConstraint):
                raise TypeError("constraints must contain LinearConstraint instances.")
            _validate_vector_length(
                constraint.coefficients,
                n_atoms,
                label=f"constraint {constraint.name!r} coefficients",
            )
        metadata = _metadata_mapping(self.metadata)
        object.__setattr__(self, "guardrails", guardrail_names)
        object.__setattr__(self, "constraints", checked_constraints)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @classmethod
    def empty(
        cls,
        guardrails: Sequence[str],
        *,
        metadata: Mapping[str, JsonScalar] | None = None,
    ) -> AssumptionSet:
        """Return an unconstrained atom simplex for the named guardrails."""

        return cls(guardrails=_guardrails_tuple(guardrails), metadata=metadata or {})

    def with_constraint(self, constraint: LinearConstraint) -> AssumptionSet:
        """Return a new assumption set with one additional constraint."""

        expected_length = 1 << len(self.guardrails)
        _validate_vector_length(
            constraint.coefficients,
            expected_length,
            label=f"constraint {constraint.name!r} coefficients",
        )
        return AssumptionSet(
            guardrails=self.guardrails,
            constraints=(*self.constraints, constraint),
            metadata=self.metadata,
        )

    def with_linear_constraint(
        self,
        name: str,
        coefficients: ArrayLike,
        sense: ConstraintSense,
        rhs: float,
        *,
        description: str | None = None,
    ) -> AssumptionSet:
        """Return a new assumption set with an arbitrary linear constraint."""

        constraint = LinearConstraint.from_coefficients(
            self.guardrails,
            name,
            coefficients,
            sense,
            rhs,
            description=description,
        )
        return self.with_constraint(constraint)

    def with_marginal_interval(
        self,
        guardrail: str,
        lower: float,
        upper: float,
    ) -> AssumptionSet:
        """Add ``lower <= P(guardrail) <= upper``."""

        lo, hi = _validate_interval(f"marginal interval for {guardrail!r}", lower, upper)
        coeffs = LinearQuery.marginal(self.guardrails, guardrail).coefficients
        return self.with_linear_constraint(
            f"marginal:{guardrail}:lower",
            coeffs,
            ">=",
            lo,
        ).with_linear_constraint(
            f"marginal:{guardrail}:upper",
            coeffs,
            "<=",
            hi,
        )

    def with_pairwise_joint_interval(
        self,
        left: str,
        right: str,
        lower: float,
        upper: float,
    ) -> AssumptionSet:
        """Add ``lower <= P(left and right) <= upper``."""

        lo, hi = _validate_interval(
            f"pairwise joint interval for {left!r}, {right!r}",
            lower,
            upper,
        )
        coeffs = LinearQuery.joint(self.guardrails, left, right).coefficients
        pair_name = f"{left}&{right}"
        return self.with_linear_constraint(
            f"pairwise:{pair_name}:lower",
            coeffs,
            ">=",
            lo,
        ).with_linear_constraint(
            f"pairwise:{pair_name}:upper",
            coeffs,
            "<=",
            hi,
        )

    def with_monotonicity(self, antecedent: str, consequent: str) -> AssumptionSet:
        """Add the implication constraint ``antecedent => consequent``."""

        _guardrail_index(self.guardrails, antecedent)
        _guardrail_index(self.guardrails, consequent)
        if antecedent == consequent:
            raise ValueError("monotonicity requires two distinct guardrails.")

        coeffs = atom_predicate_mask(
            self.guardrails,
            lambda atom: atom[antecedent] == 1 and atom[consequent] == 0,
        ).astype(float)
        return self.with_linear_constraint(
            f"monotonicity:{antecedent}=>{consequent}",
            coeffs,
            "==",
            0.0,
        )

    def add_constraint(self, constraint: LinearConstraint) -> AssumptionSet:
        """Alias for :meth:`with_constraint`; frozen sets return a new instance."""

        return self.with_constraint(constraint)

    def add_linear_constraint(
        self,
        name: str,
        coefficients: ArrayLike,
        sense: ConstraintSense,
        rhs: float,
        *,
        description: str | None = None,
    ) -> AssumptionSet:
        """Alias for :meth:`with_linear_constraint`."""

        return self.with_linear_constraint(
            name,
            coefficients,
            sense,
            rhs,
            description=description,
        )

    def add_marginal_interval(
        self,
        guardrail: str,
        lower: float,
        upper: float,
    ) -> AssumptionSet:
        """Alias for :meth:`with_marginal_interval`."""

        return self.with_marginal_interval(guardrail, lower, upper)

    def add_pairwise_joint_interval(
        self,
        left: str,
        right: str,
        lower: float,
        upper: float,
    ) -> AssumptionSet:
        """Alias for :meth:`with_pairwise_joint_interval`."""

        return self.with_pairwise_joint_interval(left, right, lower, upper)

    def add_monotonicity(self, antecedent: str, consequent: str) -> AssumptionSet:
        """Alias for :meth:`with_monotonicity`."""

        return self.with_monotonicity(antecedent, consequent)

    def stable_hash(self) -> str:
        """Return a deterministic SHA-256 hash of the declared assumptions.

        The payload includes guardrail order, constraint names, senses, right-hand
        sides, coefficient vectors, and metadata. Constraint descriptions are
        intentionally excluded because they do not change the feasible atom set.
        """

        return _stable_assumptions_hash(self)

    def identify(
        self,
        query: LinearQuery,
        *,
        feasibility_tol: float = DEFAULT_TOL,
    ) -> IdentificationResult:
        """Return sharp identified bounds for ``query`` under these assumptions."""

        return identified_region(query, self, feasibility_tol=feasibility_tol)


@dataclass(frozen=True)
class IdentificationResult:
    """Sharp lower and upper LP bounds for a linear atom query."""

    query_name: str
    lower_bound: float
    upper_bound: float
    lower_solution: FloatArray
    upper_solution: FloatArray
    solver_status: str
    active_constraints: tuple[str, ...]
    assumptions_hash: str
    lower_objective_status: str
    upper_objective_status: str
    lower_message: str
    upper_message: str
    active_constraints_lower: tuple[str, ...] = ()
    active_constraints_upper: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower_bound", float(self.lower_bound))
        object.__setattr__(self, "upper_bound", float(self.upper_bound))
        if self.upper_bound + DEFAULT_TOL < self.lower_bound:
            raise IdentificationInfeasibleError(
                f"identified interval is empty: [{self.lower_bound}, {self.upper_bound}]."
            )
        object.__setattr__(
            self,
            "lower_solution",
            _as_float_vector(self.lower_solution, label="lower_solution"),
        )
        object.__setattr__(
            self,
            "upper_solution",
            _as_float_vector(self.upper_solution, label="upper_solution"),
        )
        if self.lower_solution.size != self.upper_solution.size:
            raise ValueError("lower_solution and upper_solution must have the same length.")
        object.__setattr__(self, "active_constraints", tuple(self.active_constraints))
        object.__setattr__(
            self,
            "active_constraints_lower",
            tuple(self.active_constraints_lower),
        )
        object.__setattr__(
            self,
            "active_constraints_upper",
            tuple(self.active_constraints_upper),
        )

    @property
    def width(self) -> float:
        """Return ``upper_bound - lower_bound`` with tiny negatives clipped."""

        return max(0.0, self.upper_bound - self.lower_bound)


def enumerate_atoms(guardrails: Sequence[str]) -> IntArray:
    """Return the deterministic ``2**K x K`` binary atom matrix."""

    guardrail_names = _guardrails_tuple(guardrails)
    return atom_matrix(len(guardrail_names))


def event_mask(
    guardrails: Sequence[str],
    guardrail: str,
    *,
    value: bool = True,
) -> BoolArray:
    """Return a mask for atoms where ``guardrail`` has the requested truth value."""

    guardrail_names = _guardrails_tuple(guardrails)
    index = _guardrail_index(guardrail_names, guardrail)
    states = enumerate_atoms(guardrail_names)
    target = 1 if value else 0
    return cast(BoolArray, states[:, index] == target)


def atoms_where(
    guardrails: Sequence[str],
    guardrail: str,
    *,
    value: bool = True,
) -> IntArray:
    """Return atoms where ``guardrail`` has the requested truth value."""

    states = enumerate_atoms(guardrails)
    return states[event_mask(guardrails, guardrail, value=value)]


def atom_predicate_mask(
    guardrails: Sequence[str],
    predicate: AtomPredicate,
) -> BoolArray:
    """Return a mask for atoms satisfying an arbitrary named-atom predicate."""

    guardrail_names = _guardrails_tuple(guardrails)
    states = enumerate_atoms(guardrail_names)
    mask = np.zeros(states.shape[0], dtype=bool)
    for row_index, row in enumerate(states):
        atom = {name: int(row[col]) for col, name in enumerate(guardrail_names)}
        mask[row_index] = bool(predicate(atom))
    return cast(BoolArray, mask)


def atoms_matching(
    guardrails: Sequence[str],
    predicate: AtomPredicate,
) -> IntArray:
    """Return atoms satisfying an arbitrary named-atom predicate."""

    states = enumerate_atoms(guardrails)
    return states[atom_predicate_mask(guardrails, predicate)]


def identified_region(
    query: LinearQuery,
    assumptions: AssumptionSet,
    *,
    feasibility_tol: float = DEFAULT_TOL,
) -> IdentificationResult:
    """Solve the two atom LPs defining the sharp identified query interval."""

    if feasibility_tol <= 0.0 or not np.isfinite(feasibility_tol):
        raise ValueError("feasibility_tol must be a positive finite number.")
    n_atoms = 1 << len(assumptions.guardrails)
    _validate_vector_length(
        query.coefficients,
        n_atoms,
        label=f"query {query.name!r} coefficients",
    )
    a_eq, b_eq, a_ub, b_ub = _constraint_matrices(assumptions)

    lower = _solve_query_lp(
        query.coefficients,
        a_eq,
        b_eq,
        a_ub,
        b_ub,
        maximize=False,
        feasibility_tol=feasibility_tol,
    )
    upper = _solve_query_lp(
        query.coefficients,
        a_eq,
        b_eq,
        a_ub,
        b_ub,
        maximize=True,
        feasibility_tol=feasibility_tol,
    )

    lower_value = _snap_to_objective_range(
        lower.value,
        query.coefficients,
        feasibility_tol,
    )
    upper_value = _snap_to_objective_range(
        upper.value,
        query.coefficients,
        feasibility_tol,
    )
    if upper_value + feasibility_tol < lower_value:
        raise IdentificationInfeasibleError(
            f"Numerical LP returned an empty interval [{lower_value}, {upper_value}]."
        )
    lower_value, upper_value = _snap_interval(lower_value, upper_value, feasibility_tol)
    active_lower = _active_constraint_names(assumptions, lower.solution, feasibility_tol)
    active_upper = _active_constraint_names(assumptions, upper.solution, feasibility_tol)

    return IdentificationResult(
        query_name=query.name,
        lower_bound=lower_value,
        upper_bound=upper_value,
        lower_solution=lower.solution,
        upper_solution=upper.solution,
        solver_status="optimal",
        active_constraints=_ordered_union(active_lower, active_upper),
        assumptions_hash=assumptions.stable_hash(),
        lower_objective_status=lower.status,
        upper_objective_status=upper.status,
        lower_message=lower.message,
        upper_message=upper.message,
        active_constraints_lower=active_lower,
        active_constraints_upper=active_upper,
    )


def identify(
    query: LinearQuery,
    assumptions: AssumptionSet,
    *,
    feasibility_tol: float = DEFAULT_TOL,
) -> IdentificationResult:
    """Backward-compatible alias for :func:`identified_region`."""

    return identified_region(query, assumptions, feasibility_tol=feasibility_tol)


@dataclass(frozen=True)
class _SolvedObjective:
    value: float
    solution: FloatArray
    status: str
    message: str


def _constraint_matrices(
    assumptions: AssumptionSet,
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
    n_atoms = 1 << len(assumptions.guardrails)
    eq_rows: list[FloatArray] = [np.ones(n_atoms, dtype=float)]
    eq_rhs: list[float] = [1.0]
    ub_rows: list[FloatArray] = []
    ub_rhs: list[float] = []

    for constraint in assumptions.constraints:
        coeffs = constraint.coefficients
        if constraint.sense == "==":
            eq_rows.append(coeffs)
            eq_rhs.append(constraint.rhs)
        elif constraint.sense == "<=":
            ub_rows.append(coeffs)
            ub_rhs.append(constraint.rhs)
        else:
            ub_rows.append(-coeffs)
            ub_rhs.append(-constraint.rhs)

    a_eq = np.vstack(eq_rows).astype(float)
    b_eq = np.asarray(eq_rhs, dtype=float)
    if not ub_rows:
        return a_eq, b_eq, None, None
    return (
        a_eq,
        b_eq,
        np.vstack(ub_rows).astype(float),
        np.asarray(ub_rhs, dtype=float),
    )


def _solve_query_lp(
    objective: FloatArray,
    a_eq: FloatArray,
    b_eq: FloatArray,
    a_ub: FloatArray | None,
    b_ub: FloatArray | None,
    *,
    maximize: bool,
    feasibility_tol: float,
) -> _SolvedObjective:
    c = -objective if maximize else objective
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, 1.0)] * objective.size,
        method="highs",
        options={
            "dual_feasibility_tolerance": feasibility_tol,
            "primal_feasibility_tolerance": feasibility_tol,
        },
    )
    if not result.success or result.x is None:
        message = str(result.message) if result.message else "linear program failed"
        raise IdentificationInfeasibleError(
            "No atom distribution satisfies the supplied partial-identification "
            f"assumptions: {message}"
        )

    raw_solution = np.asarray(cast(FloatArray, result.x), dtype=float)
    _validate_lp_solution(
        raw_solution,
        a_eq,
        b_eq,
        a_ub,
        b_ub,
        feasibility_tol,
        label="raw LP solution",
    )
    solution = _clean_solution(raw_solution, feasibility_tol)
    _validate_lp_solution(
        solution,
        a_eq,
        b_eq,
        a_ub,
        b_ub,
        feasibility_tol,
        label="cleaned LP solution",
    )
    value = float(objective @ solution)
    return _SolvedObjective(
        value=value,
        solution=solution,
        status=_status_label(int(result.status)),
        message=str(result.message),
    )


def _clean_solution(solution: FloatArray, tol: float) -> FloatArray:
    effective_tol = _effective_validation_tol(tol)
    cleaned = np.asarray(solution, dtype=float).copy()
    cleaned[(cleaned < 0.0) & (cleaned >= -effective_tol)] = 0.0
    cleaned[(cleaned > 1.0) & (cleaned <= 1.0 + effective_tol)] = 1.0
    if np.any(cleaned < -effective_tol):
        raise IdentificationInfeasibleError("LP returned negative atom probabilities.")
    if np.any(cleaned > 1.0 + effective_tol):
        raise IdentificationInfeasibleError("LP returned atom probabilities above 1.")
    total = float(np.sum(cleaned))
    correction = 1.0 - total
    if 0.0 < abs(correction) <= effective_tol:
        pivot = int(np.argmax(cleaned))
        cleaned[pivot] = cleaned[pivot] + correction
    cleaned = cleaned.astype(np.float64, copy=False)
    cleaned.setflags(write=False)
    return cast(FloatArray, cleaned)


def _validate_lp_solution(
    solution: FloatArray,
    a_eq: FloatArray,
    b_eq: FloatArray,
    a_ub: FloatArray | None,
    b_ub: FloatArray | None,
    tol: float,
    *,
    label: str,
) -> None:
    effective_tol = _effective_validation_tol(tol)
    if solution.ndim != 1:
        raise IdentificationInfeasibleError(f"{label} is not a probability vector.")
    if not np.all(np.isfinite(solution)):
        raise IdentificationInfeasibleError(f"{label} contains non-finite atom probabilities.")
    if np.any(solution < -effective_tol):
        raise IdentificationInfeasibleError(f"{label} has negative atom mass.")
    if np.any(solution > 1.0 + effective_tol):
        raise IdentificationInfeasibleError(f"{label} has atom mass above 1.")
    total = float(np.sum(solution))
    if abs(total - 1.0) > effective_tol:
        raise IdentificationInfeasibleError(f"{label} atom probabilities sum to {total}, not 1.")
    eq_residual = float(np.max(np.abs(a_eq @ solution - b_eq)))
    if eq_residual > effective_tol:
        raise IdentificationInfeasibleError(
            f"{label} violates equality constraints by {eq_residual:.3e}."
        )
    if a_ub is not None and b_ub is not None:
        ub_residual = float(np.max(a_ub @ solution - b_ub))
        if ub_residual > effective_tol:
            raise IdentificationInfeasibleError(
                f"{label} violates inequality constraints by {ub_residual:.3e}."
            )


def _effective_validation_tol(tol: float) -> float:
    return float(tol + 100.0 * np.finfo(np.float64).eps)


def _active_constraint_names(
    assumptions: AssumptionSet,
    solution: FloatArray,
    tol: float,
) -> tuple[str, ...]:
    names: list[str] = []
    for constraint in assumptions.constraints:
        lhs = float(constraint.coefficients @ solution)
        residual = abs(lhs - constraint.rhs)
        if residual <= tol:
            names.append(constraint.name)
    return tuple(names)


def _ordered_union(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for name in (*left, *right):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _stable_assumptions_hash(assumptions: AssumptionSet) -> str:
    payload = {
        "guardrails": list(assumptions.guardrails),
        "constraints": [
            {
                "name": constraint.name,
                "sense": constraint.sense,
                "rhs": _canonical_float(constraint.rhs),
                "coefficients": [_canonical_float(value) for value in constraint.coefficients],
            }
            for constraint in assumptions.constraints
        ],
        "metadata": {
            key: _canonical_json_scalar(value)
            for key, value in sorted(assumptions.metadata.items())
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guardrails_tuple(guardrails: Sequence[str]) -> tuple[str, ...]:
    if isinstance(guardrails, str):
        raise TypeError("guardrails must be a sequence of names, not a string.")
    names = tuple(str(name) for name in guardrails)
    if not names:
        raise ValueError("at least one guardrail is required.")
    for name in names:
        _validate_name("guardrail name", name)
    if len(set(names)) != len(names):
        raise ValueError("guardrail names must be unique.")
    return names


def _event_tuple(events: Sequence[str], guardrails: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(events, str):
        raise TypeError("events must be a sequence of guardrail names, not a string.")
    event_names = tuple(str(event) for event in events)
    if not event_names:
        raise ValueError("at least one event is required.")
    for event in event_names:
        _guardrail_index(guardrails, event)
    return event_names


def _guardrail_index(guardrails: Sequence[str], guardrail: str) -> int:
    try:
        return tuple(guardrails).index(guardrail)
    except ValueError as exc:
        raise ValueError(f"unknown guardrail {guardrail!r}.") from exc


def _n_atoms_for_guardrails(guardrails: Sequence[str]) -> int:
    return 1 << len(_guardrails_tuple(guardrails))


def _as_float_vector(
    values: ArrayLike,
    *,
    expected_length: int | None = None,
    label: str,
) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional vector.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain finite values.")
    if expected_length is not None:
        _validate_vector_length(arr, expected_length, label=label)
    copied = arr.astype(np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def _validate_vector_length(values: ArrayLike, expected_length: int, *, label: str) -> None:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size != expected_length:
        raise ValueError(f"{label} length must equal 2**K ({expected_length}).")


def _validate_name(label: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value.strip():
        raise ValueError(f"{label} must be nonempty.")


def _validate_interval(label: str, lower: float, upper: float) -> tuple[float, float]:
    lo = float(lower)
    hi = float(upper)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{label} endpoints must be finite.")
    if lo < -DEFAULT_TOL or hi > 1.0 + DEFAULT_TOL:
        raise ValueError(f"{label} endpoints must lie in [0, 1].")
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if lo > hi + DEFAULT_TOL:
        raise ValueError(f"{label} must satisfy lower <= upper.")
    if abs(lo - hi) <= DEFAULT_TOL:
        midpoint = 0.5 * (lo + hi)
        return midpoint, midpoint
    return lo, hi


def _metadata_mapping(metadata: Mapping[str, JsonScalar]) -> dict[str, JsonScalar]:
    out: dict[str, JsonScalar] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata keys must be nonempty strings.")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"metadata value for {key!r} must be finite.")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise TypeError("metadata values must be JSON scalar values.")
        out[key] = value
    return out


def _canonical_json_scalar(value: JsonScalar) -> JsonScalar | str:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        return _canonical_float(value)
    return value


def _canonical_float(value: Any) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("canonical numeric values must be finite.")
    if abs(number) <= DEFAULT_TOL:
        number = 0.0
    return format(number, ".17g")


def _snap_to_objective_range(value: float, coefficients: FloatArray, tol: float) -> float:
    lower = float(np.min(coefficients))
    upper = float(np.max(coefficients))
    slack = max(1.0e-8, 100.0 * tol)
    if value < lower and lower - value <= slack:
        return lower
    if value > upper and value - upper <= slack:
        return upper
    return float(value)


def _snap_interval(lower: float, upper: float, tol: float) -> tuple[float, float]:
    if abs(lower - upper) <= tol:
        midpoint = 0.5 * (lower + upper)
        return midpoint, midpoint
    return lower, upper


def _status_label(status: int) -> str:
    if status == 0:
        return "optimal"
    return f"status:{status}"
