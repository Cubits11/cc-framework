"""Budget-constrained systemic stress tests for composed guardrails.

The stress kernel treats guardrail failures as Bernoulli events on the finite
atom space ``{0, 1}**n``.  A baseline dependence structure is a fixed-marginal
joint law over those atoms.  Stressing the stack means moving to another joint
law with the same guardrail-level failure probabilities while staying inside a
distance budget around the baseline coupling.

This is intentionally different from the unconstrained Frechet-Hoeffding
envelope.  The FH upper endpoint is returned as the budget-to-infinity limit,
but finite budgets only report risks reachable inside the specified local
dependence neighborhood.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isinf, log
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog, minimize  # type: ignore[import-untyped]

from cc.kernel.frechet_classes import (
    PairwiseDependence,
    atom_matrix,
    dependence_to_joint_probability,
    distribution_moments,
    event_probability,
    frechet_bounds,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
StressMetric: TypeAlias = Literal["wasserstein", "kl"]
StressEvent: TypeAlias = Literal["and", "or"]

_TOL = 1.0e-10
_DEFAULT_KL_SMOOTHING = 1.0e-12

__all__ = [
    "BaselineDependence",
    "ConditionalComposedRiskResult",
    "StressBudget",
    "StressMetric",
    "StressTestResult",
    "composed_risk_given_guardrail_X_failure",
    "composed_risk_given_guardrail_x_failure",
    "stress_test",
]


@dataclass(frozen=True)
class BaselineDependence:
    """Estimated fixed-marginal dependence structure for guardrail failures.

    Prefer passing an atom ``distribution`` when it is available.  The atom
    ordering is the one returned by :func:`cc.kernel.frechet_classes.atom_matrix`:
    little-endian binary atoms ``00..0, 10..0, 01..0, ...``.  Empirical binary
    ``samples`` are converted to that atom distribution.  For two guardrails,
    callers may instead pass ``marginals`` plus one pairwise dependence value.
    """

    marginals: Sequence[float] | None = None
    distribution: Sequence[float] | None = None
    samples: ArrayLike | None = None
    pairwise: Iterable[PairwiseDependence | Mapping[str, object]] | None = None
    name: str = "baseline"


@dataclass(frozen=True)
class StressBudget:
    """Distance budget around the baseline copula/dependence law.

    ``metric="wasserstein"`` uses a one-Wasserstein distance on binary atoms
    with normalized Hamming ground cost.  ``metric="kl"`` uses a regularized
    relative-entropy distance between the stressed and baseline atom laws.
    """

    amount: float
    metric: StressMetric = "wasserstein"
    event: StressEvent = "and"
    kl_smoothing: float = _DEFAULT_KL_SMOOTHING
    solver_tol: float = _TOL


@dataclass(frozen=True)
class StressTestResult:
    """Worst composed-system risk reachable inside a finite stress budget."""

    baseline_risk: float
    stressed_risk: float
    risk_increase: float
    stress_budget: StressBudget
    realized_distance: float
    baseline_distribution: FloatArray
    stressed_distribution: FloatArray
    marginals: FloatArray
    frechet_lower: float
    frechet_upper: float
    frechet_limit_distribution: FloatArray
    event: StressEvent
    method: str
    message: str

    @property
    def baseline_effective_protection(self) -> float:
        """Return ``1 - baseline_risk``."""

        return 1.0 - self.baseline_risk

    @property
    def stressed_effective_protection(self) -> float:
        """Return ``1 - stressed_risk``."""

        return 1.0 - self.stressed_risk

    @property
    def protection_drop(self) -> float:
        """Return the loss of effective protection under the stress move."""

        return self.stressed_risk - self.baseline_risk

    @property
    def gap_to_frechet_limit(self) -> float:
        """Return the remaining distance from the unbudgeted FH upper endpoint."""

        return max(0.0, self.frechet_upper - self.stressed_risk)


@dataclass(frozen=True)
class ConditionalComposedRiskResult:
    """CoVaR-style risk conditional on one guardrail already failing."""

    guardrail_index: int
    unconditional_risk: float
    conditional_risk: float
    unconditional_effective_protection: float
    conditional_effective_protection: float
    protection_drop: float
    risk_lift: float
    conditioning_probability: float
    distribution: FloatArray
    stress_result: StressTestResult | None = None


@dataclass(frozen=True)
class _ResolvedDependence:
    distribution: FloatArray
    marginals: FloatArray
    n_events: int
    name: str


def stress_test(
    baseline_dependence: BaselineDependence | Mapping[str, object] | ArrayLike,
    stress_budget: StressBudget | Mapping[str, object] | float,
) -> StressTestResult:
    """Return the worst composed risk reachable within ``stress_budget``.

    The optimization keeps the guardrail-level failure marginals fixed and
    perturbs only the dependence law.  For infinite budgets, the result is the
    classical FH upper endpoint, explicitly marked as the unbudgeted limit.
    """

    baseline = _resolve_dependence(baseline_dependence)
    budget = _coerce_budget(stress_budget)
    states = atom_matrix(baseline.n_events).astype(float)
    objective = _event_vector(states, budget.event)
    baseline_risk = _clip01(event_probability(
        baseline.distribution,
        event=budget.event,
        n_events=baseline.n_events,
    ))

    fh = frechet_bounds(
        baseline.marginals,
        event=budget.event,
        return_distributions=True,
        feasibility_tol=budget.solver_tol,
    )
    if fh.upper_distribution is None:
        raise RuntimeError("FH solver did not return an upper-limit distribution.")
    fh_distribution = fh.upper_distribution

    if isinf(budget.amount):
        stressed = fh_distribution.copy()
        realized = _distance_between(baseline.distribution, stressed, states, budget)
        stressed_risk = _clip01(float(objective @ stressed))
        return StressTestResult(
            baseline_risk=baseline_risk,
            stressed_risk=stressed_risk,
            risk_increase=stressed_risk - baseline_risk,
            stress_budget=budget,
            realized_distance=realized,
            baseline_distribution=baseline.distribution.copy(),
            stressed_distribution=stressed,
            marginals=baseline.marginals.copy(),
            frechet_lower=fh.lower,
            frechet_upper=fh.upper,
            frechet_limit_distribution=fh_distribution.copy(),
            event=budget.event,
            method=f"{budget.metric}:fh_limit",
            message="Infinite stress budget: returned the Frechet-Hoeffding upper limit.",
        )

    if budget.amount <= budget.solver_tol:
        stressed = baseline.distribution.copy()
        stressed_risk = baseline_risk
        realized = 0.0
        method = f"{budget.metric}:zero_budget"
        message = "Zero stress budget: baseline dependence is the only feasible move."
    elif budget.metric == "wasserstein":
        stressed, realized, message = _solve_wasserstein_stress(
            baseline.distribution,
            baseline.marginals,
            states,
            objective,
            budget,
        )
        stressed_risk = _clip01(float(objective @ stressed))
        method = "wasserstein_lp"
    else:
        stressed, realized, message = _solve_kl_stress(
            baseline.distribution,
            baseline.marginals,
            states,
            objective,
            budget,
        )
        stressed_risk = _clip01(float(objective @ stressed))
        method = "kl_slsqp"

    if stressed_risk > fh.upper and stressed_risk <= fh.upper + 1.0e-7:
        stressed_risk = fh.upper
    return StressTestResult(
        baseline_risk=baseline_risk,
        stressed_risk=stressed_risk,
        risk_increase=max(0.0, stressed_risk - baseline_risk),
        stress_budget=budget,
        realized_distance=realized,
        baseline_distribution=baseline.distribution.copy(),
        stressed_distribution=stressed.copy(),
        marginals=baseline.marginals.copy(),
        frechet_lower=fh.lower,
        frechet_upper=fh.upper,
        frechet_limit_distribution=fh_distribution.copy(),
        event=budget.event,
        method=method,
        message=message,
    )


def composed_risk_given_guardrail_X_failure(
    baseline_dependence: BaselineDependence | Mapping[str, object] | ArrayLike,
    guardrail_index: int,
    stress_budget: StressBudget | Mapping[str, object] | float | None = None,
) -> ConditionalComposedRiskResult:
    """Return all-rails-fail risk conditional on guardrail ``X`` failing.

    If ``stress_budget`` is supplied, the conditional measure is evaluated on
    the worst-case stressed distribution returned by :func:`stress_test`.
    Otherwise it is evaluated on the supplied baseline dependence law.
    """

    stress_result: StressTestResult | None = None
    if stress_budget is None:
        resolved = _resolve_dependence(baseline_dependence)
        distribution = resolved.distribution
        marginals = resolved.marginals
        n_events = resolved.n_events
    else:
        stress_result = stress_test(baseline_dependence, stress_budget)
        distribution = stress_result.stressed_distribution
        marginals = stress_result.marginals
        n_events = int(marginals.size)

    if not isinstance(guardrail_index, (int, np.integer)):
        raise TypeError("guardrail_index must be an integer.")
    index = int(guardrail_index)
    if index < 0 or index >= n_events:
        raise ValueError(f"guardrail_index must be in [0, {n_events - 1}].")

    conditioning_probability = float(marginals[index])
    if conditioning_probability <= _TOL:
        raise ValueError("Cannot condition on a zero-probability guardrail failure.")

    states = atom_matrix(n_events).astype(float)
    all_fail = _event_vector(states, "and")
    guardrail_failed = states[:, index]
    unconditional_risk = _clip01(float(all_fail @ distribution))
    joint = _clip01(float((all_fail * guardrail_failed) @ distribution))
    conditional_risk = _clip01(joint / conditioning_probability)
    risk_lift = conditional_risk - unconditional_risk

    return ConditionalComposedRiskResult(
        guardrail_index=index,
        unconditional_risk=unconditional_risk,
        conditional_risk=conditional_risk,
        unconditional_effective_protection=1.0 - unconditional_risk,
        conditional_effective_protection=1.0 - conditional_risk,
        protection_drop=risk_lift,
        risk_lift=risk_lift,
        conditioning_probability=conditioning_probability,
        distribution=distribution.copy(),
        stress_result=stress_result,
    )


composed_risk_given_guardrail_x_failure = composed_risk_given_guardrail_X_failure


def _resolve_dependence(
    dependence: BaselineDependence | Mapping[str, object] | ArrayLike,
) -> _ResolvedDependence:
    if isinstance(dependence, BaselineDependence):
        return _resolve_from_parts(
            marginals=dependence.marginals,
            distribution=dependence.distribution,
            samples=dependence.samples,
            pairwise=dependence.pairwise,
            name=dependence.name,
        )

    if isinstance(dependence, Mapping):
        return _resolve_from_parts(
            marginals=cast(Sequence[float] | None, dependence.get("marginals")),
            distribution=cast(Sequence[float] | None, _first_present(
                dependence,
                ("distribution", "probabilities", "pmf"),
            )),
            samples=cast(ArrayLike | None, dependence.get("samples")),
            pairwise=cast(
                Iterable[PairwiseDependence | Mapping[str, object]] | None,
                dependence.get("pairwise"),
            ),
            name=str(dependence.get("name", "baseline")),
        )

    arr = np.asarray(dependence, dtype=float)
    if arr.ndim == 1:
        return _resolve_from_parts(distribution=arr, name="baseline")
    if arr.ndim == 2:
        return _resolve_from_parts(samples=arr, name="baseline")
    raise ValueError("baseline_dependence must be a distribution, samples, or mapping.")


def _resolve_from_parts(
    *,
    marginals: Sequence[float] | None = None,
    distribution: Sequence[float] | ArrayLike | None = None,
    samples: ArrayLike | None = None,
    pairwise: Iterable[PairwiseDependence | Mapping[str, object]] | None = None,
    name: str = "baseline",
) -> _ResolvedDependence:
    if distribution is not None and samples is not None:
        raise ValueError("Pass either distribution or samples, not both.")

    if distribution is not None:
        pmf, n_events = _as_distribution(distribution)
        observed_marginals, _ = distribution_moments(pmf, n_events)
        if marginals is not None:
            requested = _as_probability_vector(marginals)
            if requested.size != n_events:
                raise ValueError("marginals length must match distribution dimension.")
            if not np.allclose(requested, observed_marginals, atol=1.0e-8):
                raise ValueError("distribution marginals do not match supplied marginals.")
        return _ResolvedDependence(
            distribution=pmf,
            marginals=observed_marginals,
            n_events=n_events,
            name=name,
        )

    if samples is not None:
        pmf, n_events = _distribution_from_samples(samples)
        observed_marginals, _ = distribution_moments(pmf, n_events)
        if marginals is not None:
            requested = _as_probability_vector(marginals)
            if requested.size != n_events:
                raise ValueError("marginals length must match sample dimension.")
            if not np.allclose(requested, observed_marginals, atol=1.0e-8):
                raise ValueError("sample marginals do not match supplied marginals.")
        return _ResolvedDependence(
            distribution=pmf,
            marginals=observed_marginals,
            n_events=n_events,
            name=name,
        )

    if marginals is None:
        raise ValueError("A baseline distribution, samples, or marginals are required.")

    p = _as_probability_vector(marginals)
    pairwise_values = tuple(pairwise or ())
    if not pairwise_values:
        pmf = _independent_distribution(p)
    elif p.size == 2 and len(pairwise_values) == 1:
        pmf = _two_guardrail_distribution_from_pairwise(p, pairwise_values[0])
    else:
        raise ValueError(
            "Pairwise-only baselines do not identify a unique dependence law for "
            "more than two guardrails. Pass an atom distribution or binary samples."
        )
    return _ResolvedDependence(distribution=pmf, marginals=p, n_events=int(p.size), name=name)


def _coerce_budget(stress_budget: StressBudget | Mapping[str, object] | float) -> StressBudget:
    if isinstance(stress_budget, StressBudget):
        budget = stress_budget
    elif isinstance(stress_budget, Mapping):
        raw_amount = _first_present(stress_budget, ("amount", "budget", "epsilon"))
        if raw_amount is None:
            raise ValueError("stress_budget mapping must include amount, budget, or epsilon.")
        budget = StressBudget(
            amount=float(cast(float, raw_amount)),
            metric=_canonical_metric(str(stress_budget.get("metric", "wasserstein"))),
            event=_canonical_event(str(stress_budget.get("event", "and"))),
            kl_smoothing=float(cast(Any, stress_budget.get("kl_smoothing", _DEFAULT_KL_SMOOTHING))),
            solver_tol=float(cast(Any, stress_budget.get("solver_tol", _TOL))),
        )
    else:
        budget = StressBudget(amount=float(stress_budget))

    if not np.isfinite(budget.amount) and not isinf(budget.amount):
        raise ValueError("stress budget amount must be finite or infinity.")
    if budget.amount < 0.0:
        raise ValueError("stress budget amount must be nonnegative.")
    if budget.kl_smoothing < 0.0 or budget.kl_smoothing >= 1.0:
        raise ValueError("kl_smoothing must lie in [0, 1).")
    if budget.solver_tol <= 0.0:
        raise ValueError("solver_tol must be positive.")
    return StressBudget(
        amount=float(budget.amount),
        metric=_canonical_metric(budget.metric),
        event=_canonical_event(budget.event),
        kl_smoothing=float(budget.kl_smoothing),
        solver_tol=float(budget.solver_tol),
    )


def _solve_wasserstein_stress(
    baseline: FloatArray,
    marginals: FloatArray,
    states: FloatArray,
    objective: FloatArray,
    budget: StressBudget,
) -> tuple[FloatArray, float, str]:
    n_atoms = int(baseline.size)
    cost = _normalized_hamming_cost(states)
    objective_over_transport = np.tile(objective, n_atoms)
    c = -objective_over_transport

    a_eq_rows: list[FloatArray] = []
    b_eq: list[float] = []
    for source in range(n_atoms):
        row = np.zeros(n_atoms * n_atoms, dtype=float)
        row[source * n_atoms : (source + 1) * n_atoms] = 1.0
        a_eq_rows.append(row)
        b_eq.append(float(baseline[source]))

    for event_index in range(states.shape[1]):
        row = np.tile(states[:, event_index], n_atoms)
        a_eq_rows.append(row.astype(float))
        b_eq.append(float(marginals[event_index]))

    res = linprog(
        c,
        A_ub=cost.reshape(1, -1),
        b_ub=np.asarray([budget.amount], dtype=float),
        A_eq=np.vstack(a_eq_rows),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=(0.0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Wasserstein stress LP failed: {res.message}")

    transport = np.asarray(res.x, dtype=float).reshape(n_atoms, n_atoms)
    stressed = np.clip(np.sum(transport, axis=0), 0.0, 1.0)
    stressed = stressed / float(np.sum(stressed))
    realized = max(0.0, float(cost.reshape(-1) @ np.asarray(res.x, dtype=float)))
    return stressed, realized, "Solved fixed-marginal Wasserstein stress LP."


def _solve_kl_stress(
    baseline: FloatArray,
    marginals: FloatArray,
    states: FloatArray,
    objective: FloatArray,
    budget: StressBudget,
) -> tuple[FloatArray, float, str]:
    a_eq = np.vstack([np.ones(baseline.size, dtype=float), states.T])
    b_eq = np.concatenate([np.asarray([1.0], dtype=float), marginals])

    def fun(q: FloatArray) -> float:
        return -float(objective @ q)

    def jac(_q: FloatArray) -> FloatArray:
        return -objective

    def eq_fun(q: FloatArray) -> FloatArray:
        return cast(FloatArray, a_eq @ q - b_eq)

    def eq_jac(_q: FloatArray) -> FloatArray:
        return a_eq

    def ineq_fun(q: FloatArray) -> float:
        return float(budget.amount - _regularized_kl(q, baseline, budget.kl_smoothing))

    def ineq_jac(q: FloatArray) -> FloatArray:
        return -_regularized_kl_grad(q, baseline, budget.kl_smoothing)

    result = minimize(
        fun,
        baseline.copy(),
        jac=jac,
        bounds=[(0.0, 1.0)] * baseline.size,
        constraints=[
            {"type": "eq", "fun": eq_fun, "jac": eq_jac},
            {"type": "ineq", "fun": ineq_fun, "jac": ineq_jac},
        ],
        method="SLSQP",
        options={"ftol": budget.solver_tol, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"KL stress optimization failed: {result.message}")

    stressed = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    stressed = stressed / float(np.sum(stressed))
    realized = _regularized_kl(stressed, baseline, budget.kl_smoothing)
    if realized > budget.amount + max(1.0e-7, 10.0 * budget.solver_tol):
        raise RuntimeError(
            "KL stress optimization returned a distribution outside the requested budget: "
            f"{realized} > {budget.amount}."
        )
    return stressed, realized, "Solved fixed-marginal KL stress optimization."


def _distance_between(
    baseline: FloatArray,
    target: FloatArray,
    states: FloatArray,
    budget: StressBudget,
) -> float:
    if budget.metric == "kl":
        return _regularized_kl(target, baseline, budget.kl_smoothing)
    return _wasserstein_distance(baseline, target, states)


def _wasserstein_distance(source: FloatArray, target: FloatArray, states: FloatArray) -> float:
    n_atoms = int(source.size)
    cost = _normalized_hamming_cost(states)
    a_eq_rows: list[FloatArray] = []
    b_eq: list[float] = []
    for row_index in range(n_atoms):
        row = np.zeros(n_atoms * n_atoms, dtype=float)
        row[row_index * n_atoms : (row_index + 1) * n_atoms] = 1.0
        a_eq_rows.append(row)
        b_eq.append(float(source[row_index]))
    for col_index in range(n_atoms):
        row = np.zeros(n_atoms * n_atoms, dtype=float)
        row[col_index::n_atoms] = 1.0
        a_eq_rows.append(row)
        b_eq.append(float(target[col_index]))

    res = linprog(
        cost.reshape(-1),
        A_eq=np.vstack(a_eq_rows),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=(0.0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Wasserstein distance LP failed: {res.message}")
    return max(0.0, float(cost.reshape(-1) @ np.asarray(res.x, dtype=float)))


def _regularized_kl(q: FloatArray, baseline: FloatArray, smoothing: float) -> float:
    q_arr = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    base = np.clip(np.asarray(baseline, dtype=float), 0.0, 1.0)
    uniform = np.full_like(base, 1.0 / base.size)
    if smoothing > 0.0:
        q_arr = (1.0 - smoothing) * q_arr + smoothing * uniform
        base = (1.0 - smoothing) * base + smoothing * uniform
    mask = q_arr > 0.0
    return max(0.0, float(np.sum(q_arr[mask] * np.log(q_arr[mask] / base[mask]))))


def _regularized_kl_grad(q: FloatArray, baseline: FloatArray, smoothing: float) -> FloatArray:
    q_arr = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    base = np.clip(np.asarray(baseline, dtype=float), 0.0, 1.0)
    uniform = np.full_like(base, 1.0 / base.size)
    scale = 1.0
    if smoothing > 0.0:
        scale = 1.0 - smoothing
        q_arr = scale * q_arr + smoothing * uniform
        base = scale * base + smoothing * uniform
    q_arr = np.maximum(q_arr, np.finfo(float).tiny)
    return cast(FloatArray, scale * (np.log(q_arr / base) + 1.0))


def _normalized_hamming_cost(states: FloatArray) -> FloatArray:
    return np.mean(np.abs(states[:, None, :] - states[None, :, :]), axis=2)


def _event_vector(states: FloatArray, event: StressEvent) -> FloatArray:
    if event == "and":
        return cast(FloatArray, np.all(states == 1.0, axis=1).astype(float))
    return cast(FloatArray, np.any(states == 1.0, axis=1).astype(float))


def _distribution_from_samples(samples: ArrayLike) -> tuple[FloatArray, int]:
    arr = np.asarray(samples, dtype=int)
    if arr.ndim != 2:
        raise ValueError("samples must be a two-dimensional binary matrix.")
    if arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError("samples must contain at least one row and two guardrails.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("samples must contain only 0/1 failure indicators.")
    n_events = int(arr.shape[1])
    shifts = (1 << np.arange(n_events, dtype=np.int64)).astype(np.int64)
    atom_ids = arr.astype(np.int64) @ shifts
    counts = np.bincount(atom_ids, minlength=1 << n_events).astype(float)
    distribution = counts / float(np.sum(counts))
    return distribution.astype(np.float64, copy=False), n_events


def _as_distribution(distribution: ArrayLike) -> tuple[FloatArray, int]:
    pmf = np.asarray(distribution, dtype=float)
    if pmf.ndim != 1:
        raise ValueError("distribution must be a one-dimensional atom probability vector.")
    if pmf.size == 0 or pmf.size & (pmf.size - 1):
        raise ValueError("distribution length must be a positive power of two.")
    if not np.all(np.isfinite(pmf)):
        raise ValueError("distribution must contain finite values.")
    if np.any(pmf < -_TOL):
        raise ValueError("distribution entries must be nonnegative.")
    total = float(np.sum(pmf))
    if not isclose(total, 1.0, abs_tol=1.0e-8):
        raise ValueError(f"distribution must sum to 1. Got {total}.")
    n_events = int(log(pmf.size, 2))
    if n_events < 2:
        raise ValueError("distribution must describe at least two guardrails.")
    pmf = np.clip(pmf, 0.0, 1.0)
    pmf = pmf / float(np.sum(pmf))
    return cast(FloatArray, pmf.astype(float, copy=False)), n_events


def _as_probability_vector(values: Sequence[float]) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("marginals must be one-dimensional.")
    if arr.size < 2:
        raise ValueError("at least two guardrail marginals are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("marginals must contain finite values.")
    if np.any(arr < -_TOL) or np.any(arr > 1.0 + _TOL):
        raise ValueError("marginals must lie in [0, 1].")
    return cast(FloatArray, np.clip(arr, 0.0, 1.0).astype(float, copy=False))


def _independent_distribution(marginals: FloatArray) -> FloatArray:
    states = atom_matrix(int(marginals.size)).astype(float)
    probs = np.ones(states.shape[0], dtype=float)
    for idx, p_i in enumerate(marginals):
        probs *= np.where(states[:, idx] == 1.0, p_i, 1.0 - p_i)
    probs = probs / float(np.sum(probs))
    return cast(FloatArray, probs)


def _two_guardrail_distribution_from_pairwise(
    marginals: FloatArray,
    pairwise: PairwiseDependence | Mapping[str, object],
) -> FloatArray:
    dep = _coerce_pairwise(pairwise)
    pair = {dep.i, dep.j}
    if pair != {0, 1}:
        raise ValueError("two-guardrail pairwise baseline must refer to indices 0 and 1.")
    joint = dependence_to_joint_probability(
        float(marginals[0]),
        float(marginals[1]),
        dep.value,
        dep.kind,
    )
    p0, p1 = float(marginals[0]), float(marginals[1])
    pmf = np.asarray(
        [
            1.0 - p0 - p1 + joint,
            p0 - joint,
            p1 - joint,
            joint,
        ],
        dtype=float,
    )
    if np.any(pmf < -1.0e-8):
        raise ValueError("pairwise dependence implies a negative atom probability.")
    pmf = np.clip(pmf, 0.0, 1.0)
    return cast(FloatArray, pmf / float(np.sum(pmf)))


def _coerce_pairwise(pairwise: PairwiseDependence | Mapping[str, object]) -> PairwiseDependence:
    if isinstance(pairwise, PairwiseDependence):
        return pairwise
    try:
        return PairwiseDependence(
            i=int(cast(int, pairwise["i"])),
            j=int(cast(int, pairwise["j"])),
            kind=cast(Any, pairwise.get("kind", "joint_probability")),
            value=float(cast(float, pairwise["value"])),
        )
    except KeyError as exc:
        raise ValueError("pairwise mapping must include i, j, and value.") from exc


def _canonical_metric(metric: str) -> StressMetric:
    normalized = metric.strip().lower().replace("-", "_")
    if normalized in {"wasserstein", "wasserstein_1", "w1", "earth_mover", "emd"}:
        return "wasserstein"
    if normalized in {"kl", "relative_entropy", "relative_entropic", "entropy"}:
        return "kl"
    raise ValueError('stress metric must be "wasserstein" or "kl".')


def _canonical_event(event: str) -> StressEvent:
    normalized = event.strip().lower()
    if normalized == "and":
        return "and"
    if normalized == "or":
        return "or"
    raise ValueError('stress event must be "and" or "or".')


def _first_present(mapping: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
