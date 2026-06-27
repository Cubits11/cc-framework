"""Formal estimand-layer metrics for guardrail failure composition.

Convention
----------
``Z_i = 1`` means guardrail failure or unsafe pass.  The functions in this
module do not solve identification problems; the atom LP in
``cc.kernel.sensitivity`` remains the source of truth for sharp bounds and
witness distributions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from cc.kernel.sensitivity import LinearQuery, enumerate_atoms

FloatArray: TypeAlias = NDArray[np.float64]
RateInput: TypeAlias = Mapping[str, float] | Sequence[float]

DEFAULT_TOL = 1.0e-9
DEFAULT_EPS = 1.0e-12

__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_TOL",
    "MetricDomainError",
    "cc_gain",
    "cc_shift",
    "fh_position",
    "fh_width",
    "independence_regret",
    "independent_event_probability",
]


class MetricDomainError(ValueError):
    """Raised when a metric is undefined for the supplied estimand domain."""


def fh_width(lower: float, upper: float, *, tol: float = DEFAULT_TOL) -> float:
    """Return the identified interval width ``U_phi - L_phi``.

    Tiny negative widths within ``tol`` are treated as numerical zero.  Bounds
    must be finite probabilities.
    """

    lo, hi, _tolerance = _validated_bounds(lower, upper, tol=tol)
    if hi < lo:
        return 0.0
    return float(hi - lo)


def fh_position(
    observed: float,
    lower: float,
    upper: float,
    *,
    tol: float = DEFAULT_TOL,
) -> float | None:
    """Return the observed risk's normalized position inside ``[L_phi, U_phi]``.

    ``None`` means the identified interval is degenerate.  Values that sit just
    outside an endpoint by at most ``tol`` are snapped to that endpoint; values
    farther outside the feasible interval raise ``MetricDomainError``.
    """

    obs = _validate_probability("observed", observed)
    lo, hi, tolerance = _validated_bounds(lower, upper, tol=tol)
    if abs(hi - lo) <= tolerance:
        return None
    if obs < lo - tolerance or obs > hi + tolerance:
        raise MetricDomainError("observed must lie inside the identified interval")
    if obs < lo:
        obs = lo
    elif obs > hi:
        obs = hi
    position = (obs - lo) / (hi - lo)
    return float(min(1.0, max(0.0, position)))


def independent_event_probability(
    marginals: Mapping[str, float],
    query: LinearQuery,
    *,
    labels: Sequence[str],
) -> float:
    """Evaluate ``query`` under the product coupling of singleton marginals.

    The caller must supply the label order used by the query's atom ordering.
    This function never parses Boolean strings, evaluates expressions, or
    infers hidden success/failure inversions.
    """

    label_names = _validate_labels(labels)
    _validate_marginal_keys(marginals, label_names)
    n_atoms = 1 << len(label_names)
    if query.coefficients.size != n_atoms:
        raise MetricDomainError(
            f"query dimension must equal 2**len(labels) ({n_atoms}); "
            f"got {query.coefficients.size}"
        )
    _validate_event_coefficients(query.coefficients)

    atoms = enumerate_atoms(label_names)
    probabilities = np.ones(n_atoms, dtype=np.float64)
    for col, label in enumerate(label_names):
        marginal = _validate_probability(f"marginals[{label!r}]", marginals[label])
        probabilities *= np.where(atoms[:, col] == 1, marginal, 1.0 - marginal)

    value = float(query.coefficients @ probabilities)
    return _snap_probability("independent event probability", value)


def independence_regret(observed: float, independent: float) -> float:
    """Return ``observed - independent`` for a composition event probability."""

    obs = _validate_probability("observed", observed)
    ind = _validate_probability("independent", independent)
    return float(obs - ind)


def cc_gain(
    composition_risk: float,
    singleton_failures: RateInput,
    *,
    eps: float = DEFAULT_EPS,
) -> float | None:
    """Return composition risk divided by the largest singleton failure risk.

    This is a one-world, operator-relative normalization.  It is not a causal
    effect, not a universal performance gain, and not meaningful without the
    composition operator that defines ``composition_risk``.
    """

    risk = _validate_probability("composition_risk", composition_risk)
    epsilon = _validate_positive("eps", eps)
    values = _rate_values("singleton_failures", singleton_failures)
    denom = max(values)
    if denom <= epsilon:
        return None
    return float(risk / denom)


def cc_shift(
    composition_baseline: float,
    composition_deployed: float,
    singleton_baseline: RateInput,
    singleton_deployed: RateInput,
    *,
    eps: float = DEFAULT_EPS,
) -> float | None:
    """Return composed failure movement per largest singleton movement.

    The numerator is ``composition_deployed - composition_baseline``.  The
    denominator is the largest absolute singleton failure-rate movement, so the
    scale is stable even when singleton risks move in opposite directions.
    """

    baseline = _validate_probability("composition_baseline", composition_baseline)
    deployed = _validate_probability("composition_deployed", composition_deployed)
    epsilon = _validate_positive("eps", eps)
    singleton_shifts = _singleton_shifts(singleton_baseline, singleton_deployed)
    denom = max(abs(value) for value in singleton_shifts)
    if denom <= epsilon:
        return None
    return float((deployed - baseline) / denom)


def _validated_bounds(lower: float, upper: float, *, tol: float) -> tuple[float, float, float]:
    tolerance = _validate_positive("tol", tol)
    lo = _validate_probability("lower", lower)
    hi = _validate_probability("upper", upper)
    if lo > hi + tolerance:
        raise MetricDomainError("lower must not exceed upper beyond tolerance")
    return lo, hi, tolerance


def _validate_finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise MetricDomainError(f"{name} must be a finite real number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricDomainError(f"{name} must be a finite real number") from exc
    if not math.isfinite(out):
        raise MetricDomainError(f"{name} must be finite")
    return out


def _validate_probability(name: str, value: object) -> float:
    out = _validate_finite(name, value)
    if out < 0.0 or out > 1.0:
        raise MetricDomainError(f"{name} must lie in [0, 1]")
    return out


def _validate_positive(name: str, value: object) -> float:
    out = _validate_finite(name, value)
    if out <= 0.0:
        raise MetricDomainError(f"{name} must be positive")
    return out


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, str):
        raise MetricDomainError("labels must be a sequence of names, not a string")
    label_names = tuple(labels)
    if not label_names:
        raise MetricDomainError("labels must be nonempty")
    for label in label_names:
        if not isinstance(label, str) or not label.strip():
            raise MetricDomainError("labels must contain nonempty strings")
    if len(set(label_names)) != len(label_names):
        raise MetricDomainError("labels must be unique")
    return label_names


def _validate_marginal_keys(marginals: Mapping[str, float], labels: tuple[str, ...]) -> None:
    if not isinstance(marginals, Mapping) or not marginals:
        raise MetricDomainError("marginals must be a nonempty mapping")
    for key in marginals:
        if not isinstance(key, str) or not key.strip():
            raise MetricDomainError("marginal keys must be nonempty strings")
    if set(marginals.keys()) != set(labels):
        raise MetricDomainError("marginal keys must exactly match labels")


def _validate_event_coefficients(coefficients: FloatArray) -> None:
    if np.any(coefficients < -DEFAULT_TOL) or np.any(coefficients > 1.0 + DEFAULT_TOL):
        raise MetricDomainError("query coefficients must define event weights in [0, 1]")


def _snap_probability(name: str, value: float) -> float:
    if value < -DEFAULT_TOL or value > 1.0 + DEFAULT_TOL:
        raise MetricDomainError(f"{name} must lie in [0, 1]")
    return float(min(1.0, max(0.0, value)))


def _rate_values(name: str, rates: RateInput) -> tuple[float, ...]:
    if isinstance(rates, Mapping):
        return _mapping_rate_values(name, rates)
    if isinstance(rates, (str, bytes)):
        raise MetricDomainError(f"{name} must not be a string")
    values = tuple(
        _validate_probability(f"{name}[{index}]", value)
        for index, value in enumerate(rates)
    )
    if not values:
        raise MetricDomainError(f"{name} must be nonempty")
    return values


def _mapping_rate_values(name: str, rates: Mapping[str, float]) -> tuple[float, ...]:
    if not rates:
        raise MetricDomainError(f"{name} must be nonempty")
    values: list[float] = []
    for key, value in rates.items():
        if not isinstance(key, str) or not key.strip():
            raise MetricDomainError(f"{name} keys must be nonempty strings")
        values.append(_validate_probability(f"{name}[{key!r}]", value))
    return tuple(values)


def _singleton_shifts(baseline: RateInput, deployed: RateInput) -> tuple[float, ...]:
    if isinstance(baseline, Mapping) or isinstance(deployed, Mapping):
        if not isinstance(baseline, Mapping) or not isinstance(deployed, Mapping):
            raise MetricDomainError("singleton_baseline and singleton_deployed must use the same shape")
        return _mapping_singleton_shifts(baseline, deployed)
    if isinstance(baseline, (str, bytes)) or isinstance(deployed, (str, bytes)):
        raise MetricDomainError("singleton rates must not be strings")

    base_values = tuple(
        _validate_probability(f"singleton_baseline[{index}]", value)
        for index, value in enumerate(baseline)
    )
    deployed_values = tuple(
        _validate_probability(f"singleton_deployed[{index}]", value)
        for index, value in enumerate(deployed)
    )
    if not base_values or not deployed_values:
        raise MetricDomainError("singleton rates must be nonempty")
    if len(base_values) != len(deployed_values):
        raise MetricDomainError("singleton rate sequences must have equal length")
    return tuple(right - left for left, right in zip(base_values, deployed_values, strict=True))


def _mapping_singleton_shifts(
    baseline: Mapping[str, float],
    deployed: Mapping[str, float],
) -> tuple[float, ...]:
    if not baseline or not deployed:
        raise MetricDomainError("singleton rate mappings must be nonempty")
    for source_name, source in (
        ("singleton_baseline", baseline),
        ("singleton_deployed", deployed),
    ):
        for key in source:
            if not isinstance(key, str) or not key.strip():
                raise MetricDomainError(f"{source_name} keys must be nonempty strings")
    if set(baseline.keys()) != set(deployed.keys()):
        raise MetricDomainError("singleton rate mappings must have matching keys")
    return tuple(
        _validate_probability(f"singleton_deployed[{key!r}]", deployed[key])
        - _validate_probability(f"singleton_baseline[{key!r}]", baseline[key])
        for key in sorted(baseline)
    )
