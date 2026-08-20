"""Composition bounds over named binary events.

This module is the answer to a rejected-adoption report. A downstream consumer
read ``cc.core.composition_theory``, found that it operates on ROC point sets
and bounds the Youden J statistic, and concluded -- correctly -- that a
deterministic refusal rule has no threshold, no operating point, and no
false-positive rate to trade against. They reimplemented the inequality by hand
rather than force their controls into a detector shape.

Nothing in this module's public surface mentions ROC curves, thresholds,
operating points, TPR, or FPR. A binary event is a name and a marginal
probability. Detectors reach this surface through
:func:`cc.compose.marginal_from_operating_point`, which converts an operating
point into a marginal -- so the detector framing sits *above* the event
calculus rather than underneath it.

The mathematics is classical. Frechet-Hoeffding is 1935 and this module claims
no novelty for it. What it provides is a surface shaped like the way composed
binary events are actually reasoned about, and a result object that carries the
independence baseline, the binding event, and the non-claims alongside the
interval so none of them can be quoted apart from the others.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = [
    "CompositionBounds",
    "CountermonotoneUndefinedError",
    "DependenceAssumption",
    "EventKind",
    "MarginalProvenance",
    "SensitivityRow",
    "compose_bounds",
    "marginal_from_operating_point",
    "sensitivity",
]

#: ``"all"`` bounds P(every event occurs) -- a conjunction. ``"any"`` bounds
#: P(at least one occurs) -- a union. The aliases ``"and"``/``"or"`` are
#: accepted because downstream documents use that vocabulary.
EventKind: TypeAlias = Literal["all", "any"]

#: What is assumed about the joint dependence structure.
#:
#: ``"unconstrained"``
#:     Nothing is assumed. Returns the sharp Frechet-Hoeffding interval. This
#:     is the honest default when no dependence evidence exists.
#: ``"independent"``
#:     A point value, returned as a degenerate interval. Reported as a
#:     *baseline for comparison*, never as an answer -- see the module note on
#:     ``independence_point``.
#: ``"comonotone"``
#:     The upper Frechet corner, returned as a degenerate interval.
#: ``"countermonotone"``
#:     Defined only for exactly two events. Refused for more; see
#:     :class:`CountermonotoneUndefinedError`.
DependenceAssumption: TypeAlias = Literal[
    "unconstrained", "independent", "comonotone", "countermonotone"
]

#: Where each marginal came from. This is a producer declaration, not a
#: statistical test, and it is carried on the result so a number cannot be
#: quoted without it.
MarginalProvenance: TypeAlias = Literal["measured", "assumed", "supplied"]

_ALIASES: dict[str, EventKind] = {
    "all": "all",
    "and": "all",
    "AND": "all",
    "intersection": "all",
    "any": "any",
    "or": "any",
    "OR": "any",
    "union": "any",
}

_TOL = 1.0e-12

_NON_CLAIMS: tuple[str, ...] = (
    "A bound is not a measurement. It states what the supplied marginals "
    "permit under unknown dependence, and nothing else.",
    "These bounds do not identify a copula, establish independence, infer "
    "causation, or validate stationarity.",
    "A narrow interval can result from extreme marginals and is not evidence "
    "that the upstream measurements are valid.",
    "The independence point value is reported as a comparison baseline. It is "
    "not an estimate of the joint probability and must not be quoted as one.",
    "This is not a safety certification, a compliance statement, or a claim "
    "about any deployed system.",
)


class CountermonotoneUndefinedError(ValueError):
    """Raised when countermonotonicity is requested for other than two events.

    Countermonotonicity is a strictly bivariate concept. Two events can be
    perfectly negatively dependent -- one occurs exactly when the other does
    not -- but three cannot all be pairwise mutually exclusive and exhaustive
    in that way, and there is no n-dimensional countermonotonic structure for
    ``n > 2``.

    The matching fact about the bound: the Frechet-Hoeffding lower bound
    ``max(0, sum(p) - (n-1))`` **is not a copula in dimension >= 3**. It
    remains *pointwise sharp* -- for any fixed marginals some joint
    distribution attains it -- but no single dependence structure attains it
    everywhere, so there is no countermonotone regime to tabulate beside
    independence and comonotonicity.

    This error exists because silently offering a "countermonotone" option for
    four events would produce a number with the form of a dependence regime and
    none of the content.
    """


@dataclass(frozen=True)
class SensitivityRow:
    """How the interval moves when one event's marginal is perturbed."""

    event: str
    marginal: float
    perturbed_marginal: float
    lower_delta: float
    upper_delta: float

    @property
    def moves_upper(self) -> bool:
        """Whether this event moves the upper bound at all."""
        return abs(self.upper_delta) > _TOL

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "marginal": self.marginal,
            "perturbed_marginal": self.perturbed_marginal,
            "lower_delta": self.lower_delta,
            "upper_delta": self.upper_delta,
            "moves_upper": self.moves_upper,
        }


@dataclass(frozen=True)
class CompositionBounds:
    """A sharp interval for a composed binary event, with its context attached.

    The interval is the result. Everything else on this object exists so the
    interval cannot be quoted without the things that qualify it: what was
    assumed, which event the bound turns on, and what the number does not mean.
    """

    lower: float
    upper: float
    event: EventKind
    marginals: Mapping[str, float]
    dependence: DependenceAssumption
    #: The product of the marginals. A comparison baseline, never an answer.
    independence_point: float
    #: Name of the event whose marginal the upper bound turns on, when the
    #: bound is determined by a single event. ``None`` when the upper bound is
    #: a sum or a clip rather than a single marginal.
    binding_event: str | None
    marginal_provenance: MarginalProvenance
    non_claims: tuple[str, ...] = field(default=_NON_CLAIMS)

    @property
    def width(self) -> float:
        """``upper - lower``, with numerical negatives clipped to zero."""
        return max(0.0, self.upper - self.lower)

    @property
    def independence_regret(self) -> float:
        """How much the independence baseline understates the admissible worst case.

        ``upper - independence_point``. This is the quantity that makes the
        product assumption's error visible in absolute terms.
        """
        return self.upper - self.independence_point

    @property
    def understatement_factor(self) -> float:
        """``upper / independence_point``, or infinity when the product is zero.

        The relative form of :attr:`independence_regret`. For a homogeneous
        stack of ``m`` events at rate ``p`` this grows like ``p**(1-m)``, which
        is the correlation cliff in one number.
        """
        if self.independence_point <= 0.0:
            return math.inf
        return self.upper / self.independence_point

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-native mapping. Used by the conformance corpus."""
        factor = self.understatement_factor
        return {
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "event": self.event,
            "marginals": dict(self.marginals),
            "dependence": self.dependence,
            "independence_point": self.independence_point,
            "independence_regret": self.independence_regret,
            # JSON has no infinity. A null here means the product was zero, so
            # the ratio is undefined rather than large.
            "understatement_factor": None if math.isinf(factor) else factor,
            "binding_event": self.binding_event,
            "marginal_provenance": self.marginal_provenance,
            "non_claims": list(self.non_claims),
        }


def _canonical_event(event: str) -> EventKind:
    try:
        return _ALIASES[event]
    except KeyError:
        allowed = ", ".join(sorted(set(_ALIASES)))
        raise ValueError(f"unknown event kind {event!r}; expected one of: {allowed}") from None


def _validated_marginals(marginals: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(marginals, Mapping):
        raise TypeError(
            "marginals must be a mapping of event name to probability; "
            f"got {type(marginals).__name__}. A bare sequence is refused "
            "because an unnamed marginal cannot be reported against a "
            "binding event."
        )
    if not marginals:
        raise ValueError("marginals must name at least one event.")

    out: dict[str, float] = {}
    for name, value in marginals.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"event name must be a non-empty string; got {name!r}")
        try:
            p = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"marginal for {name!r} must be a real number; got {value!r}") from None
        if not math.isfinite(p):
            raise ValueError(f"marginal for {name!r} must be finite; got {p!r}")
        if not (-_TOL <= p <= 1.0 + _TOL):
            raise ValueError(f"marginal for {name!r} must lie in [0, 1]; got {p!r}")
        out[name] = min(1.0, max(0.0, p))
    return out


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _classical(values: Sequence[float], event: EventKind) -> tuple[float, float]:
    """Closed-form n-way Frechet-Hoeffding bounds."""
    n = len(values)
    total = math.fsum(values)
    if event == "all":
        return _clip01(max(0.0, total - (n - 1))), _clip01(min(values))
    return _clip01(max(values)), _clip01(min(1.0, total))


def _binding_event(names: Sequence[str], values: Sequence[float], event: EventKind) -> str | None:
    """Name the event the upper bound turns on, when a single event determines it.

    For a conjunction the upper bound is ``min(p)``, so the argmin binds and
    improving any other event moves the upper bound not at all -- which is the
    actionable finding. For a union the upper bound is ``min(1, sum(p))``,
    which is a sum rather than a single event, so no event binds unless the
    clip at 1 is inactive and only one event is present.
    """
    if event == "all":
        best = min(range(len(values)), key=lambda i: values[i])
        ties = [i for i, v in enumerate(values) if abs(v - values[best]) <= _TOL]
        # A tie means no single event binds: improving either leaves the bound
        # pinned by the other. Reporting one of them would be misleading.
        return names[best] if len(ties) == 1 else None
    if len(values) == 1:
        return names[0]
    return None


def compose_bounds(
    marginals: Mapping[str, float],
    *,
    event: str = "all",
    dependence: DependenceAssumption = "unconstrained",
    marginal_provenance: MarginalProvenance = "supplied",
) -> CompositionBounds:
    """Bound the probability of a composed binary event from its marginals.

    Each event is a named binary failure indicator with a marginal probability.
    No ROC curve, threshold, or operating point is involved: a deterministic
    refusal rule is as valid an input as a tuned classifier.

    Args:
        marginals: Mapping of event name to marginal probability in ``[0, 1]``.
            Names are required -- an unnamed marginal cannot be reported
            against a binding event.
        event: ``"all"`` for a conjunction, ``"any"`` for a union. ``"and"``
            and ``"or"`` are accepted aliases.
        dependence: What is assumed about the joint structure. The default,
            ``"unconstrained"``, assumes nothing and returns the sharp
            interval.
        marginal_provenance: Whether the marginals were ``"measured"``,
            ``"assumed"``, or merely ``"supplied"``. Carried on the result so a
            bound over assumed rates cannot be quoted as a measurement.

    Returns:
        A :class:`CompositionBounds` carrying the interval, the independence
        baseline, the binding event, and the non-claims.

    Raises:
        CountermonotoneUndefinedError: ``dependence="countermonotone"`` with
            more than two events.
        ValueError: A marginal outside ``[0, 1]``, an empty mapping, or an
            unknown event kind.

    Example:
        Three deterministic controls, each with an assumed evasion
        probability::

            >>> b = compose_bounds(
            ...     {"SELF_REPORTED": 0.30, "STALE": 0.01, "SEPARATION": 0.40},
            ...     event="all",
            ...     marginal_provenance="assumed",
            ... )
            >>> b.lower, b.upper
            (0.0, 0.01)
            >>> b.binding_event
            'STALE'

        The upper bound is ``min(p)``: under arbitrary dependence a conjunction
        of controls is no stronger than its single strongest member. Improving
        ``SEPARATION`` moves the upper bound not at all.

        When two events tie for the minimum, no single event binds and
        ``binding_event`` is ``None`` -- improving either one leaves the bound
        pinned by the other, so naming one of them would misdirect effort.
    """
    kind = _canonical_event(event)
    checked = _validated_marginals(marginals)
    names = tuple(checked)
    values = tuple(checked[n] for n in names)

    if dependence == "countermonotone" and len(values) != 2:
        if len(values) < 2:
            raise CountermonotoneUndefinedError(
                f"countermonotonicity is undefined for {len(values)} event(s). It "
                "is a relation between two events; with fewer than two there is "
                "nothing for an event to be countermonotone with. Use "
                "dependence='unconstrained'."
            )
        raise CountermonotoneUndefinedError(
            f"countermonotonicity is undefined for {len(values)} events. It is a "
            "strictly bivariate concept: there is no n-dimensional "
            "countermonotonic structure for n > 2, and the Frechet-Hoeffding "
            "lower bound is not a copula in dimension >= 3 (though it remains "
            "pointwise sharp). Use dependence='unconstrained' for the sharp "
            "interval, whose lower endpoint is that pointwise-sharp floor."
        )

    # The independence baseline is computed once and reported on every result,
    # whatever the dependence assumption, so the gap it understates stays
    # visible even when the caller asked for a different regime.
    independence_point = (
        math.prod(values) if kind == "all" else _clip01(1.0 - math.prod(1.0 - v for v in values))
    )
    lower, upper = _classical(values, kind)

    if dependence == "independent":
        lower = upper = independence_point
    elif dependence == "comonotone":
        lower = upper = _clip01(min(values)) if kind == "all" else _clip01(max(values))
    elif dependence == "countermonotone":
        # Exactly two events; every other count was refused above.
        lower = upper = (
            _clip01(max(0.0, values[0] + values[1] - 1.0))
            if kind == "all"
            else _clip01(min(1.0, values[0] + values[1]))
        )
    elif dependence != "unconstrained":
        raise ValueError(
            f"unknown dependence assumption {dependence!r}; expected one of: "
            "unconstrained, independent, comonotone, countermonotone"
        )

    binding = _binding_event(names, values, kind) if dependence == "unconstrained" else None

    return CompositionBounds(
        lower=lower,
        upper=upper,
        event=kind,
        marginals=checked,
        dependence=dependence,
        independence_point=independence_point,
        binding_event=binding,
        marginal_provenance=marginal_provenance,
    )


def sensitivity(
    marginals: Mapping[str, float],
    *,
    event: str = "all",
    delta: float = 0.05,
    direction: Literal["increase", "decrease"] = "decrease",
    marginal_provenance: MarginalProvenance = "supplied",
) -> tuple[SensitivityRow, ...]:
    """Perturb each marginal in turn and report how the interval moves.

    Under a conjunction the upper bound is ``min(p)``, so only the event
    holding the minimum can move it. Every other row will report an
    ``upper_delta`` of zero, and that zero is the finding: effort spent on a
    weak event buys nothing against the worst admissible dependence.

    Args:
        marginals: As :func:`compose_bounds`.
        event: As :func:`compose_bounds`.
        delta: Magnitude of the perturbation, in probability units.
        direction: ``"decrease"`` lowers each marginal (a control improving, if
            the marginal is an evasion probability); ``"increase"`` raises it.
        marginal_provenance: As :func:`compose_bounds`.

    Returns:
        One :class:`SensitivityRow` per event, in the order given.
    """
    if not (0.0 < delta <= 1.0):
        raise ValueError(f"delta must lie in (0, 1]; got {delta!r}")

    base = compose_bounds(marginals, event=event, marginal_provenance=marginal_provenance)
    checked = dict(base.marginals)
    sign = -1.0 if direction == "decrease" else 1.0

    rows: list[SensitivityRow] = []
    for name, value in checked.items():
        perturbed = _clip01(value + sign * delta)
        trial = dict(checked)
        trial[name] = perturbed
        after = compose_bounds(trial, event=event, marginal_provenance=marginal_provenance)
        rows.append(
            SensitivityRow(
                event=name,
                marginal=value,
                perturbed_marginal=perturbed,
                lower_delta=after.lower - base.lower,
                upper_delta=after.upper - base.upper,
            )
        )
    return tuple(rows)


def marginal_from_operating_point(
    *,
    false_negative_rate: float,
    prevalence: float = 1.0,
) -> float:
    """Convert a detector's operating point into an event marginal.

    This is the *only* bridge from detector vocabulary into this module, and it
    points inward: a classifier supplies its miss rate at a chosen threshold
    and receives a marginal. The ROC framing sits above the event calculus and
    is optional; nothing downstream of this function knows a threshold existed.

    Args:
        false_negative_rate: Probability the detector fails to flag a case that
            should be flagged, at the operating point actually deployed. For a
            guardrail this is the unsafe-pass rate.
        prevalence: Probability the case arises at all. Defaults to 1.0, which
            makes the marginal equal to the miss rate -- appropriate when the
            composition is conditioned on an attack already occurring.

    Returns:
        ``false_negative_rate * prevalence``, suitable as a
        :func:`compose_bounds` marginal.

    Note:
        Choosing the threshold is outside this function and outside this
        module. A marginal derived here inherits every assumption of the
        threshold choice, and is no more valid than that choice was.
    """
    for label, value in (
        ("false_negative_rate", false_negative_rate),
        ("prevalence", prevalence),
    ):
        p = float(value)
        if not math.isfinite(p) or not (-_TOL <= p <= 1.0 + _TOL):
            raise ValueError(f"{label} must be a finite probability in [0, 1]; got {value!r}")
    return _clip01(float(false_negative_rate) * float(prevalence))
