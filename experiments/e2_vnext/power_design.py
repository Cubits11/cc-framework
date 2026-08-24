#!/usr/bin/env python3
"""Pre-outcome sample-size calculations for the paired E2-VNEXT estimand.

This module deliberately has no model, dataset, or network dependency.  It
derives a completed-pair requirement from the precision target for
``Delta = p11 - pA * pB`` rather than borrowing the old field-test count.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from statistics import NormalDist


DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.95
DEFAULT_DELTA_RESOLUTION = 0.02


@dataclass(frozen=True)
class PowerPoint:
    """A feasible paired-Bernoulli design point."""

    p_a: float
    p_b: float
    delta: float
    variance_null: float
    variance_alternative: float
    required_complete_pairs: int


def feasible_delta_bounds(p_a: float, p_b: float) -> tuple[float, float]:
    """Return the sharp feasible interval for Delta at fixed Bernoulli margins."""

    _validate_probability(p_a, "p_a")
    _validate_probability(p_b, "p_b")
    product = p_a * p_b
    return max(0.0, p_a + p_b - 1.0) - product, min(p_a, p_b) - product


def delta_is_feasible(p_a: float, p_b: float, delta: float, *, tolerance: float = 1e-12) -> bool:
    """Whether the four implied paired-Bernoulli cells are nonnegative."""

    lower, upper = feasible_delta_bounds(p_a, p_b)
    return lower - tolerance <= delta <= upper + tolerance


def delta_variance(p_a: float, p_b: float, delta: float) -> float:
    """Delta-method variance of the bias-corrected paired Delta estimator.

    For paired Bernoulli unsafe-pass indicators A and B,

        Delta = P(A=1,B=1) - P(A=1)P(B=1).

    The influence function is ``(A-p_a)(B-p_b)-Delta``.  Its variance is the
    expression below.  A non-feasible input is rejected rather than silently
    yielding a numerical value for impossible cell probabilities.
    """

    if not delta_is_feasible(p_a, p_b, delta):
        raise ValueError("p_a, p_b, and delta imply an infeasible 2x2 table")
    variance = (
        p_a * p_b * (1.0 - p_a) * (1.0 - p_b)
        + (1.0 - 2.0 * p_a) * (1.0 - 2.0 * p_b) * delta
        - delta * delta
    )
    if variance < -1e-12:
        raise ValueError("negative variance outside numerical tolerance")
    return max(0.0, variance)


def required_complete_pairs(
    p_a: float,
    p_b: float,
    delta_alternative: float,
    *,
    delta_null: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> PowerPoint:
    """Return the normal-approximation completed-pair requirement.

    The calculation is for one prespecified pair and a two-sided test of the
    point null ``delta_null``.  It is a planning approximation, not an
    after-the-fact confidence procedure.  The caller must separately freeze
    the target population and whether it is a sampled frame or a census.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if math.isclose(delta_alternative, delta_null, abs_tol=0.0):
        raise ValueError("alternative and null Delta values must differ")
    variance_null = delta_variance(p_a, p_b, delta_null)
    variance_alternative = delta_variance(p_a, p_b, delta_alternative)
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    separation = abs(delta_alternative - delta_null)
    requirement = math.ceil(
        ((z_alpha * math.sqrt(variance_null) + z_power * math.sqrt(variance_alternative)) / separation)
        ** 2
    )
    return PowerPoint(
        p_a=p_a,
        p_b=p_b,
        delta=delta_alternative,
        variance_null=variance_null,
        variance_alternative=variance_alternative,
        required_complete_pairs=requirement,
    )


def worst_case_requirement(
    *,
    delta_resolution: float = DEFAULT_DELTA_RESOLUTION,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    grid_steps: int = 1_000,
) -> PowerPoint:
    """Maximise the predeclared Delta requirement over feasible margins.

    Both signs are searched because a product baseline can be optimistic or
    conservative.  The grid is an auditable conservative design check; the
    protocol also records the analytic central-margin value, which is the
    maximiser to the precision needed for the selected integer requirement.
    """

    if not 1 <= grid_steps:
        raise ValueError("grid_steps must be positive")
    if not 0.0 < delta_resolution < 1.0:
        raise ValueError("delta_resolution must be in (0, 1)")
    winner: PowerPoint | None = None
    for a_index in range(grid_steps + 1):
        p_a = a_index / grid_steps
        for b_index in range(grid_steps + 1):
            p_b = b_index / grid_steps
            for sign in (-1.0, 1.0):
                delta = sign * delta_resolution
                if not delta_is_feasible(p_a, p_b, delta):
                    continue
                point = required_complete_pairs(
                    p_a,
                    p_b,
                    delta,
                    alpha=alpha,
                    power=power,
                )
                if winner is None or point.required_complete_pairs > winner.required_complete_pairs:
                    winner = point
    if winner is None:
        raise ValueError("no feasible design point found")
    return winner


def raw_delta_from_counts(n_11: int, n_10: int, n_01: int, n_00: int) -> float:
    """Return the literal finite-frame plug-in Delta for a complete 2x2 table."""

    counts = (n_11, n_10, n_01, n_00)
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("cell counts must be nonnegative integers")
    total = sum(counts)
    if total == 0:
        raise ValueError("at least one paired observation is required")
    p_a = (n_11 + n_10) / total
    p_b = (n_11 + n_01) / total
    p_11 = n_11 / total
    return p_11 - p_a * p_b


def bias_corrected_delta(raw_delta: float, completed_pairs: int) -> float:
    """Correct the plug-in estimator's iid finite-sample attenuation.

    This correction is only for sampling/superpopulation analysis.  The raw
    value remains the correct value when all records in a finite target frame
    have been scored.
    """

    if completed_pairs < 2:
        raise ValueError("at least two complete pairs are required")
    return raw_delta * completed_pairs / (completed_pairs - 1)


def _validate_probability(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")


def _report(point: PowerPoint, *, delta_resolution: float, alpha: float, power: float) -> dict[str, object]:
    return {
        "alpha_two_sided": alpha,
        "delta_resolution": delta_resolution,
        "design_alternative_delta": point.delta,
        "design_margins": {"p_a": point.p_a, "p_b": point.p_b},
        "power": power,
        "required_complete_pairs": point.required_complete_pairs,
        "variance_alternative": point.variance_alternative,
        "variance_null": point.variance_null,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-steps", type=int, default=1_000)
    parser.add_argument("--power", type=float, default=DEFAULT_POWER)
    parser.add_argument("--delta-resolution", type=float, default=DEFAULT_DELTA_RESOLUTION)
    args = parser.parse_args()
    point = worst_case_requirement(
        delta_resolution=args.delta_resolution,
        power=args.power,
        grid_steps=args.grid_steps,
    )
    print(
        json.dumps(
            _report(
                point,
                delta_resolution=args.delta_resolution,
                alpha=DEFAULT_ALPHA,
                power=args.power,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
