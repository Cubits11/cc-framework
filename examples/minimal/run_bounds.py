#!/usr/bin/env python
"""Minimal atom-LP bounds example for reviewer reproduction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cc.kernel.metrics import (
    fh_position,
    fh_width,
    independence_regret,
    independent_event_probability,
)
from cc.kernel.sensitivity import AssumptionSet, LinearQuery

FAILURE_EVENT_CONVENTION = "Z_i=1 denotes a guardrail failure or unsafe pass."
DEFAULT_TOL = 1.0e-8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observed",
        type=float,
        default=0.12,
        help="Optional observed AND composition failure probability.",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    payload = run_example(observed=args.observed)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def run_example(*, observed: float) -> dict[str, Any]:
    labels = ("input_filter", "semantic_judge")
    marginals = {"input_filter": 0.2, "semantic_judge": 0.35}
    assumptions = _exact_marginal_assumptions(labels, marginals)
    query = LinearQuery.intersection(labels, labels, name="AND_failure")
    result = assumptions.identify(query)
    independent = independent_event_probability(marginals, query, labels=labels)

    lower_ok = _verify_endpoint(
        result.lower_solution,
        assumptions=assumptions,
        query=query,
        expected=result.lower_bound,
    )
    upper_ok = _verify_endpoint(
        result.upper_solution,
        assumptions=assumptions,
        query=query,
        expected=result.upper_bound,
    )

    return {
        "failure_event_convention": FAILURE_EVENT_CONVENTION,
        "guardrails": list(labels),
        "declared_marginals": marginals,
        "query": {
            "name": query.name,
            "event": "and",
            "coefficients": [float(value) for value in query.coefficients],
        },
        "assumptions_hash": result.assumptions_hash,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "fh_width": fh_width(result.lower_bound, result.upper_bound),
        "observed": observed,
        "fh_position": fh_position(observed, result.lower_bound, result.upper_bound),
        "independent_baseline": independent,
        "independence_regret": independence_regret(observed, independent),
        "witnesses_verified": {
            "lower": lower_ok,
            "upper": upper_ok,
        },
    }


def _exact_marginal_assumptions(
    labels: Sequence[str],
    marginals: dict[str, float],
) -> AssumptionSet:
    assumptions = AssumptionSet.empty(labels)
    for label in labels:
        value = marginals[label]
        assumptions = assumptions.with_marginal_interval(label, value, value)
    return assumptions


def _verify_endpoint(
    distribution: np.ndarray[Any, Any],
    *,
    assumptions: AssumptionSet,
    query: LinearQuery,
    expected: float,
) -> bool:
    if distribution.shape != (1 << len(assumptions.guardrails),):
        return False
    if not np.all(np.isfinite(distribution)):
        return False
    if float(np.min(distribution)) < -DEFAULT_TOL:
        return False
    if abs(float(np.sum(distribution)) - 1.0) > DEFAULT_TOL:
        return False
    for constraint in assumptions.constraints:
        lhs = float(constraint.coefficients @ distribution)
        if constraint.sense == "==" and abs(lhs - constraint.rhs) > DEFAULT_TOL:
            return False
        if constraint.sense == "<=" and lhs > constraint.rhs + DEFAULT_TOL:
            return False
        if constraint.sense == ">=" and lhs + DEFAULT_TOL < constraint.rhs:
            return False
    return abs(float(query.coefficients @ distribution) - expected) <= DEFAULT_TOL


if __name__ == "__main__":
    raise SystemExit(main())
