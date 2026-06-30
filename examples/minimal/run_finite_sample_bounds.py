#!/usr/bin/env python
"""Minimal finite-sample count-to-bounds example."""

from __future__ import annotations

import json
from typing import Any

from cc.kernel.strict import (
    LinearQuery,
    PairwiseCountEvidence,
    SingletonCountEvidence,
    composition_bounds_from_counts,
)


def main() -> int:
    labels = ("input_filter", "semantic_judge")
    query = LinearQuery.intersection(labels, labels, name="AND_failure")
    singletons = [
        SingletonCountEvidence("input_filter", failures=20, n=100),
        SingletonCountEvidence("semantic_judge", failures=35, n=100),
    ]

    singleton_only = composition_bounds_from_counts(query, labels, singletons, delta=0.05)
    singleton_and_pairwise = composition_bounds_from_counts(
        query,
        labels,
        singletons,
        pairwise_counts=[
            PairwiseCountEvidence("input_filter", "semantic_judge", co_failures=12, n=100),
        ],
        delta=0.05,
    )

    print(
        json.dumps(
            {
                "failure_event_convention": "Z_i = 1 means guardrail failure / unsafe pass.",
                "singleton_only": _summary(singleton_only),
                "singleton_plus_pairwise": _summary(singleton_and_pairwise),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def _summary(result: Any) -> dict[str, Any]:
    return {
        "intervals": [
            {
                "name": interval.name,
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "radius": interval.radius,
            }
            for interval in result.intervals
        ],
        "composition_lower_bound": result.identification.lower_bound,
        "composition_upper_bound": result.identification.upper_bound,
        "assumptions_hash": result.identification.assumptions_hash,
    }


if __name__ == "__main__":
    raise SystemExit(main())
