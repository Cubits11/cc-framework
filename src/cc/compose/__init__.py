"""ROC-free composition bounds over named binary events.

The entry point is :func:`compose_bounds`: marginals in, a sharp interval out.
See :mod:`cc.compose._bounds` for the design note on why this surface exists.
"""

from __future__ import annotations

from cc.compose._bounds import (
    CompositionBounds,
    CountermonotoneUndefinedError,
    DependenceAssumption,
    EventKind,
    MarginalProvenance,
    SensitivityRow,
    compose_bounds,
    marginal_from_operating_point,
    sensitivity,
)

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
