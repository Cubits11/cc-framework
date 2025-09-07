# src/cc/guardrails/composites.py
"""Logical composition helpers for guardrails.

These classes allow combining existing guardrails using boolean logic. They
delegate core behaviour (`blocks`, `score`, `calibrate`) to the child
guardrails while handling aggregation of the results.

Usage:

>>> from cc.guardrails.keyword_blocker import KeywordBlocker
>>> g1 = KeywordBlocker(["foo"])
>>> g2 = KeywordBlocker(["bar"])
>>> both = ANDGuardrail([g1, g2])
>>> either = ORGuardrail([g1, g2])
"""

from __future__ import annotations

from typing import Iterable, List

from .base import Guardrail


class CompositeGuardrail(Guardrail):
    """Base class for guardrails composed of other guardrails."""

    def __init__(self, guardrails: Iterable[Guardrail] | None = None):
        self.guardrails: List[Guardrail] = list(guardrails or [])

    # Guardrail API -----------------------------------------------------
    def calibrate(
        self, benign_texts: Iterable[str], target_fpr: float = 0.05
    ) -> None:
        for g in self.guardrails:
            g.calibrate(benign_texts, target_fpr)


class ANDGuardrail(CompositeGuardrail):
    """Blocks only if *all* child guardrails block."""

    def blocks(self, text: str) -> bool:  # type: ignore[override]
        if not self.guardrails:
            return False
        return all(g.blocks(text) for g in self.guardrails)

    def score(self, text: str) -> float:  # type: ignore[override]
        if not self.guardrails:
            return 0.0
        return min(g.score(text) for g in self.guardrails)


class ORGuardrail(CompositeGuardrail):
    """Blocks if *any* child guardrail blocks."""

    def blocks(self, text: str) -> bool:  # type: ignore[override]
        if not self.guardrails:
            return False
        return any(g.blocks(text) for g in self.guardrails)

    def score(self, text: str) -> float:  # type: ignore[override]
        if not self.guardrails:
            return 0.0
        return max(g.score(text) for g in self.guardrails)
