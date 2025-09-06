"""Abstractions for interacting with guardrails.

This module provides a minimal API that the rest of the framework can use
without depending on concrete guardrail implementations. Guardrails expose an
`evaluate` method returning both the blocking decision and a raw score, as well
as a `calibrate` method for threshold fitting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from cc.guardrails.base import Guardrail


class GuardrailAPI(ABC):
    """Abstract interface for guardrail interaction."""

    @abstractmethod
    def evaluate(self, text: str) -> Tuple[bool, float]:
        """Return ``(blocked, score)`` for the supplied text."""

    @abstractmethod
    def calibrate(self, benign_texts: Sequence[str], target_fpr: float = 0.05) -> None:
        """Calibrate guardrail using benign examples."""


class GuardrailAdapter(GuardrailAPI):
    """Adapter wrapping any :class:`~cc.guardrails.base.Guardrail` instance."""

    def __init__(self, guardrail: Guardrail):
        self.guardrail = guardrail

    def evaluate(self, text: str) -> Tuple[bool, float]:  # pragma: no cover - simple delegation
        """Delegate evaluation to underlying guardrail."""
        return self.guardrail.blocks(text), self.guardrail.score(text)

    def calibrate(self, benign_texts: Sequence[str], target_fpr: float = 0.05) -> None:
        """Delegate calibration to underlying guardrail."""
        self.guardrail.calibrate(list(benign_texts), target_fpr=target_fpr)


__all__ = ["GuardrailAPI", "GuardrailAdapter"]
