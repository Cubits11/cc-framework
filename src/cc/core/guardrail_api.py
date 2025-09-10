"""Adapters and helper utilities for guardrails.

This module currently exposes :class:`GuardrailAdapter`, a thin wrapper
that ensures any guardrail implementation provides the minimal interface
expected by the core protocol (``score``/``blocks``/``calibrate``).
"""

from __future__ import annotations

from typing import Any, Iterable

from cc.guardrails.base import Guardrail


class GuardrailAdapter(Guardrail):
    """Wrap an arbitrary guardrail to provide a stable API.

    The adapter delegates ``score``, ``blocks`` and ``calibrate`` to the
    wrapped guardrail if they are available. Missing methods raise
    ``NotImplementedError`` where appropriate. Additional attributes are
    proxied via ``__getattr__`` to maintain access to implementation-specific
    details (e.g. thresholds).
    """

    def __init__(self, guardrail: Any) -> None:  # pragma: no cover - trivial
        self._guardrail = guardrail

    # ------------------------------------------------------------------
    # Guardrail interface
    # ------------------------------------------------------------------
    def score(self, text: str) -> float:
        if not hasattr(self._guardrail, "score"):
            raise NotImplementedError("wrapped guardrail lacks score()")
        return float(self._guardrail.score(text))

    def blocks(self, text: str) -> bool:
        if hasattr(self._guardrail, "blocks"):
            return bool(self._guardrail.blocks(text))
        if hasattr(self._guardrail, "score"):
            # Fallback: block when score > 0.0
            return bool(self._guardrail.score(text) > 0.0)
        raise NotImplementedError("wrapped guardrail lacks blocks()")

    def calibrate(self, benign_texts: Iterable[str], target_fpr: float = 0.05) -> None:
        if hasattr(self._guardrail, "calibrate"):
            self._guardrail.calibrate(benign_texts, target_fpr)
        else:
            raise NotImplementedError("wrapped guardrail lacks calibrate()")
    def evaluate(self, text: str) -> tuple[bool, float]:
        """Return ``(blocked, score)`` pair for ``text``."""
        score = self.score(text)
        return self.blocks(text), score
    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple proxy
        return getattr(self._guardrail, name)


__all__ = ["GuardrailAdapter"]
