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
            threshold = float(getattr(self._guardrail, "threshold", 0.5))
            return bool(self.score(text) > threshold)
        raise NotImplementedError("wrapped guardrail lacks blocks()")

    def calibrate(self, benign_texts: Iterable[str], target_fpr: float = 0.05) -> None:
        if hasattr(self._guardrail, "calibrate"):
            self._guardrail.calibrate(benign_texts, target_fpr)
        else:
            raise NotImplementedError("wrapped guardrail lacks calibrate()")
    def evaluate(self, text: str) -> tuple[bool, float]:
        """Return ``(blocked, score)`` pair for ``text``.

        This method ensures the wrapped guardrail's expensive ``score``
        calculation is executed at most once even if ``blocks`` internally
        calls ``score`` again. We compute the score first, then temporarily
        patch the underlying ``score`` method to return the cached value while
        calling ``blocks``.
        """
        score = self.score(text)
        # Call blocks with score memoised to avoid duplicate work
        blocked = False
        if hasattr(self._guardrail, "blocks"):
            if hasattr(self._guardrail, "score"):
                orig_score = self._guardrail.score
                try:
                    self._guardrail.score = lambda _t: score  # type: ignore[assignment]
                    blocked = bool(self._guardrail.blocks(text))
                finally:
                    self._guardrail.score = orig_score  # type: ignore[assignment]
            else:
                blocked = bool(self._guardrail.blocks(text))
        else:
            # Fallback: block when score > 0.0
            blocked = bool(score > 0.0)

        return blocked, score
    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple proxy
        return getattr(self._guardrail, name)


__all__ = ["GuardrailAdapter"]
