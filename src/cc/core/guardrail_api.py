# src/cc/core/guardrail_adapter.py
"""
Adapters and helper utilities for guardrails.

This module exposes :class:`GuardrailAdapter`, a robust wrapper that ensures any
guardrail implementation presents the minimal interface expected by the core
protocol: ``score`` / ``blocks`` / ``calibrate``.

Updated: 2025-09-28

Design goals
------------
- Stable, defensive API around heterogeneous guardrail implementations.
- Zero behavior surprises: never double-compute a score during evaluation.
- Graceful fallbacks when methods are missing, with clear error messages.
- Convenience utilities: batch scoring/evaluation, capability checks, and
  light threshold management for guardrails that don't store one.

Notes
-----
- ``evaluate`` computes ``score`` once, then calls ``blocks`` without
  recomputing. If the wrapped guardrail's ``blocks`` calls ``score`` internally,
  we temporarily monkey-patch it to return the cached value for the duration of
  that single call. A re-entrant lock provides basic thread-safety for that
  patching window.
- If the wrapped guardrail has no ``blocks``, we fall back to comparing
  ``score`` to a threshold. The adapter exposes a ``threshold`` attribute that
  proxies the underlying guardrail's threshold if present, otherwise stores a
  local one (default 0.5).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

# -----------------------------------------------------------------------------
# Lightweight structural typing for guardrails (duck typing > inheritance)
# -----------------------------------------------------------------------------


@runtime_checkable
class GuardrailLike(Protocol):
    # Minimal scoring interface
    def score(self, text: str) -> float: ...

    # Optional blocking decision
    def blocks(self, text: str) -> bool: ...  # pragma: no cover - optional

    # Optional calibration hook
    def calibrate(
        self, benign_texts: Iterable[str], target_fpr: float = 0.05
    ) -> Any: ...  # pragma: no cover - optional

    # Optional threshold attribute
    threshold: float  # pragma: no cover - optional


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------


@dataclass
class GuardrailAdapter:
    """Wrap an arbitrary guardrail to provide a stable API.

    Parameters
    ----------
    guardrail:
        Any object that (ideally) implements some subset of:
        ``score(str)->float``, ``blocks(str)->bool``, ``calibrate(iterable, float)``,
        and optionally exposes a ``threshold`` attribute.
    default_threshold:
        Used when the wrapped guardrail does not expose ``threshold``.
    """

    guardrail: GuardrailLike | Any
    default_threshold: float = 0.5

    # Internal state
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _local_threshold: Optional[float] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------
    def supports(self, method: str) -> bool:
        """Return True if the underlying guardrail has `method`."""
        return hasattr(self.guardrail, method)

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------
    @property
    def threshold(self) -> float:
        """Proxy threshold; falls back to adapter-local value or default."""
        if hasattr(self.guardrail, "threshold"):
            try:
                return float(getattr(self.guardrail, "threshold"))
            except Exception:
                pass
        if self._local_threshold is None:
            self._local_threshold = float(self.default_threshold)
        return self._local_threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        v = float(value)
        if math.isnan(v) or not math.isfinite(v):
            raise ValueError("threshold must be a finite float")
        if hasattr(self.guardrail, "threshold"):
            try:
                setattr(self.guardrail, "threshold", v)
                return
            except Exception:
                # fall back to local storage
                pass
        self._local_threshold = v

    # ------------------------------------------------------------------
    # Guardrail interface
    # ------------------------------------------------------------------
    def score(self, text: str) -> float:
        if not hasattr(self.guardrail, "score"):
            raise NotImplementedError("wrapped guardrail lacks score()")
        return float(self.guardrail.score(text))  # type: ignore[no-any-return]

    def blocks(self, text: str) -> bool:
        if hasattr(self.guardrail, "blocks"):
            return bool(self.guardrail.blocks(text))  # type: ignore[no-any-return]
        if hasattr(self.guardrail, "score"):
            return bool(self.score(text) > self.threshold)
        raise NotImplementedError("wrapped guardrail lacks blocks(); no score() fallback available")

    def calibrate(self, benign_texts: Iterable[str], target_fpr: float = 0.05) -> Any:
        if not 0.0 <= float(target_fpr) < 1.0:
            raise ValueError("target_fpr must be in [0,1)")
        if hasattr(self.guardrail, "calibrate"):
            return self.guardrail.calibrate(benign_texts, target_fpr)  # type: ignore[misc]
        raise NotImplementedError("wrapped guardrail lacks calibrate()")

    def evaluate(self, text: str) -> Tuple[bool, float]:
        """Return ``(blocked, score)`` pair for ``text`` without double work.

        The method computes the score exactly once. If the wrapped guardrail's
        ``blocks`` implementation internally calls ``score``, we temporarily
        shim ``score`` to return the cached value for that single call.
        """
        s = self.score(text)
        if hasattr(self.guardrail, "blocks"):
            if hasattr(self.guardrail, "score"):
                with self._memoised_score(s):
                    blocked = bool(self.guardrail.blocks(text))  # type: ignore[no-any-return]
            else:
                blocked = bool(self.guardrail.blocks(text))  # type: ignore[no-any-return]
        else:
            blocked = bool(s > self.threshold)
        return blocked, float(s)

    # ------------------------------------------------------------------
    # Batch helpers (quality-of-life; deterministic order preserved)
    # ------------------------------------------------------------------
    def batch_score(self, texts: Iterable[str]) -> List[float]:
        """Score a batch of texts."""
        return [self.score(t) for t in texts]

    def batch_evaluate(self, texts: Iterable[str]) -> List[Tuple[bool, float]]:
        """Evaluate a batch of texts, returning ``[(blocked, score), ...]``."""
        # Compute all scores first (single pass), then reuse during blocks()
        texts_list = list(texts)
        scores = [self.score(t) for t in texts_list]
        results: List[Tuple[bool, float]] = []
        if hasattr(self.guardrail, "blocks") and hasattr(self.guardrail, "score"):
            # One patch per call to keep the shim scope tight and thread-safe
            for t, s in zip(texts_list, scores):
                with self._memoised_score(s):
                    blocked = bool(self.guardrail.blocks(t))  # type: ignore[no-any-return]
                results.append((blocked, float(s)))
            return results

        # Fallback path uses blocks() directly or threshold rule
        for t, s in zip(texts_list, scores):
            if hasattr(self.guardrail, "blocks"):
                blocked = bool(self.guardrail.blocks(t))  # type: ignore[no-any-return]
            else:
                blocked = bool(s > self.threshold)
            results.append((blocked, float(s)))
        return results

    # ------------------------------------------------------------------
    # Attribute delegation & repr
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple proxy
        # Avoid recursion on dataclass internals
        if name in {"guardrail", "default_threshold", "_lock", "_local_threshold"}:
            raise AttributeError(name)
        return getattr(self.guardrail, name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        cls = type(self.guardrail).__name__
        return f"GuardrailAdapter({cls}, threshold={self.threshold:.3f})"

    # ------------------------------------------------------------------
    # Internal: score memoisation shim
    # ------------------------------------------------------------------
    @contextmanager
    def _memoised_score(self, value: float):
        """Context manager to temporarily replace guardrail.score with a lambda.

        Thread-safety: protected by an RLock so nested usage is safe.
        """
        if not hasattr(self.guardrail, "score"):
            # Nothing to memoise
            yield
            return
        with self._lock:
            orig = self.guardrail.score  # type: ignore[attr-defined]
            try:
                # mypy: guardrail is dynamic; runtime attribute assignment is intended.
                self.guardrail.score = lambda _t: value  # type: ignore[assignment]
                yield
            finally:
                self.guardrail.score = orig  # type: ignore[assignment]


__all__ = ["GuardrailAdapter", "GuardrailLike"]
