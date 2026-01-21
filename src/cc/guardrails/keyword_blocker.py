# src/cc/guardrails/keyword_blocker.py
"""
Keyword-based blocking guardrail.

Policy:
- Deterministic, CPU-cheap scoring for smoke/MVP runs.
- Exact-match + lightweight fuzzy match (character bigram Jaccard).
- Calibrate threshold on benign texts via quantile to hit target FPR.
- Strict decision rule: blocks(text) ⇔ score(text) > threshold (ties pass).

Notes:
- Fuzzy matches are down-weighted vs exact to reduce overblocking.
- All scores are clamped to [0,1] and invariant to keyword set size via normalization.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .base import Guardrail


def _bigram_set(s: str) -> set[str]:
    """Return the set of character bigrams for a lowercased string."""
    if not s:
        return set()
    s = s.lower()
    if len(s) < 2:
        return {s}  # treat single-char token as degenerate bigram
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity for sets (safe on empties)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


@dataclass
class KeywordBlockerConfig:
    fuzzy_threshold: float = 0.80  # token ≈ keyword if Jaccard ≥ this
    fuzzy_weight: float = 0.50  # contribution of fuzzy vs exact (∈[0,1])
    initial_threshold: float = 0.50  # pre-calibration threshold
    verbose: bool = False


class KeywordBlocker(Guardrail):
    """Keyword-based content blocker with fuzzy matching and quantile calibration."""

    def __init__(self, keywords: Sequence[str], cfg: KeywordBlockerConfig | None = None):
        self.cfg = cfg or KeywordBlockerConfig()

        # Normalize/keep only non-empty
        self.keywords: set[str] = {k.strip().lower() for k in keywords if k and k.strip()}
        # Precompute bigrams for fuzzy similarity
        self._kw_bigrams: dict[str, set[str]] = {k: _bigram_set(k) for k in self.keywords}

        # Decision threshold (strictly greater than)
        self.blocking_threshold: float = float(self.cfg.initial_threshold)

        # Normalization constants
        self._fuzzy_weight = float(np.clip(self.cfg.fuzzy_weight, 0.0, 1.0))
        self._max_exact = max(1, len(self.keywords))

    # ----------------- Public API -----------------

    def blocks(self, text: str) -> bool:
        """Return True if the text should be blocked (strict threshold)."""
        return self.score(text) > self.blocking_threshold

    def score(self, text: str) -> float:
        """
        Score in [0,1] combining:
          - exact substring keyword hits (weight 1.0 each)
          - fuzzy token ≈ keyword matches via char-bigram Jaccard (down-weighted)
        """
        if not text:
            return 0.0
        tl = text.lower()

        # Exact matches (substring presence)
        exact = sum(1 for kw in self.keywords if kw and kw in tl)

        # Fuzzy matches (per-token max similarity vs any keyword)
        tokens = [tok for tok in tl.split() if tok]
        token_bigrams = [_bigram_set(tok) for tok in tokens]

        fuzzy_count = 0.0
        for tb in token_bigrams:
            best = 0.0
            # Find best similarity to any keyword
            for kb in self._kw_bigrams.values():
                s = _jaccard(tb, kb)
                if s > best:
                    best = s
                    if best >= 1.0:
                        break
            if best >= self.cfg.fuzzy_threshold:
                fuzzy_count += 1.0

        raw = float(exact) + self._fuzzy_weight * float(fuzzy_count)

        # Normalize by capacity: exact capacity + weighted “potential” fuzzy capacity
        denom = float(self._max_exact + self._fuzzy_weight * max(0, len(tokens)))
        denom = max(1.0, denom)
        score = raw / denom
        return float(np.clip(score, 0.0, 1.0))

    # ----------------- Calibration -----------------

    def calibrate(self, benign_texts: Iterable[str], target_fpr: float = 0.05) -> None:
        """
        Choose blocking_threshold so benign FPR ≈ target_fpr using quantiles:
            threshold := quantile(scores_benign, q = 1 - target_fpr)
        Decision rule uses strict '>' (ties pass), matching ROC sweeps.

        Edge cases:
          - No texts or all non-finite scores → threshold = 1.0 (block nothing).
        """
        texts = list(benign_texts)
        if not texts:
            self.blocking_threshold = 1.0
            if self.cfg.verbose:
                print("KeywordBlocker calibrate: no benign texts; threshold → 1.0")
            return

        scores = [float(self.score(t)) for t in texts]
        scores = [s for s in scores if np.isfinite(s)]
        if not scores:
            self.blocking_threshold = 1.0
            if self.cfg.verbose:
                print("KeywordBlocker calibrate: benign scores non-finite; threshold → 1.0")
            return

        q = float(np.clip(1.0 - float(target_fpr), 0.0, 1.0))
        thr = float(np.quantile(scores, q))
        self.blocking_threshold = thr

        if self.cfg.verbose:
            n = len(scores)
            approx_fpr = sum(1 for s in scores if s > self.blocking_threshold) / n
            print(
                f"KeywordBlocker calibrated: threshold={self.blocking_threshold:.3f}, "
                f"target_fpr={target_fpr:.3f}, FPR≈{approx_fpr:.3f}, n={n}"
            )

    # ----------------- Utilities -----------------

    def keywords_list(self) -> list[str]:
        """Sorted keywords for UIs/tests."""
        return sorted(self.keywords)

    def set_threshold(self, threshold: float) -> None:
        """Manual override for decision threshold (clipped to [0,1])."""
        self.blocking_threshold = float(np.clip(threshold, 0.0, 1.0))
