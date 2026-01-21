"""Toy guardrail with tunable keyword threshold for Week 5 pilots."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .base import Guardrail


class ToyThresholdGuardrail(Guardrail):
    """Counts harmful keywords and blocks when the normalized score exceeds a threshold."""

    def __init__(self, keywords: Iterable[str], threshold: float = 0.5, alpha_cap: float = 0.05):
        self.keywords = [k.strip().lower() for k in keywords if k and k.strip()]
        if not self.keywords:
            raise ValueError("ToyThresholdGuardrail requires at least one keyword")
        self.threshold = float(np.clip(threshold, 0.0, 1.0))
        self.alpha_cap = float(np.clip(alpha_cap, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Guardrail API
    # ------------------------------------------------------------------
    def blocks(self, text: str) -> bool:
        return self.score(text) > self.threshold

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = [tok.strip().lower() for tok in text.split() if tok.strip()]
        if not tokens:
            return 0.0
        harmful = sum(1 for tok in tokens if tok in self.keywords)
        # Normalize by token count to keep scores in [0,1]
        return float(np.clip(harmful / len(tokens), 0.0, 1.0))

    def calibrate(self, benign_texts: list, target_fpr: float = 0.05) -> None:
        if not benign_texts:
            return
        scores = np.array([self.score(text) for text in benign_texts], dtype=float)
        if scores.size == 0:
            return
        # Strict greater-than decision boundary; adjust threshold using quantiles.
        q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
        candidate = float(np.quantile(scores, q))
        # Enforce alpha cap so calibration never exceeds the allowed ceiling.
        if target_fpr > self.alpha_cap:
            target_fpr = self.alpha_cap
        self.threshold = float(np.clip(candidate, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Helpers for calibration script
    # ------------------------------------------------------------------
    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(np.clip(threshold, 0.0, 1.0))

    def get_threshold(self) -> float:
        return self.threshold

    def false_positive_rate(self, benign_texts: Iterable[str]) -> float:
        texts = list(benign_texts)
        if not texts:
            return 0.0
        blocked = sum(1 for text in texts if self.blocks(text))
        return float(blocked / len(texts))
