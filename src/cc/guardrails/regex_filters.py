# src/cc/guardrails/regex_filter.py
"""Regular expression based guardrail"""

import re

from .base import Guardrail


class RegexFilter(Guardrail):
    """Regular expression based content filter"""

    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = re.compile(pattern, flags)
        self.threshold = 0.5  # Binary for regex

    def blocks(self, text: str) -> bool:
        """Check if text matches blocking pattern"""
        return bool(self.pattern.search(text))

    def score(self, text: str) -> float:
        """Return binary score (0 or 1)"""
        return 1.0 if self.blocks(text) else 0.0

    def calibrate(self, benign_texts: list, target_fpr: float = 0.05) -> None:
        """Regex is binary - calibration not applicable"""
        # Calculate actual FPR for logging
        false_positives = sum(1 for text in benign_texts if self.blocks(text))
        actual_fpr = false_positives / len(benign_texts) if benign_texts else 0
        print(f"RegexFilter FPR: {actual_fpr:.3f} (target: {target_fpr:.3f})")
