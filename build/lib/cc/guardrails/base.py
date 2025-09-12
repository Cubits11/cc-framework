# src/cc/guardrails/base.py
"""Base guardrail interface"""

from abc import ABC, abstractmethod


class Guardrail(ABC):
    """Abstract base class for guardrails"""

    @abstractmethod
    def blocks(self, text: str) -> bool:
        """Returns True if text should be blocked"""
        pass

    @abstractmethod
    def score(self, text: str) -> float:
        """Returns blocking score (0=safe, 1=block)"""
        pass

    @abstractmethod
    def calibrate(self, benign_texts: list, target_fpr: float = 0.05) -> None:
        """Calibrate guardrail to target false positive rate"""
        pass
