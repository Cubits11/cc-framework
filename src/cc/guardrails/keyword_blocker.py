# src/cc/guardrails/keyword_blocker.py
"""Keyword-based blocking guardrail"""

from typing import List

from .base import Guardrail


class KeywordBlocker(Guardrail):
    """Keyword-based content blocker with fuzzy matching"""

    def __init__(self, keywords: List[str], fuzzy_threshold: float = 0.8):
        self.keywords = set(k.lower() for k in keywords)
        self.fuzzy_threshold = fuzzy_threshold
        self.blocking_threshold = 0.5

    def blocks(self, text: str) -> bool:
        """Check if text contains blocking keywords"""
        return self.score(text) > self.blocking_threshold

    def score(self, text: str) -> float:
        """Compute blocking score based on keyword matches"""
        text_lower = text.lower()

        # Exact matches
        exact_matches = sum(1 for keyword in self.keywords if keyword in text_lower)

        # Fuzzy matches (simple Levenshtein-style)
        fuzzy_matches = 0
        words = text_lower.split()

        for word in words:
            for keyword in self.keywords:
                similarity = self._fuzzy_similarity(word, keyword)
                if similarity > self.fuzzy_threshold:
                    fuzzy_matches += 1
                    break

        total_matches = exact_matches + fuzzy_matches * 0.5  # Weight fuzzy lower
        max_score = len(self.keywords)

        return min(1.0, total_matches / max(1, max_score))

    def _fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Simple character-based similarity"""
        if not s1 or not s2:
            return 0.0

        # Simple Jaccard similarity on character bigrams
        bigrams1 = {s1[i : i + 2] for i in range(len(s1) - 1)}
        bigrams2 = {s2[i : i + 2] for i in range(len(s2) - 1)}

        if not bigrams1 and not bigrams2:
            return 1.0

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)

        return intersection / union if union > 0 else 0.0

    def calibrate(self, benign_texts: list, target_fpr: float = 0.05) -> None:
        """Calibrate blocking threshold to achieve target FPR"""
        if not benign_texts:
            return

        scores = [self.score(text) for text in benign_texts]
        scores.sort(reverse=True)

        # Find threshold that gives approximately target FPR
        n = len(scores)
        target_idx = int(target_fpr * n)

        if target_idx < n:
            self.blocking_threshold = scores[target_idx]
        else:
            self.blocking_threshold = 1.0  # No blocking if all scores too low

        # Validate
        actual_fpr = sum(1 for s in scores if s > self.blocking_threshold) / n
        print(
            f"KeywordBlocker calibrated: threshold={self.blocking_threshold:.3f}, FPR={actual_fpr:.3f}"
        )
