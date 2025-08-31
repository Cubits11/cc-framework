# src/cc/core/composition_theory.py (New module)
"""
Theoretical foundations for guardrail composition
Bridges IST 496 to PhD Year 1
"""

from typing import Tuple


class CompositionAlgebra:
    """
    Algebraic structure for guardrail composition
    Foundation for PhD Year 1 theoretical work
    """

    @staticmethod
    def binary_composition_bound(
        j1: float, j2: float, correlation: float = 0
    ) -> Tuple[float, float]:
        """
        Analytical bounds for binary composition

        Returns:
            (lower_bound, upper_bound) for composed J-statistic
        """
        # Independent case
        j_independent = j1 + j2 - j1 * j2

        # Perfect correlation cases
        j_redundant = max(j1, j2)
        j_complementary = min(j1 + j2, 1.0)

        # Correlation-adjusted bounds
        lower = (1 - abs(correlation)) * j_independent + abs(correlation) * j_redundant
        upper = (1 - abs(correlation)) * j_independent + abs(correlation) * j_complementary

        return (lower, upper)

    @staticmethod
    def n_way_complexity_bound(n: int) -> float:
        """
        Computational complexity bound for n-way composition
        Preview of PhD Year 3 work
        """
        # O(n^2) pairwise interactions
        # O(2^n) for full interaction graph
        return n * (n - 1) / 2  # Simplified for IST 496
