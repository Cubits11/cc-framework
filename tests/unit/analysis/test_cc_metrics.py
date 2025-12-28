# tests/unit/test_cc_metrics.py
"""
Unit tests for CC metrics computation
"""

import numpy as np

from cc.core.stats import bootstrap_ci_with_cc, compute_j_statistic
from tests._factories import mk_attack_result


class TestCCMetrics:
    def test_perfect_defense(self):
        """Test J=0 for perfect defense"""
        results = [
            mk_attack_result(
                world_bit=0,
                success=True,
                attack_id=f"a{i}",
                guardrails_applied="",
                rng_seed=i,
            )
            for i in range(100)
        ] + [
            mk_attack_result(
                world_bit=1,
                success=False,
                attack_id=f"a{i}",
                guardrails_applied="g1",
                rng_seed=i,
            )
            for i in range(100, 200)
        ]
        j_stat, p0, p1 = compute_j_statistic(results)
        assert np.isclose(j_stat, 1.0)
        assert np.isclose(p0, 1.0)
        assert np.isclose(p1, 0.0)

    def test_no_effect(self):
        """Test J≈0 for no defense effect"""
        rng = np.random.default_rng(42)
        results = []

        for i in range(1000):
            world = rng.integers(0, 2)
            # Same success rate regardless of world
            success = rng.random() < 0.6
            results.append(
                mk_attack_result(
                    world_bit=world,
                    success=success,
                    attack_id=f"a{i}",
                    guardrails_applied="g1" if world else "",
                    rng_seed=i,
                )
            )

        j_stat, p0, p1 = compute_j_statistic(results)
        assert abs(j_stat) < 0.1  # Should be near zero

    def test_bootstrap_stability(self):
        """Test bootstrap CI stability"""
        # Generate synthetic results
        results = []
        for i in range(500):
            world = i % 2
            success = (i % 3) == 0 if world == 0 else (i % 5) == 0
            results.append(
                mk_attack_result(
                    world_bit=world,
                    success=success,
                    attack_id=f"a{i}",
                    guardrails_applied="",
                    rng_seed=i,
                )
            )

        # Run bootstrap twice with same seed
        j_individual = {"g1": 0.2}

        result1 = bootstrap_ci_with_cc(results, j_individual, B=500)
        result2 = bootstrap_ci_with_cc(results, j_individual, B=500)

        # Should be identical with fixed seed
        assert abs(result1.j_statistic - result2.j_statistic) < 1e-10
        assert abs(result1.cc_max - result2.cc_max) < 1e-10
        assert result1.ci_j == result2.ci_j
