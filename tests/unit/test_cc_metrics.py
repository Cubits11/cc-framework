# tests/unit/test_cc_metrics.py
"""
Unit tests for CC metrics computation
"""
import pytest
import numpy as np
from cc.core.models import AttackResult
from cc.core.stats import compute_j_statistic, bootstrap_ci_with_cc

class TestCCMetrics:
    
    def test_perfect_defense(self):
        """Test J=0 for perfect defense"""
        results = [
            AttackResult(world_bit=0, success=True, attack_id=f"a{i}", 
                        transcript_hash="", guardrails_applied="", 
                        rng_seed=i, timestamp=0)
            for i in range(100)
        ] + [
            AttackResult(world_bit=1, success=False, attack_id=f"a{i}", 
                        transcript_hash="", guardrails_applied="g1", 
                        rng_seed=i, timestamp=0)
            for i in range(100, 200)
        ]
        
        j_stat, p0, p1 = compute_j_statistic(results)
        assert j_stat == 1.0
        assert p0 == 1.0
        assert p1 == 0.0
    
    def test_no_effect(self):
        """Test J≈0 for no defense effect"""
        rng = np.random.default_rng(42)
        results = []
        
        for i in range(1000):
            world = rng.integers(0, 2)
            # Same success rate regardless of world
            success = rng.random() < 0.6
            results.append(
                AttackResult(world_bit=world, success=success,
                           attack_id=f"a{i}", transcript_hash="",
                           guardrails_applied="g1" if world else "",
                           rng_seed=i, timestamp=0)
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
                AttackResult(world_bit=world, success=success,
                           attack_id=f"a{i}", transcript_hash="",
                           guardrails_applied="", rng_seed=i, timestamp=0)
            )
        
        # Run bootstrap twice with same seed
        j_individual = {"g1": 0.2}
        
        result1 = bootstrap_ci_with_cc(results, j_individual, B=500)
        result2 = bootstrap_ci_with_cc(results, j_individual, B=500)
        
        # Should be identical with fixed seed
        assert abs(result1.j_statistic - result2.j_statistic) < 1e-10
        assert abs(result1.cc_max - result2.cc_max) < 1e-10
        assert result1.ci_j == result2.ci_j
