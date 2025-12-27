# experiments/correlation_cliff/baselines.py
"""
Baseline comparisons for correlation cliff.

Baselines:
1. Random composition (shuffle assignments)
2. Independent composition (assume p11 = pA * pB)
3. Oracle composition (know true dependence from data)
"""

import numpy as np
from theory_core import TwoWorldMarginals
from simulate import SimConfig, simulate_grid

def baseline_random_composition(w: TwoWorldMarginals, n: int, seed: int):
    """
    Random baseline: Shuffle A/B assignments (breaks correlation).
    """
    rng = np.random.default_rng(seed)
    # World 0
    A0 = rng.binomial(1, w.pA0, size=n)
    B0 = rng.permutation(rng.binomial(1, w.pB0, size=n))  # Shuffle!
    # World 1
    A1 = rng.binomial(1, w.pA1, size=n)
    B1 = rng.permutation(rng.binomial(1, w.pB1, size=n))
    
    # Compute CC for OR rule
    C0 = np.logical_or(A0, B0).astype(int)
    C1 = np.logical_or(A1, B1).astype(int)
    pC0 = C0.mean()
    pC1 = C1.mean()
    JC = abs(pC1 - pC0)
    Jbest = max(abs(w.pA1 - w.pA0), abs(w.pB1 - w.pB0))
    CC = JC / Jbest if Jbest > 0 else float('nan')
    
    return {"CC_random": CC, "JC_random": JC}


def baseline_independent_composition(w: TwoWorldMarginals):
    """
    Independent baseline: Assume p11 = pA * pB (no correlation).
    """
    # World 0
    p11_0 = w.pA0 * w.pB0
    pC0 = w.pA0 + w.pB0 - p11_0  # OR rule
    
    # World 1
    p11_1 = w.pA1 * w.pB1
    pC1 = w.pA1 + w.pB1 - p11_1
    
    JC = abs(pC1 - pC0)
    Jbest = max(abs(w.pA1 - w.pA0), abs(w.pB1 - w.pB0))
    CC = JC / Jbest if Jbest > 0 else float('nan')
    
    return {"CC_independent": CC, "JC_independent": JC}