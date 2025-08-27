# src/cc/core/stats.py
"""
Statistical methods for CC framework with proper two-world handling
Implements bootstrap CIs, analytical bounds, and composability metrics

Author: Pranav Bhave
Date: 2025-08-27
Institution: Penn State University
"""
from __future__ import annotations
import numpy as np
import math
import warnings
from typing import Tuple, List, Optional, Dict, Any, Union
from scipy import stats as scipy_stats
from dataclasses import dataclass, field
import hashlib
import json
from collections import defaultdict

# Type aliases for clarity
AttackResults = List['AttackResult']
ConfidenceInterval = Tuple[float, float]
MetricsDict = Dict[str, float]

@dataclass
class BootstrapResult:
    """Comprehensive results of bootstrap analysis with CC metrics"""
    # Primary statistics
    j_statistic: float
    p0: float  # Success rate in world 0
    p1: float  # Success rate in world 1
    
    # Composability metrics
    cc_max: float
    delta_add: float
    cc_multiplicative: Optional[float] = None
    
    # Confidence intervals
    ci_j: ConfidenceInterval = (0.0, 0.0)
    ci_cc_max: ConfidenceInterval = (0.0, 0.0)
    ci_delta_add: ConfidenceInterval = (0.0, 0.0)
    ci_width: float = 0.0
    
    # Bootstrap diagnostics
    bootstrap_samples: Optional[np.ndarray] = None
    n_sessions: int = 0
    n_bootstrap: int = 0
    convergence_diagnostic: float = 0.0
    effective_sample_size: float = 0.0
    
    # Metadata
    method: str = "percentile"
    block_size: Optional[int] = None
    
    # Statistical tests
    hypothesis_tests: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'j_statistic': float(self.j_statistic),
            'p0': float(self.p0),
            'p1': float(self.p1),
            'cc_max': float(self.cc_max),
            'delta_add': float(self.delta_add),
            'cc_multiplicative': float(self.cc_multiplicative) if self.cc_multiplicative else None,
            'confidence_intervals': {
                'j': {'lower': float(self.ci_j[0]), 'upper': float(self.ci_j[1])},
                'cc_max': {'lower': float(self.ci_cc_max[0]), 'upper': float(self.ci_cc_max[1])},
                'delta_add': {'lower': float(self.ci_delta_add[0]), 'upper': float(self.ci_delta_add[1])}
            },
            'diagnostics': {
                'n_sessions': self.n_sessions,
                'n_bootstrap': self.n_bootstrap,
                'convergence': float(self.convergence_diagnostic),
                'effective_sample_size': float(self.effective_sample_size)
            },
            'hypothesis_tests': self.hypothesis_tests,
            'method': self.method
        }

@dataclass
class CCInterpretation:
    """Interpretation of CC results"""
    regime: str  # "constructive", "independent", "destructive"
    confidence: float
    recommendation: str
    evidence: Dict[str, Any]

def compute_j_statistic(results: AttackResults) -> Tuple[float, float, float]:
    """
    Compute J-statistic (Youden's index) from attack results
    
    Args:
        results: List of AttackResult objects
        
    Returns:
        (j_statistic, p0, p1) where:
        - j_statistic: Difference in success rates (p0 - p1)
        - p0: Success rate in world 0 (no protection)
        - p1: Success rate in world 1 (with protection)
    """
    if not results:
        raise ValueError("Empty results list")
    
    world_0 = [r for r in results if r.world_bit == 0]
    world_1 = [r for r in results if r.world_bit == 1]
    
    if not world_0 or not world_1:
        raise ValueError(f"Need results from both worlds. Got {len(world_0)} from world 0, {len(world_1)} from world 1")
    
    p0 = sum(r.success for r in world_0) / len(world_0)
    p1 = sum(r.success for r in world_1) / len(world_1)
    j_stat = p0 - p1
    
    return j_stat, p0, p1

def compute_composability_coefficients(
    j_composed: float,
    j_individual: Dict[str, float],
    p0: float
) -> Dict[str, float]:
    """
    Compute all composability coefficient variants
    
    Args:
        j_composed: J-statistic for composed guardrails
        j_individual: Dict of J-statistics for individual guardrails
        p0: Baseline success rate (world 0)
    
    Returns:
        Dictionary containing CC metrics
    """
    if not j_individual:
        return {
            'cc_max': 1.0,
            'delta_add': 0.0,
            'cc_multiplicative': 1.0,
            'j_theoretical_max': j_composed,
            'j_theoretical_add': j_composed
        }
    
    j_max_individual = max(j_individual.values())
    j_min_individual = min(j_individual.values())
    
    # Primary metrics
    cc_max = j_composed / j_max_individual if j_max_individual > 0 else 0.0
    delta_add = j_composed - j_max_individual
    
    # Secondary metrics
    j_values = list(j_individual.values())
    if len(j_values) > 1:
        # Theoretical bounds
        j_theoretical_max = min(sum(j_values), p0)
        j_theoretical_add = min(sum(j_values) - np.prod(j_values), p0)
        j_multiplicative = np.prod(j_values)
    else:
        j_theoretical_max = j_values[0]
        j_theoretical_add = j_values[0]
        j_multiplicative = j_values[0]
    
    cc_multiplicative = j_composed / j_multiplicative if j_multiplicative > 0 else 0.0
    
    return {
        'cc_max': cc_max,
        'delta_add': delta_add,
        'cc_multiplicative': cc_multiplicative,
        'j_composed': j_composed,
        'j_max_individual': j_max_individual,
        'j_theoretical_max': j_theoretical_max,
        'j_theoretical_add': j_theoretical_add,
        'efficiency_ratio': j_composed / j_theoretical_max if j_theoretical_max > 0 else 0.0
    }

def bootstrap_ci_with_cc(
    results: AttackResults,
    j_individual: Optional[Dict[str, float]] = None,
    B: int = 2000,
    alpha: float = 0.05,
    block_size: Optional[int] = None,
    method: str = "percentile",
    compute_cc: bool = True,
    random_seed: int = 42
) -> BootstrapResult:
    """
    Bootstrap confidence intervals for J-statistic and CC metrics with proper two-world handling
    
    Args:
        results: List of AttackResult objects from two-world protocol
        j_individual: Individual J-statistics for CC computation (optional)
        B: Number of bootstrap samples
        alpha: Significance level for CIs
        block_size: Size of blocks for block bootstrap (None = auto-detect)
        method: CI method ("percentile", "bca", "basic")
        compute_cc: Whether to compute CC metrics
        random_seed: Random seed for reproducibility
    
    Returns:
        BootstrapResult with all metrics and diagnostics
    """
    n = len(results)
    if n < 50:
        warnings.warn(f"Small sample size ({n}). Bootstrap may be unreliable.")
    
    # Auto-detect block size if needed
    if block_size is None:
        block_size = _estimate_block_size(results)
    
    # Empirical estimates
    j_emp, p0_emp, p1_emp = compute_j_statistic(results)
    
    # CC metrics if requested
    if compute_cc and j_individual:
        cc_metrics = compute_composability_coefficients(j_emp, j_individual, p0_emp)
        cc_max_emp = cc_metrics['cc_max']
        delta_add_emp = cc_metrics['delta_add']
        cc_mult_emp = cc_metrics['cc_multiplicative']
        j_max_individual = cc_metrics['j_max_individual']
    else:
        cc_max_emp = delta_add_emp = cc_mult_emp = 0.0
        j_max_individual = j_emp
    
    # Bootstrap resampling
    bootstrap_j = []
    bootstrap_cc = []
    bootstrap_delta = []
    bootstrap_p0 = []
    bootstrap_p1 = []
    
    rng = np.random.default_rng(random_seed)
    
    # Separate results by world for stratified bootstrap
    world_0_results = [r for r in results if r.world_bit == 0]
    world_1_results = [r for r in results if r.world_bit == 1]
    
    for b in range(B):
        # Stratified bootstrap - maintain world balance
        n0 = len(world_0_results)
        n1 = len(world_1_results)
        
        # Resample within each world
        boot_0_idx = rng.choice(n0, size=n0, replace=True)
        boot_1_idx = rng.choice(n1, size=n1, replace=True)
        
        boot_sample = [world_0_results[i] for i in boot_0_idx] + \
                     [world_1_results[i] for i in boot_1_idx]
        
        # Compute statistics
        try:
            j_boot, p0_boot, p1_boot = compute_j_statistic(boot_sample)
            bootstrap_j.append(j_boot)
            bootstrap_p0.append(p0_boot)
            bootstrap_p1.append(p1_boot)
            
            if compute_cc and j_individual:
                cc_boot = j_boot / j_max_individual if j_max_individual > 0 else 0.0
                delta_boot = j_boot - j_max_individual
                bootstrap_cc.append(cc_boot)
                bootstrap_delta.append(delta_boot)
        except:
            # Handle edge cases in bootstrap samples
            continue
    
    bootstrap_j = np.array(bootstrap_j)
    bootstrap_p0 = np.array(bootstrap_p0)
    bootstrap_p1 = np.array(bootstrap_p1)
    
    if compute_cc and bootstrap_cc:
        bootstrap_cc = np.array(bootstrap_cc)
        bootstrap_delta = np.array(bootstrap_delta)
    
    # Compute confidence intervals
    if method == "percentile":
        ci_j = _percentile_ci(bootstrap_j, alpha)
        ci_cc = _percentile_ci(bootstrap_cc, alpha) if compute_cc and len(bootstrap_cc) > 0 else (0, 0)
        ci_delta = _percentile_ci(bootstrap_delta, alpha) if compute_cc and len(bootstrap_delta) > 0 else (0, 0)
    elif method == "bca":
        ci_j = _bca_ci(bootstrap_j, j_emp, results, alpha)
        ci_cc = _bca_ci(bootstrap_cc, cc_max_emp, results, alpha) if compute_cc else (0, 0)
        ci_delta = _bca_ci(bootstrap_delta, delta_add_emp, results, alpha) if compute_cc else (0, 0)
    else:  # basic
        ci_j = _basic_ci(bootstrap_j, j_emp, alpha)
        ci_cc = _basic_ci(bootstrap_cc, cc_max_emp, alpha) if compute_cc else (0, 0)
        ci_delta = _basic_ci(bootstrap_delta, delta_add_emp, alpha) if compute_cc else (0, 0)
    
    # Convergence diagnostics
    convergence = _compute_convergence_diagnostic(bootstrap_j)
    ess = _effective_sample_size(bootstrap_j)
    
    # Hypothesis tests
    hypothesis_tests = _run_hypothesis_tests(
        j_emp, bootstrap_j, cc_max_emp if compute_cc else None,
        bootstrap_cc if compute_cc else None, alpha
    )
    
    return BootstrapResult(
        j_statistic=j_emp,
        p0=p0_emp,
        p1=p1_emp,
        cc_max=cc_max_emp if compute_cc else 0.0,
        delta_add=delta_add_emp if compute_cc else 0.0,
        cc_multiplicative=cc_mult_emp if compute_cc else None,
        ci_j=ci_j,
        ci_cc_max=ci_cc if compute_cc else (0, 0),
        ci_delta_add=ci_delta if compute_cc else (0, 0),
        ci_width=ci_j[1] - ci_j[0],
        bootstrap_samples=bootstrap_j,
        n_sessions=n,
        n_bootstrap=B,
        convergence_diagnostic=convergence,
        effective_sample_size=ess,
        method=method,
        block_size=block_size,
        hypothesis_tests=hypothesis_tests
    )

def dkw_confidence_bound(n: int, alpha: float = 0.05) -> float:
    """
    Dvoretzky-Kiefer-Wolfowitz uniform confidence band halfwidth
    Provides analytical fallback for bootstrap
    
    Args:
        n: Sample size
        alpha: Significance level
    
    Returns:
        DKW bound (halfwidth of confidence band)
    """
    return math.sqrt(math.log(2/alpha) / (2*max(n, 1)))

def analytical_j_ci(
    results: AttackResults,
    alpha: float = 0.05
) -> Tuple[float, ConfidenceInterval]:
    """
    Analytical confidence interval for J-statistic using normal approximation
    
    Args:
        results: Attack results
        alpha: Significance level
    
    Returns:
        (j_statistic, confidence_interval)
    """
    j, p0, p1 = compute_j_statistic(results)
    
    # Count samples in each world
    n0 = sum(1 for r in results if r.world_bit == 0)
    n1 = sum(1 for r in results if r.world_bit == 1)
    
    # Standard error via delta method
    se_p0 = math.sqrt(p0 * (1 - p0) / n0)
    se_p1 = math.sqrt(p1 * (1 - p1) / n1)
    se_j = math.sqrt(se_p0**2 + se_p1**2)
    
    # Normal approximation CI
    z = scipy_stats.norm.ppf(1 - alpha/2)
    ci = (j - z * se_j, j + z * se_j)
    
    # Bound to [0, 1]
    ci = (max(0, ci[0]), min(1, ci[1]))
    
    return j, ci

def sanity_check_j_statistic(results: AttackResults) -> Dict[str, bool]:
    """
    Comprehensive sanity checks for J-statistic validity
    
    Args:
        results: Attack results
    
    Returns:
        Dictionary of check names to pass/fail status
    """
    if not results:
        return {'has_results': False}
    
    j_stat, p0, p1 = compute_j_statistic(results)
    
    # Count world distribution
    n0 = sum(1 for r in results if r.world_bit == 0)
    n1 = sum(1 for r in results if r.world_bit == 1)
    
    checks = {
        'has_results': len(results) > 0,
        'j_in_range': 0 <= j_stat <= 1,
        'p0_valid': 0 <= p0 <= 1,
        'p1_valid': 0 <= p1 <= 1,
        'j_consistent': abs(j_stat - (p0 - p1)) < 1e-10,
        'sufficient_samples': len(results) >= 100,
        'both_worlds_present': n0 > 0 and n1 > 0,
        'balanced_worlds': 0.4 < n0/len(results) < 0.6,  # Roughly balanced
        'non_trivial': 0.01 < j_stat < 0.99,  # Not at extremes
        'sufficient_variation': 0 < p0 < 1 and 0 < p1 < 1
    }
    
    return checks

def interpret_cc_results(
    result: BootstrapResult,
    threshold_constructive: float = 0.95,
    threshold_destructive: float = 1.05
) -> CCInterpretation:
    """
    Interpret CC results with confidence assessment
    
    Args:
        result: Bootstrap results
        threshold_constructive: CC threshold for constructive regime
        threshold_destructive: CC threshold for destructive regime
    
    Returns:
        CCInterpretation with regime classification and recommendations
    """
    cc = result.cc_max
    ci = result.ci_cc_max
    
    # Determine regime with confidence
    if cc < threshold_constructive:
        regime = "constructive"
        confidence = 1.0 if ci[1] < threshold_constructive else (threshold_constructive - cc) / (ci[1] - ci[0])
        recommendation = "Deploy composition - synergistic protection detected"
    elif cc > threshold_destructive:
        regime = "destructive"
        confidence = 1.0 if ci[0] > threshold_destructive else (cc - threshold_destructive) / (ci[1] - ci[0])
        recommendation = "Avoid composition - interference detected"
    else:
        regime = "independent"
        confidence = 1.0 if ci[0] > threshold_constructive and ci[1] < threshold_destructive else 0.5
        recommendation = "Use single best guardrail - no interaction benefit"
    
    evidence = {
        'cc_max': cc,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'ci_width': ci[1] - ci[0],
        'delta_add': result.delta_add,
        'j_statistic': result.j_statistic,
        'n_sessions': result.n_sessions,
        'convergence': result.convergence_diagnostic
    }
    
    return CCInterpretation(
        regime=regime,
        confidence=confidence,
        recommendation=recommendation,
        evidence=evidence
    )

# Helper functions

def _estimate_block_size(results: AttackResults) -> int:
    """Estimate optimal block size for block bootstrap"""
    n = len(results)
    # Rule of thumb: n^(1/3) for moderate dependence
    return max(1, int(n**(1/3)))

def _percentile_ci(samples: np.ndarray, alpha: float) -> ConfidenceInterval:
    """Compute percentile confidence interval"""
    if len(samples) == 0:
        return (0, 0)
    lower = np.percentile(samples, 100 * alpha / 2)
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))

def _basic_ci(samples: np.ndarray, theta_hat: float, alpha: float) -> ConfidenceInterval:
    """Compute basic bootstrap confidence interval"""
    if len(samples) == 0:
        return (0, 0)
    percentiles = _percentile_ci(samples, alpha)
    lower = 2 * theta_hat - percentiles[1]
    upper = 2 * theta_hat - percentiles[0]
    return (float(lower), float(upper))

def _bca_ci(samples: np.ndarray, theta_hat: float, 
            original_data: AttackResults, alpha: float) -> ConfidenceInterval:
    """Bias-corrected and accelerated (BCa) confidence interval"""
    # Simplified BCa implementation
    # For full implementation, would need jackknife estimates
    if len(samples) == 0:
        return (0, 0)
    
    # Bias correction
    z0 = scipy_stats.norm.ppf((samples < theta_hat).mean())
    
    # For simplicity, use percentile with bias correction
    # Full BCa would compute acceleration parameter via jackknife
    z_alpha = scipy_stats.norm.ppf(alpha/2)
    z_1_alpha = scipy_stats.norm.ppf(1 - alpha/2)
    
    alpha_1 = scipy_stats.norm.cdf(z0 + (z0 + z_alpha))
    alpha_2 = scipy_stats.norm.cdf(z0 + (z0 + z_1_alpha))
    
    lower = np.percentile(samples, 100 * alpha_1)
    upper = np.percentile(samples, 100 * alpha_2)
    
    return (float(lower), float(upper))

def _compute_convergence_diagnostic(samples: np.ndarray) -> float:
    """Compute convergence diagnostic (coefficient of variation)"""
    if len(samples) == 0:
        return float('inf')
    mean = np.mean(samples)
    if abs(mean) < 1e-10:
        return float('inf')
    return float(np.std(samples) / abs(mean))

def _effective_sample_size(samples: np.ndarray) -> float:
    """Estimate effective sample size accounting for autocorrelation"""
    if len(samples) <= 1:
        return float(len(samples))
    
    # Simple ESS estimate using autocorrelation at lag 1
    from scipy.stats import pearsonr
    if len(samples) > 2:
        corr, _ = pearsonr(samples[:-1], samples[1:])
        # ESS = n / (1 + 2 * sum of autocorrelations)
        # Simplified: use only lag-1 correlation
        ess = len(samples) / (1 + 2 * abs(corr))
    else:
        ess = len(samples)
    
    return float(ess)

def _run_hypothesis_tests(j_emp: float, j_bootstrap: np.ndarray,
                         cc_emp: Optional[float], cc_bootstrap: Optional[np.ndarray],
                         alpha: float) -> Dict[str, Any]:
    """Run hypothesis tests on bootstrap samples"""
    tests = {}
    
    # Test 1: J significantly different from 0
    tests['j_nonzero'] = {
        'null_hypothesis': 'J = 0',
        'p_value': float((j_bootstrap <= 0).mean() * 2),  # Two-sided
        'reject': (j_bootstrap <= 0).mean() < alpha/2 or (j_bootstrap >= 0).mean() < alpha/2,
        'test_statistic': j_emp
    }
    
    # Test 2: CC significantly different from 1 (if computed)
    if cc_emp is not None and cc_bootstrap is not None and len(cc_bootstrap) > 0:
        tests['cc_not_independent'] = {
            'null_hypothesis': 'CC = 1',
            'p_value': float(min((cc_bootstrap <= 1).mean(), (cc_bootstrap >= 1).mean()) * 2),
            'reject': abs(cc_emp - 1) > 0.05,  # Using threshold from spec
            'test_statistic': cc_emp
        }
    
    return tests

# Export key functions
__all__ = [
    'BootstrapResult',
    'CCInterpretation',
    'compute_j_statistic',
    'compute_composability_coefficients',
    'bootstrap_ci_with_cc',
    'dkw_confidence_bound',
    'analytical_j_ci',
    'sanity_check_j_statistic',
    'interpret_cc_results'
]
