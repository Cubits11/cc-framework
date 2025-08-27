# src/cc/core/stats.py
"""
Module: stats
Purpose: Bootstrap CI, DKW backup, and J-statistic computation
Dependencies: numpy, scipy, math
Author: Pranav Bhave
Date: 2025-08-27
"""
from __future__ import annotations
import numpy as np
import math
from typing import Tuple, List, Optional
from scipy import stats
from dataclasses import dataclass

@dataclass
class BootstrapResult:
    """Results of bootstrap analysis"""
    statistic: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    bootstrap_samples: np.ndarray
    method: str = "percentile"

def youden_j_statistic(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Youden's J-statistic from binary classification scores
    
    Returns:
        j_statistic: Maximum J value
        optimal_threshold: Threshold that maximizes J
        auc: Area under ROC curve
    """
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Compute J for each threshold
    j_scores = tpr - fpr
    
    # Find optimal threshold
    optimal_idx = np.argmax(j_scores)
    j_statistic = j_scores[optimal_idx]
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute AUC
    auc_score = auc(fpr, tpr)
    
    return j_statistic, optimal_threshold, auc_score

def bootstrap_ci_j_statistic(
    world_0_results: np.ndarray,
    world_1_results: np.ndarray, 
    B: int = 2000,
    alpha: float = 0.05,
    method: str = "percentile"
) -> BootstrapResult:
    """
    Bootstrap confidence interval for J-statistic difference
    
    Args:
        world_0_results: Binary outcomes from world 0 (no protection)
        world_1_results: Binary outcomes from world 1 (with protection)
        B: Number of bootstrap samples
        alpha: Significance level
        method: CI method ("percentile", "bca", or "basic")
    """
    n0, n1 = len(world_0_results), len(world_1_results)
    
    # Compute empirical J-statistic
    p0_hat = np.mean(world_0_results)
    p1_hat = np.mean(world_1_results) 
    j_empirical = p0_hat - p1_hat
    
    # Bootstrap resampling
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    bootstrap_js = []
    
    for _ in range(B):
        # Resample with replacement
        boot_0 = rng.choice(world_0_results, size=n0, replace=True)
        boot_1 = rng.choice(world_1_results, size=n1, replace=True)
        
        # Compute bootstrap J
        boot_j = np.mean(boot_0) - np.mean(boot_1)
        bootstrap_js.append(boot_j)
    
    bootstrap_js = np.array(bootstrap_js)
    
    # Compute confidence interval
    if method == "percentile":
        ci_lower = np.percentile(bootstrap_js, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_js, 100 * (1 - alpha / 2))
    elif method == "basic":
        ci_lower = 2 * j_empirical - np.percentile(bootstrap_js, 100 * (1 - alpha / 2))
        ci_upper = 2 * j_empirical - np.percentile(bootstrap_js, 100 * alpha / 2)
    elif method == "bca":
        # Bias-corrected and accelerated
        ci_lower, ci_upper = _bootstrap_bca_ci(
            world_0_results, world_1_results, bootstrap_js, j_empirical, alpha
        )
    else:
        raise ValueError(f"Unknown CI method: {method}")
    
    return BootstrapResult(
        statistic=j_empirical,
        ci_lower=ci_lower,
        ci_upper=ci_upper, 
        ci_width=ci_upper - ci_lower,
        bootstrap_samples=bootstrap_js,
        method=method
    )

def _bootstrap_bca_ci(
    world_0: np.ndarray, 
    world_1: np.ndarray,
    bootstrap_samples: np.ndarray,
    theta_hat: float,
    alpha: float
) -> Tuple[float, float]:
    """Bias-corrected and accelerated confidence interval"""
    B = len(bootstrap_samples)
    
    # Bias correction
    bias_corr = stats.norm.ppf((bootstrap_samples < theta_hat).mean())
    
    # Acceleration (jackknife)
    n = len(world_0) + len(world_1)
    jack_estimates = []
    
    # Jackknife for acceleration
    for i in range(min(n, 100)):  # Limit for computational efficiency
        if i < len(world_0):
            jack_0 = np.delete(world_0, i)
            jack_1 = world_1
        else:
            jack_0 = world_0
            jack_1 = np.delete(world_1, i - len(world_0))
        
        jack_est = np.mean(jack_0) - np.mean(jack_1)
        jack_estimates.append(jack_est)
    
    jack_estimates = np.array(jack_estimates)
    jack_mean = np.mean(jack_estimates)
    
    # Acceleration
    acceleration = np.sum((jack_mean - jack_estimates)**3) / (6 * (np.sum((jack_mean - jack_estimates)**2))**1.5)
    
    # Adjust percentiles
    z_alpha_2 = stats.norm.ppf(alpha / 2)
    z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
    
    alpha_1 = stats.norm.cdf(bias_corr + (bias_corr + z_alpha_2)/(1 - acceleration * (bias_corr + z_alpha_2)))
    alpha_2 = stats.norm.cdf(bias_corr + (bias_corr + z_1_alpha_2)/(1 - acceleration * (bias_corr + z_1_alpha_2)))
    
    alpha_1 = max(0, min(1, alpha_1))
    alpha_2 = max(0, min(1, alpha_2))
    
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha_1)
    ci_upper = np.percentile(bootstrap_samples, 100 * alpha_2)
    
    return ci_lower, ci_upper

def dkw_confidence_interval(empirical_cdf: np.ndarray, n: int, alpha: float = 0.05) -> float:
    """
    Dvoretzky-Kiefer-Wolfowitz uniform confidence band halfwidth
    """
    epsilon = math.sqrt(2 * math.log(4 / alpha) / max(n, 1))
    return epsilon

def compute_composability_coefficients(
    j_individual: List[float],
    j_composed: float
) -> Dict[str, float]:
    """
    Compute all composability coefficient variants
    
    Args:
        j_individual: List of J-statistics for individual guardrails
        j_composed: J-statistic for composed guardrails
    """
    if not j_individual:
        raise ValueError("At least one individual J-statistic required")
    
    j_max_individual = max(j_individual)
    j_min_individual = min(j_individual)
    
    # Primary metrics from the framework
    cc_max = j_composed / j_max_individual if j_max_individual > 0 else 0.0
    delta_add = j_composed - j_max_individual
    
    # Secondary metrics
    j_multiplicative = np.prod(j_individual) if len(j_individual) > 1 else j_individual[0]
    cc_multiplicative = j_composed / j_multiplicative if j_multiplicative > 0 else 0.0
    
    # Theoretical bounds
    j_theoretical_max = min(sum(j_individual), 1.0)
    j_theoretical_add = min(sum(j_individual) - np.prod(j_individual), 1.0)
    
    return {
        "cc_max": cc_max,
        "delta_add": delta_add, 
        "cc_multiplicative": cc_multiplicative,
        "j_composed": j_composed,
        "j_max_individual": j_max_individual,
        "j_theoretical_max": j_theoretical_max,
        "j_theoretical_add": j_theoretical_add,
        "efficiency_ratio": j_composed / j_theoretical_max if j_theoretical_max > 0 else 0.0
    }
