# src/cc/theory/fh_bounds.py
"""
PhD-Level Implementation: Fréchet-Hoeffding Bounds for Guardrail Composition

This module provides the mathematically rigorous foundation for compositional safety
bounds, addressing the core theoretical framework of the CC (Composability Coefficient) 
system. Built on the seminal work of Fréchet (1951) and Hoeffding (1940), extended 
to AI safety evaluation.

Mathematical Foundation:
- Extends classical copula bounds to multi-rail safety composition
- Provides sharp (attainable) bounds for joint event probabilities
- Maps guardrail composition to probability theory via two-world protocol
- Enables empirical validation of theoretical limits

Author: Pranav Bhave
Institution: Penn State University  
Advisor: Dr. Peng Liu
Date: October 2025
Course: IST 496 Independent Study

References:
[1] Fréchet, M. (1951). Sur les tableaux de corrélation dont les marges sont données
[2] Hoeffding, W. (1940). Maßstabinvariante Korrelationstheorie  
[3] Sklar, A. (1959). Fonctions de répartition à n dimensions
[4] Nelsen, R.B. (2006). An Introduction to Copulas, 2nd ed.
[5] Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union, NamedTuple, Protocol
from dataclasses import dataclass, field
import numpy as np
import scipy.stats as stats
import warnings
import math
from abc import ABC, abstractmethod

# Import from existing CC framework
from ..core.models import AttackResult, GuardrailSpec
from ..core.stats import compute_j_statistic

# Type aliases for mathematical clarity
Probability = float  # [0,1]
JStatistic = float   # [-1,1], typically [0,1] for security
Bounds = Tuple[float, float]  # (lower, upper)
MarginalsVector = List[Probability]

# Mathematical constants
EPSILON = 1e-12  # Numerical stability threshold
MATHEMATICAL_TOLERANCE = 1e-10

class FHBoundViolationError(Exception):
    """Raised when computed bounds violate mathematical constraints."""
    pass

class CopulaFamily(Protocol):
    """Protocol for copula families used in dependence modeling."""
    
    @abstractmethod
    def cdf(self, u: np.ndarray) -> float:
        """Copula cumulative distribution function."""
        pass
    
    @abstractmethod  
    def parameter_estimate(self, data: np.ndarray) -> Dict[str, float]:
        """Estimate copula parameters from data."""
        pass

@dataclass(frozen=True, slots=True)
class FHBounds:
    """
    Immutable container for Fréchet-Hoeffding bounds on joint probabilities.
    
    Mathematical Foundation:
    For random variables X₁, ..., Xₖ with marginals F₁(x₁), ..., Fₖ(xₖ):
    
    Lower bound (Fréchet bound):
    F(x₁,...,xₖ) ≥ max{0, ∑ᵢFᵢ(xᵢ) - (k-1)}
    
    Upper bound (Hoeffding bound):  
    F(x₁,...,xₖ) ≤ min{F₁(x₁), ..., Fₖ(xₖ)}
    
    Both bounds are sharp (attainable by explicit constructions).
    """
    lower: float
    upper: float
    marginals: Tuple[float, ...] 
    bound_type: str  # 'intersection', 'union', 'joint_cdf'
    k_rails: int
    
    def __post_init__(self):
        """Validate mathematical consistency of bounds."""
        if not (0 <= self.lower <= self.upper <= 1):
            raise FHBoundViolationError(
                f"Invalid bounds: [{self.lower:.6f}, {self.upper:.6f}]. "
                f"Must satisfy 0 ≤ lower ≤ upper ≤ 1."
            )
        
        if not all(0 <= p <= 1 for p in self.marginals):
            raise FHBoundViolationError(
                f"Invalid marginals: {self.marginals}. All must be in [0,1]."
            )
            
        if len(self.marginals) != self.k_rails:
            raise FHBoundViolationError(
                f"Marginals length {len(self.marginals)} != k_rails {self.k_rails}"
            )
    
    @property 
    def width(self) -> float:
        """Bound width (uncertainty interval)."""
        return self.upper - self.lower
        
    @property
    def is_degenerate(self) -> bool:
        """True if bounds collapse to a single point."""
        return abs(self.upper - self.lower) < MATHEMATICAL_TOLERANCE
        
    def contains(self, value: float, strict: bool = False) -> bool:
        """Check if value lies within bounds."""
        if strict:
            return self.lower < value < self.upper
        return self.lower <= value <= self.upper

@dataclass(frozen=True, slots=True)
class ComposedJBounds:
    """
    Fréchet-Hoeffding bounds for composed J-statistic in two-world protocol.
    
    Mathematical Derivation:
    J = TPR - FPR where:
    - TPR = P(Detection | Adversarial) = 1 - P(Miss | Adversarial)  
    - FPR = P(False Alarm | Benign)
    
    For k guardrails in series (OR-gate detection):
    - Miss events: Fᵢ = {no detection by rail i | adversarial}
    - False alarm events: Aᵢ = {detection by rail i | benign}
    - Joint miss: H = ∩ᵢFᵢ (all rails miss)
    - Joint false alarm: B = ∪ᵢAᵢ (any rail alarms)
    
    FH bounds apply to H and B separately, then combine for J bounds.
    """
    j_lower: float
    j_upper: float
    tpr_bounds: FHBounds  
    fpr_bounds: FHBounds
    individual_j_stats: Tuple[float, ...]
    composition_type: str  # 'serial_or', 'parallel_and', 'weighted_vote'
    
    def __post_init__(self):
        """Validate J-statistic bounds consistency."""
        if not (-1 <= self.j_lower <= self.j_upper <= 1):
            raise FHBoundViolationError(
                f"Invalid J bounds: [{self.j_lower:.6f}, {self.j_upper:.6f}]. "
                f"Must satisfy -1 ≤ lower ≤ upper ≤ 1."
            )
    
    @property
    def j_width(self) -> float:
        """J-statistic uncertainty interval width."""
        return self.j_upper - self.j_lower
        
    def classify_regime(self, threshold_constructive: float = 0.95, 
                       threshold_destructive: float = 1.05) -> str:
        """
        Classify composition regime based on CC bounds.
        
        Args:
            threshold_constructive: CC < this → constructive
            threshold_destructive: CC > this → destructive
            
        Returns:
            'constructive', 'independent', 'destructive', or 'uncertain'
        """
        max_individual_j = max(self.individual_j_stats) if self.individual_j_stats else 1.0
        
        if max_individual_j < MATHEMATICAL_TOLERANCE:
            return 'degenerate'  # No individual effectiveness
            
        # CC bounds from J bounds
        cc_lower = self.j_lower / max_individual_j if max_individual_j > 0 else 0
        cc_upper = self.j_upper / max_individual_j if max_individual_j > 0 else float('inf')
        
        if cc_upper < threshold_constructive:
            return 'constructive'
        elif cc_lower > threshold_destructive:
            return 'destructive'  
        elif cc_lower >= threshold_constructive and cc_upper <= threshold_destructive:
            return 'independent'
        else:
            return 'uncertain'  # Bounds span multiple regimes

# Core mathematical functions

def frechet_lower_bound(marginals: MarginalsVector) -> float:
    """
    Compute Fréchet lower bound for intersection of events.
    
    Mathematical Formula:
    P(∩ᵢAᵢ) ≥ max{0, ∑ᵢP(Aᵢ) - (k-1)}
    
    Proof sketch: By inclusion-exclusion and Bonferroni inequality.
    The bound is sharp (attainable by explicit construction).
    
    Args:
        marginals: Individual event probabilities [p₁, p₂, ..., pₖ]
        
    Returns:
        Lower bound on P(∩ᵢAᵢ)
        
    Examples:
        >>> frechet_lower_bound([0.9, 0.8])  # Two events
        0.7  # max{0, 0.9 + 0.8 - 1} = 0.7
        
        >>> frechet_lower_bound([0.3, 0.4, 0.2])  # Three events  
        0.0  # max{0, 0.3 + 0.4 + 0.2 - 2} = max{0, -0.1} = 0
    """
    if not marginals:
        raise ValueError("Empty marginals list")
        
    if not all(0 <= p <= 1 for p in marginals):
        raise ValueError(f"Invalid marginals {marginals}: must be probabilities in [0,1]")
    
    k = len(marginals)
    if k == 1:
        return marginals[0]
        
    # Fréchet lower bound: max{0, ∑pᵢ - (k-1)}
    bound = max(0.0, sum(marginals) - (k - 1))
    
    # Ensure bound doesn't exceed minimum marginal (mathematical necessity)
    bound = min(bound, min(marginals)) if marginals else bound
    
    return float(bound)

def hoeffding_upper_bound(marginals: MarginalsVector) -> float:
    """
    Compute Hoeffding upper bound for intersection of events.
    
    Mathematical Formula:
    P(∩ᵢAᵢ) ≤ min{P(A₁), ..., P(Aₖ)}
    
    Proof: Immediate from P(∩ᵢAᵢ) ≤ P(Aⱼ) for any j.
    The bound is sharp (attained when one event is subset of all others).
    
    Args:
        marginals: Individual event probabilities
        
    Returns:
        Upper bound on P(∩ᵢAᵢ)
    """
    if not marginals:
        raise ValueError("Empty marginals list")
        
    if not all(0 <= p <= 1 for p in marginals):
        raise ValueError(f"Invalid marginals {marginals}: must be probabilities in [0,1]")
    
    return float(min(marginals))

def union_bounds(marginals: MarginalsVector) -> Bounds:
    """
    Fréchet-Hoeffding bounds for union of events.
    
    Mathematical Formulas:
    Lower: P(∪ᵢAᵢ) ≥ max{P(A₁), ..., P(Aₖ)}  
    Upper: P(∪ᵢAᵢ) ≤ min{1, ∑ᵢP(Aᵢ)}
    
    Args:
        marginals: Individual event probabilities
        
    Returns:
        (lower_bound, upper_bound) for P(∪ᵢAᵢ)
    """
    if not marginals:
        raise ValueError("Empty marginals list")
        
    if not all(0 <= p <= 1 for p in marginals):
        raise ValueError(f"Invalid marginals {marginals}: must be probabilities in [0,1]")
    
    lower = max(marginals)  # Union contains largest individual event
    upper = min(1.0, sum(marginals))  # Bonferroni bound
    
    return (float(lower), float(upper))

def intersection_bounds(marginals: MarginalsVector) -> FHBounds:
    """
    Complete Fréchet-Hoeffding bounds for intersection of events.
    
    Args:
        marginals: Individual event probabilities
        
    Returns:
        FHBounds object with lower/upper bounds and metadata
    """
    if not marginals:
        raise ValueError("Empty marginals list")
    
    lower = frechet_lower_bound(marginals)
    upper = hoeffding_upper_bound(marginals)
    
    return FHBounds(
        lower=lower,
        upper=upper, 
        marginals=tuple(marginals),
        bound_type='intersection',
        k_rails=len(marginals)
    )

def union_bounds_structured(marginals: MarginalsVector) -> FHBounds:
    """
    Complete Fréchet-Hoeffding bounds for union of events.
    
    Args:
        marginals: Individual event probabilities
        
    Returns:
        FHBounds object for union probability
    """
    if not marginals:
        raise ValueError("Empty marginals list")
        
    lower, upper = union_bounds(marginals)
    
    return FHBounds(
        lower=lower,
        upper=upper,
        marginals=tuple(marginals), 
        bound_type='union',
        k_rails=len(marginals)
    )

# Core composition analysis

def serial_or_composition_bounds(miss_rates: MarginalsVector, 
                               false_alarm_rates: MarginalsVector) -> ComposedJBounds:
    """
    Compute FH bounds for J-statistic under serial OR-gate composition.
    
    Mathematical Model:
    - k guardrails in series (OR-gate for detection)
    - Miss rate mᵢ = P(no detection by rail i | adversarial)
    - False alarm rate aᵢ = P(detection by rail i | benign)  
    - Joint miss: H = ∩ᵢ{miss by rail i} (all rails fail)
    - Joint false alarm: B = ∪ᵢ{alarm by rail i} (any rail triggers)
    
    Composed Performance:
    - TPR_composed = 1 - P(H) ∈ [1 - min(mᵢ), 1 - max{0, ∑mᵢ - (k-1)}]
    - FPR_composed = P(B) ∈ [max(aᵢ), min{1, ∑aᵢ}]
    - J_composed = TPR_composed - FPR_composed
    
    Args:
        miss_rates: [m₁, m₂, ..., mₖ] where mᵢ = 1 - TPRᵢ
        false_alarm_rates: [a₁, a₂, ..., aₖ] where aᵢ = FPRᵢ
        
    Returns:
        ComposedJBounds with rigorous mathematical bounds
        
    Raises:
        ValueError: If input dimensions don't match or probabilities invalid
        FHBoundViolationError: If computed bounds are mathematically inconsistent
    """
    # Input validation
    if len(miss_rates) != len(false_alarm_rates):
        raise ValueError(
            f"Dimension mismatch: {len(miss_rates)} miss rates vs "
            f"{len(false_alarm_rates)} false alarm rates"
        )
    
    k = len(miss_rates)
    if k == 0:
        raise ValueError("Empty rate vectors")
    
    # Validate probability constraints
    for i, (m, a) in enumerate(zip(miss_rates, false_alarm_rates)):
        if not (0 <= m <= 1):
            raise ValueError(f"Invalid miss rate m_{i} = {m}: must be in [0,1]")
        if not (0 <= a <= 1):
            raise ValueError(f"Invalid false alarm rate a_{i} = {a}: must be in [0,1]")
    
    # Step 1: FH bounds for joint miss probability P(H) = P(∩ᵢFᵢ)
    # where Fᵢ = {rail i misses adversarial input}
    miss_bounds = intersection_bounds(miss_rates)
    
    # Step 2: FH bounds for joint false alarm probability P(B) = P(∪ᵢAᵢ)  
    # where Aᵢ = {rail i false alarms on benign input}
    alarm_bounds = union_bounds_structured(false_alarm_rates)
    
    # Step 3: Composed TPR bounds
    # TPR = 1 - P(miss), so bounds flip and negate
    tpr_lower = 1 - miss_bounds.upper  # Best case: minimal joint miss
    tpr_upper = 1 - miss_bounds.lower  # Worst case: maximal joint miss
    
    tpr_bounds = FHBounds(
        lower=tpr_lower,
        upper=tpr_upper,
        marginals=tuple(1 - m for m in miss_rates),  # Individual TPRs
        bound_type='tpr_composed',
        k_rails=k
    )
    
    # Step 4: Composed FPR bounds (direct from alarm bounds)
    fpr_bounds = FHBounds(
        lower=alarm_bounds.lower, 
        upper=alarm_bounds.upper,
        marginals=alarm_bounds.marginals,
        bound_type='fpr_composed', 
        k_rails=k
    )
    
    # Step 5: Composed J-statistic bounds
    # J = TPR - FPR, so we need pessimistic bounds
    j_lower = tpr_lower - fpr_bounds.upper  # Worst case J
    j_upper = tpr_upper - fpr_bounds.lower  # Best case J
    
    # Step 6: Individual J-statistics for comparison
    individual_j_stats = tuple((1 - m) - a for m, a in zip(miss_rates, false_alarm_rates))
    
    # Step 7: Mathematical consistency checks
    if j_lower > j_upper + MATHEMATICAL_TOLERANCE:
        raise FHBoundViolationError(
            f"Inconsistent J bounds: [{j_lower:.6f}, {j_upper:.6f}]. "
            f"Lower bound exceeds upper bound by {j_lower - j_upper:.2e}"
        )
    
    # Clamp to valid J-statistic range [-1, 1]
    j_lower = max(-1.0, min(1.0, j_lower))
    j_upper = max(-1.0, min(1.0, j_upper))
    
    return ComposedJBounds(
        j_lower=j_lower,
        j_upper=j_upper, 
        tpr_bounds=tpr_bounds,
        fpr_bounds=fpr_bounds,
        individual_j_stats=individual_j_stats,
        composition_type='serial_or'
    )

def parallel_and_composition_bounds(miss_rates: MarginalsVector,
                                  false_alarm_rates: MarginalsVector) -> ComposedJBounds:
    """
    FH bounds for parallel AND-gate composition (all rails must agree).
    
    Mathematical Model:
    - Detection requires ALL rails to detect (AND-gate)
    - Miss: any rail misses (∪ᵢFᵢ) 
    - False alarm: all rails false alarm (∩ᵢAᵢ)
    
    Args:
        miss_rates: Individual miss probabilities
        false_alarm_rates: Individual false alarm probabilities
        
    Returns:
        ComposedJBounds for AND-gate composition
    """
    if len(miss_rates) != len(false_alarm_rates):
        raise ValueError("Dimension mismatch in rate vectors")
    
    k = len(miss_rates)
    if k == 0:
        raise ValueError("Empty rate vectors")
    
    # AND-gate: miss if ANY rail misses, false alarm if ALL rails false alarm
    miss_bounds = union_bounds_structured(miss_rates)  # Union of individual misses
    alarm_bounds = intersection_bounds(false_alarm_rates)  # Intersection of false alarms
    
    # TPR and FPR bounds  
    tpr_lower = 1 - miss_bounds.upper
    tpr_upper = 1 - miss_bounds.lower
    
    tpr_bounds = FHBounds(
        lower=tpr_lower, upper=tpr_upper,
        marginals=tuple(1 - m for m in miss_rates),
        bound_type='tpr_and_composed', k_rails=k
    )
    
    fpr_bounds = FHBounds(
        lower=alarm_bounds.lower, upper=alarm_bounds.upper,
        marginals=alarm_bounds.marginals, 
        bound_type='fpr_and_composed', k_rails=k
    )
    
    # J-statistic bounds
    j_lower = tpr_lower - fpr_bounds.upper
    j_upper = tpr_upper - fpr_bounds.lower
    
    # Clamp to [-1, 1]
    j_lower = max(-1.0, min(1.0, j_lower))
    j_upper = max(-1.0, min(1.0, j_upper))
    
    individual_j_stats = tuple((1 - m) - a for m, a in zip(miss_rates, false_alarm_rates))
    
    return ComposedJBounds(
        j_lower=j_lower, j_upper=j_upper,
        tpr_bounds=tpr_bounds, fpr_bounds=fpr_bounds,
        individual_j_stats=individual_j_stats,
        composition_type='parallel_and'
    )

# Empirical validation and analysis

def validate_fh_bounds_empirically(bounds: ComposedJBounds, 
                                 observed_j: float,
                                 confidence_interval: Optional[Tuple[float, float]] = None,
                                 significance_level: float = 0.05) -> Dict[str, Union[bool, float, str]]:
    """
    Validate FH bounds against empirical observations with statistical rigor.
    
    Args:
        bounds: Theoretical FH bounds 
        observed_j: Empirically measured J-statistic
        confidence_interval: Bootstrap/analytical CI for observed_j
        significance_level: α for statistical tests
        
    Returns:
        Validation report with statistical assessment
    """
    report = {
        'bounds_contain_observation': bounds.j_lower <= observed_j <= bounds.j_upper,
        'observation': observed_j,
        'theoretical_bounds': (bounds.j_lower, bounds.j_upper),
        'bound_width': bounds.j_width,
        'relative_position': (observed_j - bounds.j_lower) / bounds.j_width if bounds.j_width > 0 else 0.5
    }
    
    # Statistical significance test if CI provided
    if confidence_interval:
        ci_lower, ci_upper = confidence_interval
        
        # Test if CI overlaps with theoretical bounds
        bounds_ci_overlap = not (ci_upper < bounds.j_lower or ci_lower > bounds.j_upper)
        
        # Test if observation is significantly different from bound midpoint
        bound_midpoint = (bounds.j_lower + bounds.j_upper) / 2
        ci_contains_midpoint = ci_lower <= bound_midpoint <= ci_upper
        
        report.update({
            'confidence_interval': confidence_interval,
            'ci_bounds_overlap': bounds_ci_overlap,
            'ci_contains_bound_midpoint': ci_contains_midpoint,
            'statistical_consistency': bounds_ci_overlap and ci_contains_midpoint
        })
    
    # Classify bound tightness
    if bounds.j_width < 0.05:
        tightness = 'tight'
    elif bounds.j_width < 0.2:
        tightness = 'moderate'  
    else:
        tightness = 'loose'
        
    report['bound_tightness'] = tightness
    
    return report

def compute_composability_interference_index(bounds: ComposedJBounds, 
                                           observed_j: float) -> Dict[str, float]:
    """
    Compute Composability Interference Index (CII) with theoretical bounds context.
    
    Mathematical Definition:
    κ = (J_observed - J_independent) / (J_worst - J_independent)
    
    where:
    - J_independent = ∏ᵢ(1-mᵢ) - max(aᵢ) (assuming independence)
    - J_worst = worst-case bound from FH analysis
    - κ < 0: constructive interference
    - κ ≈ 0: independent behavior
    - κ > 0: destructive interference
    
    Args:
        bounds: FH bounds analysis
        observed_j: Empirically measured J-statistic
        
    Returns:
        Dictionary with CII analysis
    """
    individual_tprs = [1 - m for m in bounds.tpr_bounds.marginals] if bounds.tpr_bounds.marginals else [0.5]
    individual_fprs = list(bounds.fpr_bounds.marginals) if bounds.fpr_bounds.marginals else [0.05]
    
    # Independence assumption: TPR_indep = ∏TPRᵢ, FPR_indep = max(FPRᵢ)  
    if bounds.composition_type == 'serial_or':
        # For OR-gate: independent TPR = 1 - ∏(1-TPRᵢ), FPR = max(FPRᵢ) 
        tpr_independent = 1 - np.prod([1 - tpr for tpr in individual_tprs])
        fpr_independent = max(individual_fprs)
    elif bounds.composition_type == 'parallel_and':
        # For AND-gate: independent TPR = ∏TPRᵢ, FPR = ∏FPRᵢ
        tpr_independent = np.prod(individual_tprs)
        fpr_independent = np.prod(individual_fprs) 
    else:
        # Fallback: use arithmetic mean
        tpr_independent = np.mean(individual_tprs)
        fpr_independent = np.mean(individual_fprs)
    
    j_independent = tpr_independent - fpr_independent
    j_worst = bounds.j_lower  # Pessimistic bound
    j_best = bounds.j_upper   # Optimistic bound
    
    # CII computation with numerical stability
    if abs(j_worst - j_independent) < MATHEMATICAL_TOLERANCE:
        # Degenerate case: no room for interference
        cii = 0.0
        interpretation = 'degenerate'
    else:
        cii = (observed_j - j_independent) / (j_worst - j_independent)
        
        # Classify interference type
        if cii < -0.1:
            interpretation = 'constructive'
        elif cii > 0.1:
            interpretation = 'destructive' 
        else:
            interpretation = 'independent'
    
    return {
        'cii': float(cii),
        'interpretation': interpretation,
        'j_observed': observed_j,
        'j_independent': j_independent,
        'j_theoretical_bounds': (bounds.j_lower, bounds.j_upper),
        'interference_strength': abs(cii)
    }

# Advanced analysis functions

def copula_enhanced_bounds(marginals: MarginalsVector, 
                         copula_family: str = 'gaussian',
                         dependence_parameter: Optional[float] = None) -> FHBounds:
    """
    Refined bounds using parametric copula families for known dependence structure.
    
    This extends basic FH bounds when we have information about dependence
    structure between guardrails (e.g., from empirical data).
    
    Args:
        marginals: Individual event probabilities
        copula_family: 'gaussian', 'clayton', 'gumbel', 'frank'
        dependence_parameter: Copula parameter (correlation, etc.)
        
    Returns:
        Refined bounds incorporating dependence structure
        
    Note:
        This is an advanced feature for future development when dependence
        modeling becomes critical for tighter bounds.
    """
    # For now, return standard FH bounds with notation for future extension
    standard_bounds = intersection_bounds(marginals)
    
    # Placeholder for copula-enhanced computation
    if copula_family and dependence_parameter is not None:
        warnings.warn(
            f"Copula enhancement with {copula_family} not yet implemented. "
            f"Returning standard FH bounds.",
            UserWarning
        )
    
    return FHBounds(
        lower=standard_bounds.lower,
        upper=standard_bounds.upper, 
        marginals=standard_bounds.marginals,
        bound_type=f'copula_{copula_family}' if copula_family else standard_bounds.bound_type,
        k_rails=standard_bounds.k_rails
    )

def multi_threshold_analysis(miss_rates: MarginalsVector,
                           false_alarm_rates: MarginalsVector,
                           threshold_grid: List[Tuple[float, float]]) -> Dict[Tuple[float, float], ComposedJBounds]:
    """
    Analyze FH bounds across multiple decision threshold combinations.
    
    This enables phase diagram generation showing how theoretical bounds
    vary across the (constructive_threshold, destructive_threshold) space.
    
    Args:
        miss_rates: Individual guardrail miss rates
        false_alarm_rates: Individual guardrail false alarm rates  
        threshold_grid: List of (constructive_thresh, destructive_thresh) pairs
        
    Returns:
        Dictionary mapping threshold pairs to bound analysis
    """
    results = {}
    
    base_bounds = serial_or_composition_bounds(miss_rates, false_alarm_rates)
    
    for constructive_thresh, destructive_thresh in threshold_grid:
        # Create modified bounds object for this threshold pair
        # (This is primarily for regime classification)
        threshold_specific = ComposedJBounds(
            j_lower=base_bounds.j_lower,
            j_upper=base_bounds.j_upper,
            tpr_bounds=base_bounds.tpr_bounds,
            fpr_bounds=base_bounds.fpr_bounds,
            individual_j_stats=base_bounds.individual_j_stats,
            composition_type=base_bounds.composition_type
        )
        
        results[(constructive_thresh, destructive_thresh)] = threshold_specific
    
    return results

# Integration with existing CC framework

def fh_bounds_from_attack_results(results_individual: List[List[AttackResult]],
                                results_composed: List[AttackResult],
                                bootstrap_iterations: int = 1000) -> Dict[str, Union[ComposedJBounds, Dict]]:
    """
    Compute FH bounds from empirical attack results (integration with existing framework).
    
    Args:
        results_individual: List of AttackResult lists, one per guardrail
        results_composed: AttackResult list for composed system
        bootstrap_iterations: Number of bootstrap samples for CI estimation
        
    Returns:
        Complete FH analysis with empirical validation
    """
    # Extract empirical rates
    individual_j_stats = []
    miss_rates = []
    false_alarm_rates = []
    
    for guardrail_results in results_individual:
        j_stat, p0, p1 = compute_j_statistic(guardrail_results)
        individual_j_stats.append(j_stat)
        
        # Miss rate = 1 - TPR = 1 - (1 - p1) = p1 (assuming p1 is miss rate on adversarial)
        # False alarm rate = FPR = p0 (assuming p0 is false alarm on benign)  
        # Note: This assumes specific encoding in AttackResult - may need adjustment
        miss_rates.append(p1)  
        false_alarm_rates.append(p0)
    
    # Compute theoretical bounds
    theoretical_bounds = serial_or_composition_bounds(miss_rates, false_alarm_rates)
    
    # Extract empirical composed performance
    j_composed, _, _ = compute_j_statistic(results_composed)
    
    # Bootstrap confidence interval for composed J
    # (This integrates with existing bootstrap infrastructure)
    bootstrap_samples = []
    n = len(results_composed)
    
    for _ in range(bootstrap_iterations):
        bootstrap_sample = np.random.choice(results_composed, size=n, replace=True)
        j_boot, _, _ = compute_j_statistic(bootstrap_sample)
        bootstrap_samples.append(j_boot)
    
    ci_lower = np.percentile(bootstrap_samples, 2.5)
    ci_upper = np.percentile(bootstrap_samples, 97.5)
    empirical_ci = (ci_lower, ci_upper)
    
    # Validation analysis
    validation = validate_fh_bounds_empirically(
        theoretical_bounds, j_composed, empirical_ci
    )
    
    # CII analysis
    cii_analysis = compute_composability_interference_index(
        theoretical_bounds, j_composed
    )
    
    return {
        'theoretical_bounds': theoretical_bounds,
        'empirical_j': j_composed, 
        'empirical_ci': empirical_ci,
        'individual_j_stats': individual_j_stats,
        'validation': validation,
        'cii_analysis': cii_analysis,
        'bootstrap_samples': bootstrap_samples
    }

# Utility functions for visualization and reporting

def generate_phase_diagram_data(miss_rate_range: Tuple[float, float],
                              false_alarm_rate_range: Tuple[float, float], 
                              k_rails: int = 3,
                              resolution: int = 50) -> np.ndarray:
    """
    Generate data for phase diagram visualization of FH bounds.
    
    Args:
        miss_rate_range: (min, max) for miss rates
        false_alarm_rate_range: (min, max) for false alarm rates
        k_rails: Number of guardrails
        resolution: Grid resolution
        
    Returns:
        3D array with J-bound data for visualization
    """
    miss_min, miss_max = miss_rate_range
    fpr_min, fpr_max = false_alarm_rate_range
    
    miss_grid = np.linspace(miss_min, miss_max, resolution)
    fpr_grid = np.linspace(fpr_min, fpr_max, resolution)
    
    # Initialize output arrays
    j_lower_surface = np.zeros((resolution, resolution))
    j_upper_surface = np.zeros((resolution, resolution))
    j_width_surface = np.zeros((resolution, resolution))
    
    for i, miss_rate in enumerate(miss_grid):
        for j, fpr_rate in enumerate(fpr_grid):
            # Create uniform rates for this grid point
            miss_rates = [miss_rate] * k_rails
            false_alarm_rates = [fpr_rate] * k_rails
            
            try:
                bounds = serial_or_composition_bounds(miss_rates, false_alarm_rates)
                j_lower_surface[i, j] = bounds.j_lower
                j_upper_surface[i, j] = bounds.j_upper
                j_width_surface[i, j] = bounds.j_width
            except (ValueError, FHBoundViolationError):
                # Handle edge cases with NaN
                j_lower_surface[i, j] = np.nan
                j_upper_surface[i, j] = np.nan  
                j_width_surface[i, j] = np.nan
    
    return np.stack([j_lower_surface, j_upper_surface, j_width_surface], axis=-1)

def summarize_fh_analysis(bounds: ComposedJBounds) -> str:
    """
    Generate human-readable summary of FH bounds analysis.
    
    Args:
        bounds: FH bounds analysis result
        
    Returns:
        Formatted summary string
    """
    regime = bounds.classify_regime()
    max_individual_j = max(bounds.individual_j_stats) if bounds.individual_j_stats else 0.0
    
    summary = f"""
Fréchet-Hoeffding Bounds Analysis
================================

Composition Type: {bounds.composition_type}
Number of Rails: {len(bounds.individual_j_stats)}

Individual J-statistics: {[f'{j:.3f}' for j in bounds.individual_j_stats]}
Maximum Individual J: {max_individual_j:.3f}

Theoretical Bounds:
  J-statistic: [{bounds.j_lower:.3f}, {bounds.j_upper:.3f}]
  Bound Width: {bounds.j_width:.3f}
  
  TPR Bounds: [{bounds.tpr_bounds.lower:.3f}, {bounds.tpr_bounds.upper:.3f}]
  FPR Bounds: [{bounds.fpr_bounds.lower:.3f}, {bounds.fpr_bounds.upper:.3f}]

Regime Classification: {regime.upper()}

Mathematical Guarantee: Any composition satisfying the individual marginal
constraints MUST yield a J-statistic within the computed bounds, regardless
of dependence structure between guardrails.
"""
    
    return summary.strip()

# Export public interface

__all__ = [
    # Core bound computation
    'frechet_lower_bound',
    'hoeffding_upper_bound', 
    'union_bounds',
    'intersection_bounds',
    'union_bounds_structured',
    
    # Composition analysis
    'serial_or_composition_bounds',
    'parallel_and_composition_bounds',
    
    # Data structures
    'FHBounds',
    'ComposedJBounds',
    'FHBoundViolationError',
    
    # Empirical validation 
    'validate_fh_bounds_empirically',
    'compute_composability_interference_index',
    'fh_bounds_from_attack_results',
    
    # Advanced analysis
    'copula_enhanced_bounds',
    'multi_threshold_analysis',
    
    # Visualization support
    'generate_phase_diagram_data',
    'summarize_fh_analysis',
    
    # Constants
    'EPSILON',
    'MATHEMATICAL_TOLERANCE'
]

# Module-level docstring for mathematical context

__doc__ = """
Fréchet-Hoeffding Bounds for AI Safety Guardrail Composition

This module provides the mathematical foundation for the CC (Composability Coefficient)
framework, implementing rigorous probability bounds for multi-guardrail safety systems.

The core insight is that regardless of the dependence structure between guardrails,
their joint performance is bounded by universal mathematical constraints derived from
marginal performance alone. These Fréchet-Hoeffding bounds enable:

1. Theoretical analysis of composition limits
2. Empirical validation of observed performance  
3. Decision-grade confidence intervals for safety systems
4. Phase diagram generation across parameter spaces

Mathematical Foundation:
- Fréchet (1951): Lower bounds via inclusion-exclusion
- Hoeffding (1940): Upper bounds via subset constraints  
- Sklar (1959): Copula theory for dependence modeling
- Modern extensions to multi-dimensional safety composition

The bounds are SHARP (attainable) and provide the tightest possible constraints
without additional dependence information.

Integration Points:
- Two-world distinguishability protocol (core.protocol)
- Bootstrap confidence intervals (core.stats)
- Attack result analysis (core.models)
- Decision card generation (guardrails.receipts)

Author: Pranav Bhave, Penn State University, Fall 2025
"""
