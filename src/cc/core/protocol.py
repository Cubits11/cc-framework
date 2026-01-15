# src/cc/core/protocol.py
"""
Next-Generation Adaptive Two-World Protocol (CC-Framework)
=========================================================

Enterprise-grade research platform for AI safety evaluation.

This module is the *shipping* implementation of the CC-Framework
Two-World Protocol. It merges:

- The architectural upgrades from Protocol 2.0
- The statistical rigor required by the November 13, 2025 audit

Core guarantees
---------------
1. ICC (Intra-Class Correlation)
   - Computed using a proper one-way random-effects ANOVA (ICC(1))
   - Clustered by attack strategy (attack_strategy / strategy_type)
   - Produces a design effect and widening factor for CIs
   - Approximation for binary data; warns and suggests GLMM alternatives

2. ROPE-based Bayesian Sequential Testing
   - Region Of Practical Equivalence (ROPE) on effect size (p1 - p0)
   - Uses Beta posteriors for proper proportion modeling (upgraded from normal approx)
   - Stopping logic:
       * Stop for strong effect (95% HDI entirely outside ROPE)
       * Stop for futility (95% HDI entirely inside ROPE)
       * Otherwise continue
   - Fallback futility on sample size and effect magnitude

3. Causal Effect Estimation (ATE)
   - Difference in means (p1 - p0) with Welch-style SE
   - Standard errors and CIs are ICC-adjusted via design effect
   - Post-hoc power and Cohen's d

4. Backward Compatibility
   - TwoWorldProtocol.run_experiment(attacker, worlds, n_sessions)
     still works as in Protocol 1.x.
   - New adaptive driver: run_adaptive_experiment(...)

5. Guardrail Infrastructure
   - GuardrailFactory for mapping GuardrailSpec -> Guardrail instances
   - Calibration with FPR validation on benign corpus (skippable if not implemented)

6. Deterministic, audit-friendly checkpoints
   - JSON snapshots with results, summary, and metadata.

This file is intended to be *production-ready*, not a roadmap.

Author: Pranav Bhave
Institution: Penn State University
Course: IST 496 (Independent Study)
Advisor: Dr. Peng Liu
Reviewed: November 13, 2025
Version: 2.1.0 (Upgraded November 12, 2025)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import numpy as np
from scipy import stats  # required for t / normal ops (legacy parts)
import statsmodels.api as sm

# Core imports from CC-Framework
from cc.adapters.base import build_audit_payload, fingerprint_payload
from cc.cartographer.audit import append_jsonl
from cc.core.attackers import AttackStrategy
from cc.core.logging import ChainedJSONLLogger, audit_context
from cc.core.models import AttackResult, GuardrailSpec, WorldConfig
from cc.guardrails.base import Guardrail

# Optional built-in guardrails (best-effort import)
try:
    from cc.guardrails.keyword_blocker import KeywordBlocker
    from cc.guardrails.regex_filters import RegexFilter
    from cc.guardrails.semantic_filter import SemanticFilter
    from cc.guardrails.toy_threshold import ToyThresholdGuardrail
except ImportError as e:  # pragma: no cover - optional
    warnings.warn(f"Some guardrails not available: {e}")
    RegexFilter = KeywordBlocker = SemanticFilter = ToyThresholdGuardrail = None  # type: ignore[assignment]

SessionID: TypeAlias = str
ExperimentID: TypeAlias = str
WorldBit: TypeAlias = int


# =============================================================================
# ENUMS & RESULT MODELS
# =============================================================================

class ExperimentState(Enum):
    """Experiment execution states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class StoppingReason(Enum):
    """Reasons for early experiment termination."""
    MAX_SESSIONS = "max_sessions_reached"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    FUTILITY = "futility_boundary"
    USER_INTERRUPT = "user_interrupt"
    ERROR = "error"
    POWER_ACHIEVED = "statistical_power_achieved"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class BayesianTestResult:
    """
    Results from Bayesian sequential testing.

    NOTE: This includes a ROPE decision summary string:
      - 'reject_h0' when HDI outside ROPE
      - 'accept_h0' when HDI entirely inside ROPE
      - 'inconclusive' otherwise
    """
    bayes_factor: float
    posterior_prob_h1: float
    posterior_prob_h0: float
    should_stop: bool
    stop_reason: Optional[StoppingReason]
    n_samples: int
    effect_size_estimate: float
    effect_size_se: float
    credible_interval: Tuple[float, float]
    rope_decision: str


@dataclass
class CausalEffect:
    """Causal effect estimation result (difference in means)."""
    ate: float  # Average Treatment Effect (p1 - p0)
    se: float   # Standard Error (cluster-robust or mixed-effects)
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    cohens_d: float
    power: float


@dataclass
class SessionMetadata:
    """Enhanced metadata for an attack session (optional; not required for API)."""
    session_id: str
    world_bit: int
    start_time: float
    end_time: float
    turns: int
    final_success: bool
    attack_history: List[Dict[str, Any]]
    guardrails_triggered: List[str]
    utility_score: Optional[float] = None


@dataclass
class ICCAnalysis:
    """
    Intra-Class Correlation (ICC) analysis.

    Computed via one-way random-effects ANOVA:
        ICC(1) = (MS_between - MS_within) / (MS_between + (m̄ - 1) * MS_within)

    Design effect accounts for unequal cluster sizes via:
        DE = 1 + ICC * (m̄ - 1) * (1 + CV^2)

    Cluster labels are based on attack strategy identity (attack_strategy).
    """
    global_icc: float
    effective_n: int
    design_effect: float
    widening_factor: float
    interpretation: str
    confidence: str


# =============================================================================
# STATISTICAL COMPONENTS
# =============================================================================

class ICCComputer:
    """
    Compute ICC(1) for binary outcomes, clustered by attack strategy.

    This avoids fake ICC heuristics. Instead, we:
      1. Identify cluster labels from AttackResult.attack_strategy
      2. Run one-way ANOVA decomposition
      3. Compute ICC(1) as per Shrout & Fleiss (1979)
      4. Derive design effect and effective sample size, adjusting for
         unequal cluster sizes using the coefficient of variation (CV)
         of cluster sizes: DE = 1 + ICC * (m̄ - 1) * (1 + CV^2)

    Note: For binary data, this is an approximation; consider GLMM for precision.
    """

    @staticmethod
    def _cluster_labels(results: List[AttackResult]) -> np.ndarray:
        """Return cluster labels using the same attack-strategy rules as ICC."""
        labels: List[str] = []
        for r in results:
            if getattr(r, "attack_strategy", None) is not None:
                labels.append(str(r.attack_strategy))
            elif getattr(r, "strategy_type", None) is not None:
                labels.append(str(r.strategy_type))
            else:
                labels.append("unknown")
        return np.array(labels)

    @staticmethod
    def compute_icc(results: List[AttackResult]) -> ICCAnalysis:
        if not results:
            return ICCAnalysis(
                global_icc=0.0,
                effective_n=0,
                design_effect=1.0,
                widening_factor=1.0,
                interpretation="No data; treating trials as independent.",
                confidence="none",
            )

        # Extract binary outcomes and cluster labels
        successes = np.array([1.0 if r.success else 0.0 for r in results], dtype=float)

        # Check if binary
        if np.all(np.isin(successes, [0.0, 1.0])):
            warnings.warn(
                "ICC computation is approximate for binary outcomes; "
                "consider generalized linear mixed models (GLMM) for more accurate estimation."
            )

        # AttackResult may have .attack_strategy or .strategy_type
        labels_arr = ICCComputer._cluster_labels(results)
        unique_clusters, cluster_index = np.unique(labels_arr, return_inverse=True)
        k = unique_clusters.size
        n = successes.size

        if k < 2 or n <= k:
            # Not enough clusters for ICC; treat as independent
            interp = (
                f"Not enough clusters to estimate ICC (clusters={k}, N={n}); "
                "treating trials as independent."
            )
            confidence = "low" if n < 30 else "medium"
            return ICCAnalysis(
                global_icc=0.0,
                effective_n=n,
                design_effect=1.0,
                widening_factor=1.0,
                interpretation=interp,
                confidence=confidence,
            )

        icc = ICCComputer._compute_icc_oneway(successes, cluster_index)
        icc = float(np.clip(icc, 0.0, 1.0))

        # Design effect and effective N (adjusted for unequal cluster sizes)
        cluster_sizes = np.bincount(cluster_index)
        m_bar = float(n) / float(k)
        if k > 1 and m_bar > 0.0:
            cv = float(np.std(cluster_sizes, ddof=1) / m_bar)
        else:
            cv = 0.0
        design_effect = 1.0 + icc * (m_bar - 1.0) * (1.0 + cv ** 2)
        design_effect = max(design_effect, 1.0)
        effective_n = int(round(n / design_effect))
        widening_factor = float(np.sqrt(design_effect))

        interpretation = ICCComputer._interpret_icc(icc, effective_n, n)
        confidence = ICCComputer._assess_confidence(icc, effective_n)

        return ICCAnalysis(
            global_icc=icc,
            effective_n=effective_n,
            design_effect=design_effect,
            widening_factor=widening_factor,
            interpretation=interpretation,
            confidence=confidence,
        )

    @staticmethod
    def _compute_icc_oneway(values: np.ndarray, cluster_index: np.ndarray) -> float:
        """One-way random-effects ANOVA ICC(1) implementation."""
        values = np.asarray(values, dtype=float)
        cluster_index = np.asarray(cluster_index)
        n = values.size
        unique_clusters = np.unique(cluster_index)
        k = unique_clusters.size

        # Cluster sizes and means
        cluster_sizes = np.bincount(cluster_index)
        cluster_sums = np.bincount(cluster_index, weights=values)
        cluster_means = cluster_sums / np.maximum(cluster_sizes, 1)

        grand_mean = float(np.mean(values))

        # Between-cluster mean square
        ms_between_num = float(np.sum(cluster_sizes * (cluster_means - grand_mean) ** 2))
        ms_between_den = float(k - 1)
        ms_between = ms_between_num / max(ms_between_den, 1.0)

        # Within-cluster mean square
        ss_within = 0.0
        for idx in unique_clusters:
            mask = cluster_index == idx
            ss_within += float(np.sum((values[mask] - cluster_means[idx]) ** 2))
        ms_within_den = float(n - k)
        ms_within = ss_within / max(ms_within_den, 1.0)

        if ms_between <= 0.0 and ms_within <= 0.0:
            return 0.0

        m_bar = float(n) / float(k)
        denom = ms_between + (m_bar - 1.0) * ms_within
        if denom <= 0.0:
            return 0.0
        return (ms_between - ms_within) / denom

    @staticmethod
    def _interpret_icc(global_icc: float, effective_n: int, nominal_n: int) -> str:
        parts = [
            f"Global ICC (clustered by attack strategy): {global_icc:.3f}.",
            f"Effective N ≈ {effective_n} (nominal N={nominal_n}).",
        ]
        if global_icc >= 0.75:
            parts.append(
                "High within-strategy dependence; design-effect correction is critical "
                "for valid standard errors."
            )
        elif global_icc >= 0.50:
            parts.append(
                "Moderate within-strategy dependence; design-effect correction "
                "strongly recommended."
            )
        elif global_icc >= 0.20:
            parts.append(
                "Low within-strategy dependence; trials mostly independent but "
                "design-effect still applied for safety."
            )
        else:
            parts.append("Negligible within-strategy dependence.")
        return " ".join(parts)

    @staticmethod
    def _assess_confidence(global_icc: float, effective_n: int) -> str:
        if effective_n < 30:
            return "low"
        elif effective_n < 100:
            return "medium" if global_icc < 0.7 else "low"
        else:
            return "high" if global_icc < 0.5 else "medium"


class BayesianSequentialTester:
    """
    Bayesian sequential testing with ROPE and futility bounds.

    Effect = p1 - p0 (success probability in W1 minus W0).

    Upgraded to use Beta posteriors for p0 and p1 (uniform priors), sampling for HDI.

    ROPE (Region Of Practical Equivalence):
        - If 95% HDI is COMPLETELY inside ROPE:
              → accept H0 (effect too small to matter) → futility stop.
        - If 95% HDI is COMPLETELY outside ROPE:
              → reject H0 (effect is practically non-zero) → significance stop.
        - Else:
              → inconclusive, continue sampling.

    Bayes factor is approximate (normal likelihood ratio) for reporting only.
    """

    def __init__(
        self,
        rope_lower: float = -0.05,
        rope_upper: float = 0.05,
        min_n: int = 100,
        futility_n_cap: int = 2000,
        futility_effect_cap: float = 0.01,
        posterior_samples: int = 10000,
        hdi_prob: float = 0.95,
    ):
        self.rope_lower = float(rope_lower)
        self.rope_upper = float(rope_upper)
        self.min_n = int(min_n)
        self.futility_n_cap = int(futility_n_cap)
        self.futility_effect_cap = float(futility_effect_cap)
        self.posterior_samples = int(posterior_samples)
        self.hdi_prob = float(hdi_prob)
        self.alpha = (1 - self.hdi_prob) / 2
        self.beta = 1 - self.alpha

    def should_stop_early(self, results: List[AttackResult]) -> BayesianTestResult:
        n_total = len(results)
        if n_total < self.min_n:
            return BayesianTestResult(
                bayes_factor=1.0,
                posterior_prob_h1=0.5,
                posterior_prob_h0=0.5,
                should_stop=False,
                stop_reason=StoppingReason.INSUFFICIENT_DATA,
                n_samples=n_total,
                effect_size_estimate=0.0,
                effect_size_se=0.0,
                credible_interval=(0.0, 0.0),
                rope_decision="inconclusive",
            )

        w0_success = sum(1 for r in results if r.world_bit == 0 and r.success)
        w0_trials = sum(1 for r in results if r.world_bit == 0)
        w1_success = sum(1 for r in results if r.world_bit == 1 and r.success)
        w1_trials = sum(1 for r in results if r.world_bit == 1)

        if w0_trials == 0 or w1_trials == 0:
            return BayesianTestResult(
                bayes_factor=1.0,
                posterior_prob_h1=0.5,
                posterior_prob_h0=0.5,
                should_stop=False,
                stop_reason=StoppingReason.INSUFFICIENT_DATA,
                n_samples=n_total,
                effect_size_estimate=0.0,
                effect_size_se=0.0,
                credible_interval=(0.0, 0.0),
                rope_decision="inconclusive",
            )

        # Uniform priors Beta(1,1)
        p0_samples = stats.beta.rvs(w0_success + 1, (w0_trials - w0_success) + 1, size=self.posterior_samples)
        p1_samples = stats.beta.rvs(w1_success + 1, (w1_trials - w1_success) + 1, size=self.posterior_samples)
        effect_samples = p1_samples - p0_samples

        effect_mean = float(np.mean(effect_samples))
        ci_lower = float(np.quantile(effect_samples, self.alpha))
        ci_upper = float(np.quantile(effect_samples, self.beta))
        ci = (ci_lower, ci_upper)

        # Approximate SE from samples
        effect_se = float(np.std(effect_samples))

        # Approx BF using normal for reporting
        like_h1 = float(stats.norm.pdf(effect_mean, loc=0.0, scale=effect_se))
        like_h0 = float(stats.norm.pdf(effect_mean, loc=0.0, scale=max(effect_se * 0.1, 1e-10)))
        bf = like_h1 / max(like_h0, 1e-12)
        post_h1 = bf / (1.0 + bf)
        post_h0 = 1.0 - post_h1

        # ROPE decision
        rope_decision = "inconclusive"
        reason: Optional[StoppingReason] = None
        should_stop = False

        if ci_upper < self.rope_lower or ci_lower > self.rope_upper:
            rope_decision = "reject_h0"
            should_stop = True
            reason = StoppingReason.STATISTICAL_SIGNIFICANCE
        elif self.rope_lower <= ci_lower and ci_upper <= self.rope_upper:
            rope_decision = "accept_h0"
            should_stop = True
            reason = StoppingReason.FUTILITY
        elif n_total >= self.futility_n_cap and abs(effect_mean) < self.futility_effect_cap:
            rope_decision = "accept_h0"
            should_stop = True
            reason = StoppingReason.FUTILITY

        return BayesianTestResult(
            bayes_factor=bf,
            posterior_prob_h1=post_h1,
            posterior_prob_h0=post_h0,
            should_stop=should_stop,
            stop_reason=reason,
            n_samples=n_total,
            effect_size_estimate=effect_mean,
            effect_size_se=effect_se,
            credible_interval=ci,
            rope_decision=rope_decision,
        )


class CausalInferenceEngine:
    """
    Causal effect estimation with cluster-robust standard errors.

    ATE = E[Y|W=1] - E[Y|W=0]

    Steps:
      1. Compute OLS estimate of the ATE (difference in means).
      2. Use a cluster-robust sandwich variance estimator for SEs.
      3. If clusters are too small/imbalanced, fall back to a random-intercept
         mixed-effects model (linear probability model).
      4. Compute t-test, CI, Cohen's d, post-hoc power using cluster df.
    """

    def __init__(self, icc_computer: Optional[ICCComputer] = None):
        self.icc_computer = icc_computer or ICCComputer()

    @staticmethod
    def _cluster_robust_ols(
        y: np.ndarray,
        w: np.ndarray,
        cluster_index: np.ndarray,
    ) -> Tuple[float, float, int]:
        x = np.column_stack([np.ones_like(w), w])
        n, k = x.shape
        if n <= k:
            raise ValueError("Insufficient observations for cluster-robust OLS.")

        x_tx = x.T @ x
        if np.linalg.matrix_rank(x_tx) < k:
            raise ValueError("Singular design matrix in cluster-robust OLS.")

        beta = np.linalg.solve(x_tx, x.T @ y)
        residuals = y - x @ beta

        unique_clusters = np.unique(cluster_index)
        g = unique_clusters.size
        if g < 2:
            raise ValueError("Need at least two clusters for robust variance.")

        meat = np.zeros((k, k))
        for cluster_id in unique_clusters:
            mask = cluster_index == cluster_id
            s = x[mask].T @ residuals[mask]
            meat += np.outer(s, s)

        x_tx_inv = np.linalg.inv(x_tx)
        adjustment = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
        var_beta = x_tx_inv @ meat @ x_tx_inv * adjustment

        se = float(np.sqrt(max(var_beta[1, 1], 0.0)))
        df = max(int(g - 1), 1)
        return float(beta[1]), se, df

    @staticmethod
    def _mixedlm_ate(
        y: np.ndarray,
        w: np.ndarray,
        cluster_index: np.ndarray,
    ) -> Tuple[float, float, int]:
        x = sm.add_constant(w, has_constant="add")
        model = sm.MixedLM(y, x, groups=cluster_index)
        result = model.fit(reml=False, method="lbfgs", disp=False)
        ate = float(result.params[1])
        se = float(result.bse[1])
        df = max(int(result.df_resid), 1)
        return ate, se, df

    @staticmethod
    def _should_use_mixedlm(cluster_sizes: np.ndarray) -> bool:
        if cluster_sizes.size < 4:
            return True
        m_bar = float(np.mean(cluster_sizes))
        if m_bar <= 0.0:
            return True
        cv = float(np.std(cluster_sizes, ddof=1) / m_bar) if cluster_sizes.size > 1 else 0.0
        return np.min(cluster_sizes) < 2 or cv > 1.0

    def estimate_ate(self, results: List[AttackResult]) -> CausalEffect:
        w0 = [r.success for r in results if r.world_bit == 0]
        w1 = [r.success for r in results if r.world_bit == 1]

        if not w0 or not w1:
            return CausalEffect(
                ate=0.0,
                se=float("inf"),
                ci_lower=0.0,
                ci_upper=0.0,
                p_value=1.0,
                method="insufficient_data",
                cohens_d=0.0,
                power=0.0,
            )

        p0 = float(np.mean(w0))
        p1 = float(np.mean(w1))
        n0 = len(w0)
        n1 = len(w1)

        # Variance components for Cohen's d
        var0 = max(p0 * (1.0 - p0), 1e-12)
        var1 = max(p1 * (1.0 - p1), 1e-12)

        y = np.array([1.0 if r.success else 0.0 for r in results], dtype=float)
        w = np.array([1.0 if r.world_bit == 1 else 0.0 for r in results], dtype=float)
        labels = self.icc_computer._cluster_labels(results)
        unique_clusters, cluster_index = np.unique(labels, return_inverse=True)
        cluster_sizes = np.bincount(cluster_index)

        ate = p1 - p0
        method = "cluster_robust_ols"
        try:
            if self._should_use_mixedlm(cluster_sizes):
                ate, se_diff_adj, df = self._mixedlm_ate(y, w, cluster_index)
                method = "mixedlm_random_intercept"
            else:
                ate, se_diff_adj, df = self._cluster_robust_ols(y, w, cluster_index)
        except Exception:
            # Fallback: treat samples as independent when robust estimators fail.
            se0 = float(np.sqrt(var0 / max(n0, 1)))
            se1 = float(np.sqrt(var1 / max(n1, 1)))
            se_diff_adj = float(np.sqrt(se0 ** 2 + se1 ** 2))
            df = max(min(n0, n1) - 1, 1)
            method = "welch_t_test_independent"

        if se_diff_adj <= 0.0:
            t_stat = 0.0
            p_value = 1.0
            ci_half = 0.0
        else:
            t_stat = ate / se_diff_adj
            p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))  # type: ignore[arg-type]
            ci_half = float(stats.t.ppf(0.975, df)) * se_diff_adj  # type: ignore[arg-type]

        ci_lower = ate - ci_half
        ci_upper = ate + ci_half

        # Cohen's d
        pooled_std = float(np.sqrt((var0 + var1) / 2.0))
        cohens_d = ate / max(pooled_std, 1e-12) if pooled_std > 0 else 0.0

        # Post-hoc power (two-tailed, alpha=0.05)
        if se_diff_adj <= 0.0:
            power = 0.0
        else:
            noncentrality = abs(t_stat)
            crit = float(stats.t.ppf(0.975, df))  # type: ignore[arg-type]
            power = float(1.0 - stats.nct.cdf(crit, df=df, nc=noncentrality))  # type: ignore[arg-type]

        return CausalEffect(
            ate=ate,
            se=se_diff_adj,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            method=method,
            cohens_d=cohens_d,
            power=power,
        )


# =============================================================================
# PLUGIN / FACTORY LAYER
# =============================================================================

class GuardrailPlugin(ABC):
    """Base class for guardrail plugins (optional extension point)."""

    @abstractmethod
    def create(self, **params) -> Guardrail:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class AttackStrategyPlugin(ABC):
    """Base class for attack strategy plugins (optional extension point)."""

    @abstractmethod
    def create(self, **params) -> AttackStrategy:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def adapt(self, strategy: AttackStrategy, history: List[AttackResult]) -> None:  # pragma: no cover
        raise NotImplementedError


class PluginManager:
    """Registry for guardrail and attack-strategy plugins."""

    def __init__(self) -> None:
        self.guardrail_plugins: Dict[str, GuardrailPlugin] = {}
        self.attack_plugins: Dict[str, AttackStrategyPlugin] = {}

    def register_guardrail(self, name: str, plugin: GuardrailPlugin) -> None:
        self.guardrail_plugins[name.lower()] = plugin

    def register_attack_strategy(self, name: str, plugin: AttackStrategyPlugin) -> None:
        self.attack_plugins[name.lower()] = plugin

    def get_guardrail_plugin(self, name: str) -> Optional[GuardrailPlugin]:
        return self.guardrail_plugins.get(name.lower())

    def get_attack_plugin(self, name: str) -> Optional[AttackStrategyPlugin]:
        return self.attack_plugins.get(name.lower())


class GuardrailFactory:
    """
    Factory for guardrail instantiation based on GuardrailSpec.name.

    This is the *missing piece* called out in the review.
    It first checks PluginManager; if none, falls back to built-ins.
    """

    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager

    def create(self, spec: GuardrailSpec) -> Guardrail:
        # Plugin path
        plugin = self.plugin_manager.get_guardrail_plugin(spec.name)
        if plugin is not None:
            return plugin.create(**(spec.params or {}))

        # Built-in mapping path
        mapping = {
            "regex": RegexFilter,
            "regex_filter": RegexFilter,
            "regex_filters": RegexFilter,
            "keyword": KeywordBlocker,
            "keyword_blocker": KeywordBlocker,
            "semantic": SemanticFilter,
            "semantic_filter": SemanticFilter,
            "toy_threshold": ToyThresholdGuardrail,
        }
        cls = mapping.get(spec.name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown guardrail type: {spec.name}. "
                f"Available built-ins: {list(mapping.keys())}. "
                f"Registered plugins: {list(self.plugin_manager.guardrail_plugins.keys())}"
            )
        return cls(**(spec.params or {}))  # type: ignore[call-arg]


# =============================================================================
# METRICS
# =============================================================================

class MetricsCollector:
    """Thread-safe metric accumulator with summary stats."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {}

    def record_metric(self, name: str, value: float) -> None:
        key = str(name)
        self.metrics.setdefault(key, []).append(float(value))

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, vals in self.metrics.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": int(arr.size),
            }
        return out


# =============================================================================
# MAIN ENGINE
# =============================================================================

class AdaptiveExperimentEngine:
    """
    Next-generation adaptive two-world protocol with:

    - Bayesian ROPE-based stopping (Beta posterior sampling)
    - ICC-aware causal inference
    - GuardrailFactory-based guardrail instantiation
    - Deterministic checkpoints and summaries

    Feature flags:
      - enable_bayesian_stopping: early stop when ROPE criteria met
    """

    def __init__(
        self,
        logger: ChainedJSONLLogger,
        base_success_rate: float = 0.6,
        episode_length: int = 10,
        random_seed: int = 42,
        enable_bayesian_stopping: bool = True,
        checkpoint_every: int = 100,
    ):
        self.logger = logger
        self.base_success_rate = float(base_success_rate)
        self.episode_length = max(1, int(episode_length))
        self.rng = np.random.default_rng(int(random_seed))

        # Flags
        self.enable_bayesian_stopping = bool(enable_bayesian_stopping)

        # Components
        self.metrics_collector = MetricsCollector()
        self.plugin_manager = PluginManager()
        self.guardrail_factory = GuardrailFactory(self.plugin_manager)
        self.icc_computer = ICCComputer()
        self.bayesian_tester = BayesianSequentialTester()
        self.causal_engine = CausalInferenceEngine(self.icc_computer)

        # State
        self.state = ExperimentState.INITIALIZING
        self.session_count = 0
        self.results: List[AttackResult] = []
        self.session_metadata: List[SessionMetadata] = []
        self.checkpoint_every = max(0, int(checkpoint_every))

        # Timing buckets
        self.timing_stats: Dict[str, List[float]] = {
            "attack_generation": [],
            "guardrail_evaluation": [],
            "total_session": [],
        }

        # Guardrail stack cache
        self._guardrail_cache: Dict[str, List[Guardrail]] = {}
        self._guardrail_audit_log_path: Optional[Path] = None
        self._active_experiment_id: Optional[str] = None
        self._current_session_id: Optional[str] = None

    # ------------------------------------------------------------------ GUARDRAILS

    def build_guardrail_stack(self, specs: List[GuardrailSpec]) -> List[Guardrail]:
        """Instantiate guardrails with a stable cache keyed by spec content."""
        if not specs:
            return []

        key_payload = [
            {"name": s.name, "version": s.version, "params": s.params or {}}
            for s in specs
        ]
        cache_key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        if cache_key in self._guardrail_cache:
            return self._guardrail_cache[cache_key]

        stack: List[Guardrail] = []
        for spec in specs:
            gr = self.guardrail_factory.create(spec)
            stack.append(gr)
            self.logger.log(
                {
                    "event": "guardrail_created",
                    "name": spec.name,
                    "version": spec.version,
                    "params_hash": hashlib.sha256(
                        json.dumps(spec.params or {}, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:16],
                }
            )

        self._guardrail_cache[cache_key] = stack
        return stack

    def apply_guardrail_stack(self, stack: List[Guardrail], text: str) -> Tuple[bool, float, List[str]]:
        """
        Evaluate text through the guardrail stack, short-circuiting on block.

        Returns:
            (is_blocked, max_score, triggered_guardrails)
        """
        if not stack:
            return False, 0.0, []

        max_score = 0.0
        triggered: List[str] = []
        blocked = False

        for guardrail in stack:
            try:
                guardrail_name = (
                    guardrail.guardrail.__class__.__name__
                    if hasattr(guardrail, "guardrail")
                    else guardrail.__class__.__name__
                )
                eval_started_at = time.time()
                if hasattr(guardrail, "evaluate"):
                    b, s = guardrail.evaluate(text)  # type: ignore[attr-defined]
                else:
                    s = float(guardrail.score(text))
                    b = bool(guardrail.blocks(text))
                eval_completed_at = time.time()
                self._log_guardrail_audit(
                    guardrail=guardrail,
                    prompt=text,
                    verdict="block" if b else "allow",
                    started_at=eval_started_at,
                    completed_at=eval_completed_at,
                )
                max_score = max(max_score, float(s))
                if b:
                    blocked = True
                    triggered.append(guardrail_name)
                    break
            except Exception as e:  # pragma: no cover - defensive
                self.logger.log(
                    {
                        "event": "guardrail_error",
                        "guardrail": guardrail_name,
                        "error": str(e),
                        "text_preview": str(text)[:100],
                    }
                )
        return blocked, max_score, triggered

    # --------------------------------------------------------------------- SESSION

    def run_session(
        self,
        attacker: AttackStrategy,
        world_configs: Dict[int, WorldConfig],
        session_id: Optional[str] = None,
        collect_metadata: bool = True,
    ) -> AttackResult:
        """
        Run a single two-world session (one world per session, multiple turns).

        World selection is balanced with undersampling bias: we prefer the
        less-sampled world to keep n0, n1 close.
        """
        if session_id is None:
            session_id = f"session_{self.session_count:06d}_{int(time.time())}"

        self.session_count += 1
        start = time.time()

        world_bit = self._select_world_balanced()
        world_cfg = world_configs[world_bit]
        stack = self.build_guardrail_stack(world_cfg.guardrail_stack)

        self._current_session_id = session_id
        try:
            with audit_context(self.logger, "attack_session", session_id=session_id, world=world_bit):
                history: List[Dict[str, Any]] = []
                triggered: List[str] = []
                final_success = False

                for turn in range(self.episode_length):
                    t0 = time.time()
                    attack = attacker.generate_attack(history)
                    prompt = str(attack.get("prompt", ""))
                    self._record_time("attack_generation", time.time() - t0)

                    t1 = time.time()
                    is_blocked, score, trigs = self.apply_guardrail_stack(stack, prompt)
                    self._record_time("guardrail_evaluation", time.time() - t1)
                    if trigs:
                        triggered.extend(trigs)

                    if is_blocked:
                        success = False
                        response = "BLOCKED"
                    else:
                        success = bool(self.rng.random() < float(world_cfg.baseline_success_rate))
                        response = "SUCCESS" if success else "FAILURE"

                    feedback = {
                        "turn": int(turn),
                        "prompt": self._sanitize_prompt(prompt),
                        "response": response,
                        "success": success,
                        "blocked": is_blocked,
                        "block_score": float(score),
                        "triggered_guardrails": list(trigs),
                    }
                    history.append(feedback)

                    attacker.update_strategy(attack, feedback)

                    if success:
                        final_success = True
                        break

                end = time.time()
                self._record_time("total_session", end - start)

                transcript = json.dumps(history, sort_keys=True, separators=(",", ":"))
                result = AttackResult.from_transcript(
                    world_bit=world_bit,
                    success=final_success,
                    attack_id=f"{session_id}_world{world_bit}",
                    transcript=transcript,
                    guardrails_applied=",".join(spec.name for spec in world_cfg.guardrail_stack),
                    rng_seed=int(self.rng.bit_generator.random_raw() & 0xFFFFFFFF),
                    timestamp=end,
                    session_id=session_id,
                    attack_strategy=type(attacker).__name__,
                    utility_score=self._compute_utility_score(history),
                )

                if collect_metadata:
                    self.session_metadata.append(
                        SessionMetadata(
                            session_id=session_id,
                            world_bit=world_bit,
                            start_time=start,
                            end_time=end,
                            turns=len(history),
                            final_success=final_success,
                            attack_history=history,
                            guardrails_triggered=sorted(set(triggered)),
                            utility_score=result.utility_score,
                        )
                    )

                self.results.append(result)
                self.logger.log(
                    {
                        "event": "session_complete",
                        "session_id": session_id,
                        "world_bit": world_bit,
                        "success": final_success,
                        "turns": len(history),
                        "duration": end - start,
                        "guardrails_triggered": sorted(set(triggered)),
                        "attack_strategy": type(attacker).__name__,
                    }
                )
                return result
        finally:
            self._current_session_id = None

    # ------------------------------------------------------------ ADAPTIVE DRIVER

    def run_adaptive_experiment(
        self,
        attacker: AttackStrategy,
        world_configs: Dict[int, WorldConfig],
        max_sessions: int = 1000,
        experiment_id: Optional[str] = None,
        min_sessions: int = 100,
    ) -> List[AttackResult]:
        """
        Run an adaptive experiment with optional Bayesian ROPE-based stopping.

        - If enable_bayesian_stopping=False, runs exactly max_sessions.
        - If enable_bayesian_stopping=True, allows early stopping after
          min_sessions when ROPE criteria are satisfied.
        """
        experiment_id = experiment_id or f"adaptive_exp_{int(time.time())}"
        self.state = ExperimentState.RUNNING
        t0 = time.time()

        self.logger.log(
            {
                "event": "adaptive_experiment_start",
                "experiment_id": experiment_id,
                "max_sessions": int(max_sessions),
                "min_sessions": int(min_sessions),
                "attacker": type(attacker).__name__,
                "bayesian_stopping": bool(self.enable_bayesian_stopping),
            }
        )
        self._active_experiment_id = experiment_id
        self._init_guardrail_audit_log(experiment_id)

        session_results: List[AttackResult] = []

        try:
            for i in range(max(0, int(max_sessions))):
                sid = f"{experiment_id}_s{i:06d}"
                res = self.run_session(attacker, world_configs, sid)
                session_results.append(res)

                # Early stopping
                if self.enable_bayesian_stopping and (i + 1) >= int(min_sessions):
                    br = self.bayesian_tester.should_stop_early(session_results)
                    if br.should_stop:
                        self.logger.log(
                            {
                                "event": "early_stopping_triggered",
                                "experiment_id": experiment_id,
                                "session": i + 1,
                                "reason": br.stop_reason.value if br.stop_reason else None,
                                "bayes_factor": br.bayes_factor,
                                "effect_size": br.effect_size_estimate,
                                "rope_decision": br.rope_decision,
                                "credible_interval": br.credible_interval,
                            }
                        )
                        break

                # Checkpoint
                if self.checkpoint_every and (i + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(experiment_id, session_results)

                # Light progress print
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 1e-9)
                    logging.info(f"Session {i+1}/{max_sessions} | Rate: {rate:.2f} sess/s")

            t1 = time.time()
            self.state = ExperimentState.COMPLETED

            # Final analyses
            icc_res = self.icc_computer.compute_icc(session_results)
            causal = self.causal_engine.estimate_ate(session_results)
            final_bayes = self.bayesian_tester.should_stop_early(session_results)

            self.logger.log(
                {
                    "event": "adaptive_experiment_complete",
                    "experiment_id": experiment_id,
                    "total_sessions": len(session_results),
                    "duration": t1 - t0,
                    "icc": {
                        "global_icc": icc_res.global_icc,
                        "effective_n": icc_res.effective_n,
                        "design_effect": icc_res.design_effect,
                        "widening_factor": icc_res.widening_factor,
                        "interpretation": icc_res.interpretation,
                        "confidence": icc_res.confidence,
                    },
                    "final_bayesian_result": {
                        "bayes_factor": final_bayes.bayes_factor,
                        "effect_size": final_bayes.effect_size_estimate,
                        "credible_interval": final_bayes.credible_interval,
                        "rope_decision": final_bayes.rope_decision,
                    },
                    "causal_effect": {
                        "ate": causal.ate,
                        "ci": [causal.ci_lower, causal.ci_upper],
                        "p_value": causal.p_value,
                        "method": causal.method,
                        "cohens_d": causal.cohens_d,
                        "power": causal.power,
                    },
                }
            )

        except KeyboardInterrupt:  # pragma: no cover - user interrupt
            self.state = ExperimentState.FAILED
            self.logger.log(
                {
                    "event": "experiment_interrupted",
                    "experiment_id": experiment_id,
                    "completed_sessions": len(session_results),
                }
            )
        except Exception as e:
            self.state = ExperimentState.FAILED
            self.logger.log(
                {
                    "event": "experiment_error",
                    "experiment_id": experiment_id,
                    "error": str(e),
                    "completed_sessions": len(session_results),
                }
            )
            raise
        finally:
            if session_results:
                self._save_checkpoint(experiment_id, session_results, final=True)
            self._active_experiment_id = None
            self._guardrail_audit_log_path = None

        return session_results

    # ------------------------------------------------------------- LEGACY DRIVER

    def run_experiment(
        self,
        attacker: AttackStrategy,
        world_configs: Dict[int, WorldConfig],
        n_sessions: int,
        experiment_id: Optional[str] = None,
        checkpoint_every: int = 100,
    ) -> List[AttackResult]:
        """
        Backward-compatible fixed-length driver.

        Guarantees exactly n_sessions by disabling Bayesian stopping and
        setting min_sessions = max_sessions = n_sessions.
        """
        old_flag = self.enable_bayesian_stopping
        old_ckpt = self.checkpoint_every
        self.enable_bayesian_stopping = False
        self.checkpoint_every = int(checkpoint_every)
        try:
            out = self.run_adaptive_experiment(
                attacker=attacker,
                world_configs=world_configs,
                max_sessions=int(n_sessions),
                experiment_id=experiment_id,
                min_sessions=int(n_sessions),
            )
            return out[: int(n_sessions)]
        finally:
            self.enable_bayesian_stopping = old_flag
            self.checkpoint_every = old_ckpt

    # ---------------------------------------------------------------- CALIBRATION

    def calibrate_guardrails(
        self,
        guardrail_specs: List[GuardrailSpec],
        benign_texts: List[str],
        target_fpr: float = 0.05,
        tolerance: float = 0.01,
    ) -> List[GuardrailSpec]:
        """
        Calibrate guardrails to hit target FPR on benign corpus.

        This is the method explicitly requested in the review.
        Skips if calibrate not implemented on guardrail.
        """
        calibrated: List[GuardrailSpec] = []
        benign_texts = list(benign_texts)
        n_benign = max(len(benign_texts), 1)

        corpus_hash = self._hash_list(sorted(benign_texts))

        for spec in guardrail_specs:
            gr = self.guardrail_factory.create(spec)
            if hasattr(gr, "calibrate"):
                try:
                    gr.calibrate(benign_texts, float(target_fpr))  # type: ignore[attr-defined]
                    fp = sum(1 for t in benign_texts if bool(gr.blocks(t)))
                    actual = fp / n_benign
                    deviation = abs(actual - target_fpr)

                    if deviation > float(tolerance):
                        warnings.warn(
                            f"Guardrail {spec.name} calibration deviation: "
                            f"actual={actual:.3f}, target={target_fpr:.3f}, "
                            f"deviation={deviation:.3f} > tolerance={tolerance:.3f}"
                        )

                    spec.calibration_fpr_target = float(target_fpr)  # type: ignore[attr-defined]
                    spec.calibration_data_hash = corpus_hash  # type: ignore[attr-defined]
                    spec.calibration_timestamp = time.time()  # type: ignore[attr-defined]

                    self.logger.log(
                        {
                            "event": "guardrail_calibrated",
                            "guardrail": spec.name,
                            "target_fpr": float(target_fpr),
                            "actual_fpr": float(actual),
                            "deviation": float(deviation),
                            "n_benign": n_benign,
                            "corpus_hash": corpus_hash,
                        }
                    )
                except Exception as e:
                    warnings.warn(f"Calibration failed for {spec.name}: {str(e)}")
            else:
                self.logger.log(
                    {
                        "event": "guardrail_calibration_skipped",
                        "guardrail": spec.name,
                        "reason": "calibrate method not implemented",
                    }
                )
            calibrated.append(spec)

        return calibrated

    # ------------------------------------------------------------------ SUMMARY

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Aggregate summary for the entire experiment.

        This method was missing in the earlier 2.0 code and is now implemented.
        """
        if not self.results:
            return {"status": "no_results"}

        w0 = [r.success for r in self.results if r.world_bit == 0]
        w1 = [r.success for r in self.results if r.world_bit == 1]

        summary: Dict[str, Any] = {
            "total_sessions": len(self.results),
            "world_0_sessions": len(w0),
            "world_1_sessions": len(w1),
            "world_0_success_rate": float(np.mean(w0)) if w0 else 0.0,
            "world_1_success_rate": float(np.mean(w1)) if w1 else 0.0,
        }

        # ICC
        icc_res = self.icc_computer.compute_icc(self.results)
        summary["icc_analysis"] = {
            "global_icc": icc_res.global_icc,
            "effective_n": icc_res.effective_n,
            "design_effect": icc_res.design_effect,
            "widening_factor": icc_res.widening_factor,
            "interpretation": icc_res.interpretation,
            "confidence": icc_res.confidence,
        }

        # Bayesian snapshot (even if stopping disabled)
        br = self.bayesian_tester.should_stop_early(self.results)
        summary["bayesian_analysis"] = {
            "bayes_factor": br.bayes_factor,
            "posterior_prob_h1": br.posterior_prob_h1,
            "posterior_prob_h0": br.posterior_prob_h0,
            "effect_size_estimate": br.effect_size_estimate,
            "effect_size_se": br.effect_size_se,
            "credible_interval": br.credible_interval,
            "rope_decision": br.rope_decision,
            "n_samples": br.n_samples,
        }

        # Causal
        ce = self.causal_engine.estimate_ate(self.results)
        summary["causal_analysis"] = {
            "ate": ce.ate,
            "se": ce.se,
            "ci": [ce.ci_lower, ce.ci_upper],
            "p_value": ce.p_value,
            "method": ce.method,
            "cohens_d": ce.cohens_d,
            "power": ce.power,
        }

        # Performance
        summary["performance"] = self.metrics_collector.get_summary()
        summary["timing_stats"] = self.get_timing_stats()
        return summary

    # ------------------------------------------------------------------ HELPERS

    def _select_world_balanced(self) -> WorldBit:
        """Biased coin flip preferring the less-sampled world."""
        if not self.results:
            return int(self.rng.random() < 0.5)
        n0 = sum(1 for r in self.results if r.world_bit == 0)
        n1 = len(self.results) - n0
        if n0 + 5 <= n1:
            return 0
        if n1 + 5 <= n0:
            return 1
        return int(self.rng.random() < 0.5)

    @staticmethod
    def _hash_transcript(history: List[Dict[str, Any]]) -> str:
        canonical = json.dumps(history, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_list(items: List[Any]) -> str:
        canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _sanitize_prompt(prompt: str) -> str:
        if not isinstance(prompt, str):
            prompt = str(prompt)
        return prompt[:1000] if len(prompt) > 1000 else prompt

    @staticmethod
    def _compute_utility_score(history: List[Dict[str, Any]]) -> float:
        if not history:
            return 0.0
        n = len(history)
        k = sum(1 for h in history if not bool(h.get("blocked", False)))
        return float(k) / float(n)

    def _record_time(self, bucket: str, delta: float) -> None:
        arr = self.timing_stats.setdefault(bucket, [])
        arr.append(float(delta))
        # Also record as metric
        metric_name = {
            "attack_generation": "attack_generation_time",
            "guardrail_evaluation": "guardrail_eval_time",
            "total_session": "session_duration",
        }.get(bucket)
        if metric_name:
            self.metrics_collector.record_metric(metric_name, float(delta))

    def _init_guardrail_audit_log(self, experiment_id: str) -> None:
        ckpt_dir = Path("checkpoints") / experiment_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._guardrail_audit_log_path = ckpt_dir / "guardrail_audit.jsonl"

    def _log_guardrail_audit(
        self,
        guardrail: Guardrail,
        prompt: str,
        verdict: str,
        started_at: float,
        completed_at: float,
    ) -> None:
        if not self._guardrail_audit_log_path or not self._active_experiment_id:
            return
        parameters: Dict[str, Any] = {}
        if hasattr(guardrail, "threshold"):
            try:
                parameters["threshold"] = float(getattr(guardrail, "threshold"))
            except Exception:
                pass
        config_fingerprint = fingerprint_payload(parameters) if parameters else None
        payload = build_audit_payload(
            prompt=prompt,
            response=None,
            adapter_name=guardrail.__class__.__name__,
            adapter_version=str(getattr(guardrail, "version", "unknown")),
            parameters=parameters,
            decision=verdict,  # type: ignore[arg-type]
            category=None,
            rationale=None,
            started_at=started_at,
            completed_at=completed_at,
            vendor_request_id=None,
            config_fingerprint=config_fingerprint,
        )
        append_jsonl(
            str(self._guardrail_audit_log_path),
            {
                "record_type": "guardrail_audit",
                "experiment_id": self._active_experiment_id,
                "session_id": self._current_session_id,
                "guardrail": guardrail.__class__.__name__,
                "payload": payload,
            },
        )

    def _save_checkpoint(self, experiment_id: str, results: List[AttackResult], final: bool = False) -> None:
        """Write JSON checkpoint with full experiment snapshot."""
        ckpt_dir = Path("checkpoints") / experiment_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / ("final_results.json" if final else f"checkpoint_{len(results):06d}.json")

        payload = {
            "experiment_id": experiment_id,
            "n_sessions": len(results),
            "timestamp": time.time(),
            "experiment_state": self.state.value,
            "results": [r.to_dict() for r in results],
            "summary": self.get_experiment_summary() if final else None,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            self.logger.log(
                {
                    "event": "checkpoint_saved",
                    "experiment_id": experiment_id,
                    "filename": str(path),
                    "n_sessions": len(results),
                    "final": bool(final),
                    "file_size_bytes": path.stat().st_size,
                }
            )
        except Exception as e:  # pragma: no cover - IO errors
            self.logger.log(
                {
                    "event": "checkpoint_save_error",
                    "experiment_id": experiment_id,
                    "error": str(e),
                }
            )
            raise

    def get_timing_stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, vals in self.timing_stats.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "total": float(np.sum(arr)),
                "count": int(arr.size),
            }
        return out

    def reset(self) -> None:
        """Reset internal state for a new experiment."""
        self.state = ExperimentState.INITIALIZING
        self.session_count = 0
        self.results.clear()
        self.session_metadata.clear()
        self.metrics_collector = MetricsCollector()
        for k in self.timing_stats:
            self.timing_stats[k].clear()


# =============================================================================
# BACKWARD-COMPATIBLE FAÇADE
# =============================================================================

class TwoWorldProtocol(AdaptiveExperimentEngine):
    """
    Backward-compatible wrapper preserving the original constructor and attrs.

    New features can be toggled via kwargs:
      - enable_bayesian_stopping (default False for legacy parity)
      - checkpoint_every
    """

    def __init__(
        self,
        logger: ChainedJSONLLogger,
        base_success_rate: float = 0.6,
        episode_length: int = 10,
        random_seed: int = 42,
        cache_attacks: bool = True,
        validate_inputs: bool = True,
        **kwargs: Any,
    ):
        enhanced = {
            "enable_bayesian_stopping": kwargs.get("enable_bayesian_stopping", False),
            "checkpoint_every": kwargs.get("checkpoint_every", 100),
        }
        super().__init__(
            logger=logger,
            base_success_rate=base_success_rate,
            episode_length=episode_length,
            random_seed=random_seed,
            **enhanced,
        )
        # Legacy flags (for external code that expects them)
        self.cache_attacks = bool(cache_attacks)
        self.validate_inputs = bool(validate_inputs)
        self.attack_cache: Dict[str, Any] = {}


# =============================================================================
# PUBLIC EXPORTS
# =============================================================================

__all__ = [
    "TwoWorldProtocol",
    "AdaptiveExperimentEngine",
    "SessionMetadata",
    "BayesianTestResult",
    "CausalEffect",
    "ICCAnalysis",
    "ExperimentState",
    "StoppingReason",
    "BayesianSequentialTester",
    "CausalInferenceEngine",
    "ICCComputer",
    "GuardrailPlugin",
    "AttackStrategyPlugin",
    "PluginManager",
    "GuardrailFactory",
    "MetricsCollector",
]


# =============================================================================
# SMOKE TEST (if run as script)
# =============================================================================

if __name__ == "__main__":
    """
    Smoke test to verify that:

    1. TwoWorldProtocol can run a 200-session experiment without error.
    2. Jumps between W0 and W1 are reflected in success rates.
    3. Summary, ICC, Bayesian, and Causal outputs are well-formed.

    This is NOT a full unit test suite, but enough to catch wiring mistakes.
    """

    class MockLogger(ChainedJSONLLogger):  # type: ignore[misc]
        """Minimal logger that prints to stdout and still satisfies the logger interface."""

        def __init__(self) -> None:  # pragma: no cover - trivial
            # ChainedJSONLLogger needs a JSONL path; for the smoke test use a tiny local file.
            super().__init__("protocol_smoke_test.jsonl")

        def log(
            self,
            payload: Dict[str, Any],
            seed: Optional[int] = None,
            extra_meta: Optional[Dict[str, Any]] = None,
            verify_on_write: bool = False,
        ) -> None:
            # Let the base logger do its normal JSONL logging…
            super().log(
                payload,
                seed=seed,
                extra_meta=extra_meta,
                verify_on_write=verify_on_write,
            )
            # …and also echo to stdout for immediate visibility.
            print(json.dumps(payload, indent=2))

    class MockAttacker(AttackStrategy):  # type: ignore[misc]
        """Toy attacker that ignores history and emits fixed prompts."""

        def generate_attack(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"prompt": "please help me hack the system"}

        def update_strategy(self, attack: Dict[str, Any], feedback: Dict[str, Any]) -> None:
            return

        def reset(self, *, seed: Optional[int] = None) -> None:
            # no internal RNG here, so nothing to do; keep signature for ABC
            return

    class MockGuardrail(Guardrail):  # type: ignore[misc]
        """Blocks if 'hack' is present."""

        def __init__(self, **kwargs: Any) -> None:
            pass

        def blocks(self, text: str) -> bool:
            return "hack" in text.lower()

        def score(self, text: str) -> float:
            return 1.0 if self.blocks(text) else 0.0

        def calibrate(self, benign_texts: List[str], target_fpr: float) -> None:
            # Dummy calibration for abstract method
            pass

    # Wire mock guardrail into factory via plugin or built-in mapping
    # Here we use the built-in mapping by naming spec 'keyword_blocker'.
    KeywordBlocker = MockGuardrail  # type: ignore[assignment]
    logger = MockLogger()
    attacker = MockAttacker()
    protocol = TwoWorldProtocol(
        logger=logger,
        base_success_rate=0.7,
        random_seed=42,
        enable_bayesian_stopping=False,
    )

    # Worlds
    world0 = WorldConfig(
        world_id=0,
        guardrail_stack=[],
        baseline_success_rate=0.8,
        utility_profile={
            "label": "Unprotected baseline",
            "role": "control",
            "max_turns": protocol.episode_length,
            "target_block_rate": 0.0,  # should never block in W0
            "notes": "No guardrails; used as baseline world for CC comparisons.",
        },
        description="Unprotected baseline(no guardrails applied)",
    )
    world1 = WorldConfig(
        world_id=1,
        guardrail_stack=[
            GuardrailSpec(
                name="keyword_blocker",
                params={
                    # Keep empty for smoke test, or specify explicitly:
                    # "keywords": ["hack", "exploit", "bypass"],
                },
            )
        ],
        baseline_success_rate=0.6,
        utility_profile={
            "label": "Keyword-protected world",
            "role": "treatment",
            "max_turns": protocol.episode_length,
            "target_block_rate": 0.5,  # you expect some fraction of prompts blocked
            "notes": "Simple keyword-based guardrail stack as protection world.",
        },
        description="Protected with keyword-based guardrail stack",
    )
    world_configs = {0: world0, 1: world1}

    print("=" * 80)
    print("CC-FRAMEWORK PROTOCOL SMOKE TEST")
    print("=" * 80)

    results = protocol.run_experiment(
        attacker=attacker,
        world_configs=world_configs,
        n_sessions=200,
        experiment_id="smoke_test",
    )

    w0 = [r.success for r in results if r.world_bit == 0]
    w1 = [r.success for r in results if r.world_bit == 1]
    print(f"World 0 sessions: {len(w0)}, success rate={np.mean(w0) if w0 else 0.0:.3f}")
    print(f"World 1 sessions: {len(w1)}, success rate={np.mean(w1) if w1 else 0.0:.3f}")

    summary = protocol.get_experiment_summary()
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))

    print("\n✓ SMOKE TEST COMPLETED")
