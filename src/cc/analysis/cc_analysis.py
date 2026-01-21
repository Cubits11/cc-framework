"""
CC-Framework analysis utilities.

This module aligns with the existing CC-Framework implementation:
- CC_max = J_comp / max(J_A, J_B)
- Δ_add = J_comp - (J_A + J_B - J_A * J_B)
- FH envelopes for unknown dependence via theory.fh_bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from cc.core.metrics import cc_max as cc_max_metric
from cc.core.metrics import delta_add, youden_j
from theory import fh_bounds


class Regime(Enum):
    """Dependence regime classification."""

    CONSTRUCTIVE = "constructive"
    INDEPENDENT = "independent"
    DESTRUCTIVE = "destructive"
    UNCERTAIN = "uncertain"
    DEGENERATE = "degenerate"
    UNDEFINED = "undefined"


@dataclass(frozen=True)
class CCAnalysis:
    """Results of a CC analysis for a guardrail pair."""

    cc_max: float
    cc_lower: Optional[float]
    cc_upper: Optional[float]
    regime: Regime
    confidence: float

    j_comp: float
    j_ind: float
    j_a: float
    j_b: float
    delta_add: float

    composition: str

    def __str__(self) -> str:
        header = "CC-Framework Analysis\n" + "=" * 22
        lines = [
            header,
            f"Composition: {self.composition}",
            f"CC_max: {self.cc_max:.4f}",
            f"Regime: {self.regime.value}",
            f"Confidence: {self.confidence:.2f}",
            "",
            "Youden's J:",
            f"  J_comp: {self.j_comp:.4f}",
            f"  J_ind:  {self.j_ind:.4f}",
            f"  J_A:    {self.j_a:.4f}",
            f"  J_B:    {self.j_b:.4f}",
            f"  Δ_add:  {self.delta_add:+.4f}",
        ]
        if self.cc_lower is not None and self.cc_upper is not None:
            lines.append(f"FH CC interval: [{self.cc_lower:.4f}, {self.cc_upper:.4f}]")
        return "\n".join(lines)


def compute_independence_baseline(j_a: float, j_b: float) -> float:
    """Compute the independence baseline for composed J (Δ_add baseline)."""

    return float(j_a + j_b - j_a * j_b)


def compute_cc_max(j_comp: float, j_a: float, j_b: float) -> float:
    """Compute CC_max = J_comp / max(J_A, J_B)."""

    return float(cc_max_metric(j_comp, j_a, j_b))


def _normalize_composition(composition: str) -> str:
    if composition.lower() in {"and", "parallel_and", "all_block"}:
        return "parallel_and"
    if composition.lower() in {"or", "serial_or", "any_block"}:
        return "serial_or"
    raise ValueError(f"Unknown composition: {composition!r}")


def compute_fh_cc_bounds(
    tpr: Sequence[float],
    fpr: Sequence[float],
    composition: str,
) -> Tuple[float, float]:
    """
    Compute FH CC bounds for a composition given per-rail TPR/FPR.

    Args:
        tpr: per-rail true-positive rates (threats blocked).
        fpr: per-rail false-positive rates (benign blocked).
        composition: "parallel_and" (AND) or "serial_or" (OR).

    Returns:
        (cc_lower, cc_upper) from FH J-interval divided by max single-rail J.
    """

    if len(tpr) != len(fpr):
        raise ValueError("tpr and fpr must have the same length")

    composition = _normalize_composition(composition)
    miss_rates = [1.0 - float(x) for x in tpr]
    false_alarm_rates = [float(x) for x in fpr]

    if composition == "parallel_and":
        bounds = fh_bounds.parallel_and_composition_bounds(miss_rates, false_alarm_rates)
    else:
        bounds = fh_bounds.serial_or_composition_bounds(miss_rates, false_alarm_rates)

    max_individual = max(bounds.individual_j_stats)
    cc_lower, cc_upper = fh_bounds.compute_cc_bounds(
        bounds.j_lower,
        bounds.j_upper,
        max_individual,
    )
    return float(cc_lower), float(cc_upper)


def classify_regime(
    cc_max_value: float,
    j_a: float,
    j_b: float,
    cc_lower: Optional[float] = None,
    cc_upper: Optional[float] = None,
    threshold_low: float = 0.95,
    threshold_high: float = 1.05,
) -> Regime:
    """Classify regime in line with FH/atlas thresholds."""

    if np.isnan(cc_max_value):
        return Regime.DEGENERATE

    if max(j_a, j_b) <= fh_bounds.MATHEMATICAL_TOLERANCE:
        return Regime.DEGENERATE

    if cc_lower is not None and cc_upper is not None:
        overlaps = not (cc_upper < threshold_low or cc_lower > threshold_high)
        if overlaps:
            return Regime.UNCERTAIN

    if cc_max_value < threshold_low:
        return Regime.CONSTRUCTIVE
    if cc_max_value <= threshold_high:
        return Regime.INDEPENDENT
    return Regime.DESTRUCTIVE


def analyze_composition(
    tpr_a: float,
    fpr_a: float,
    tpr_b: float,
    fpr_b: float,
    tpr_comp: float,
    fpr_comp: float,
    composition: str = "parallel_and",
    n_samples: int = 1000,
    compute_fh: bool = True,
) -> CCAnalysis:
    """
    Run CC analysis for a two-guardrail composition.

    Args:
        tpr_a, fpr_a: rates for guardrail A.
        tpr_b, fpr_b: rates for guardrail B.
        tpr_comp, fpr_comp: composed system rates.
        composition: "parallel_and" or "serial_or" (aliases: and/or).
        n_samples: sample size used for a simple confidence heuristic.
        compute_fh: whether to compute FH CC bounds.

    Returns:
        CCAnalysis with CC_max, FH interval, and regime classification.
    """

    j_a = float(youden_j(tpr_a, fpr_a))
    j_b = float(youden_j(tpr_b, fpr_b))
    j_comp = float(youden_j(tpr_comp, fpr_comp))
    j_ind = compute_independence_baseline(j_a, j_b)
    cc_max_value = compute_cc_max(j_comp, j_a, j_b)
    delta_add_value = float(delta_add(j_comp, j_a, j_b))

    cc_lower = cc_upper = None
    composition = _normalize_composition(composition)

    if compute_fh:
        cc_lower, cc_upper = compute_fh_cc_bounds(
            [tpr_a, tpr_b],
            [fpr_a, fpr_b],
            composition,
        )

    regime = classify_regime(
        cc_max_value,
        j_a,
        j_b,
        cc_lower=cc_lower,
        cc_upper=cc_upper,
    )

    sample_score = min(1.0, n_samples / 2000.0)
    effect = abs(cc_max_value - 1.0) if not np.isnan(cc_max_value) else 0.0
    effect_score = min(1.0, effect / 0.2 + 0.5)
    confidence = float(sample_score * effect_score)

    return CCAnalysis(
        cc_max=cc_max_value,
        cc_lower=cc_lower,
        cc_upper=cc_upper,
        regime=regime,
        confidence=confidence,
        j_comp=j_comp,
        j_ind=j_ind,
        j_a=j_a,
        j_b=j_b,
        delta_add=delta_add_value,
        composition=composition,
    )


def generate_regime_report(
    results: Iterable[CCAnalysis],
    output_path: str,
) -> None:
    """Write a plain-text report for a batch of CC analyses."""

    results_list = list(results)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("=" * 60 + "\n")
        handle.write("CC-FRAMEWORK REGIME ANALYSIS REPORT\n")
        handle.write("=" * 60 + "\n\n")

        handle.write("REGIME DISTRIBUTION\n")
        handle.write("-" * 30 + "\n")
        for regime in Regime:
            count = sum(1 for r in results_list if r.regime == regime)
            pct = 100 * count / len(results_list) if results_list else 0
            handle.write(f"{regime.value:15s}: {count:3d} ({pct:5.1f}%)\n")

        handle.write("\n" + "=" * 60 + "\n")
        handle.write("DETAILED RESULTS\n")
        handle.write("=" * 60 + "\n\n")

        for i, result in enumerate(results_list, start=1):
            handle.write(f"Analysis {i}\n")
            handle.write(str(result))
            handle.write("\n" + "-" * 60 + "\n\n")
