# src/cc/cartographer/atlas.py
"""
Module: atlas
Purpose: Plot the CC phase point and compose human-readable decision log lines
Dependencies: matplotlib
Author: Pranav Bhave
Date: 2025-08-31 (refined 2025-09-03)

Notes
-----
- Uses a tiny, dependency-free plotting style suitable for CI smoke runs.
- `plot_phase_point` writes a single figure marking (epsilon, T) and annotates the title
  with CC_max. It creates parent directories as needed.
- `compose_entry` returns two strings:
    1) a one-line human-readable summary for memos/PR comments
    2) a DECISION line keyed off CC_max region thresholds
"""

from __future__ import annotations

import os
from typing import Any, Final, Mapping, Optional, Tuple

import matplotlib.pyplot as plt

__all__ = ["plot_phase_point", "compose_entry"]

# Thresholds for interpreting CC_max (hard invariants for CLI policy)
_CC_CONSTRUCTIVE_MAX: Final[float] = 0.95
_CC_INDEPENDENT_MAX: Final[float] = 1.05


def plot_phase_point(cfg: Mapping[str, Any], cc_max: float, outfile: str) -> str:
    """
    Save a small phase-point plot marking (epsilon, T) and return the output path.

    Args:
        cfg: mapping with keys 'epsilon' and 'T' (floats or strings)
        cc_max: dashboard normalization (J_comp / max(J_A, J_B))
        outfile: destination path (PDF/PNG); parent dirs will be created

    Returns:
        The outfile path (for logging/audit references).
    """
    parent = os.path.dirname(outfile) or "."
    os.makedirs(parent, exist_ok=True)

    def _to_float(x: Any, default: float = 0.0) -> float:
        if x is None:
            return default
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    eps = _to_float(cfg.get("epsilon"), 0.0)
    T = _to_float(cfg.get("T"), 0.0)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    ax.scatter([eps], [T], s=80)
    ax.set_title(f"CC phase point: CC_max={cc_max:.2f}")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("T")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def compose_entry(
    cfg: Mapping[str, Any],
    j_a: float,
    j_a_ci: Optional[Tuple[Optional[float], Optional[float]]],
    j_b: float,
    j_b_ci: Optional[Tuple[Optional[float], Optional[float]]],
    j_comp: float,
    j_comp_ci: Optional[Tuple[Optional[float], Optional[float]]],
    cc_max: float,
    delta_add: float,
    comp_label: str,
    fig_path: str,
) -> Tuple[str, str]:
    """
    Compose a human-readable log entry and a decision line for CLI output.

    Args:
        cfg: experiment configuration mapping (expects keys 'epsilon','T','A','B','comp')
        j_a, j_b, j_comp: Youden’s J statistics for A, B, and composition (or bound)
        j_a_ci, j_b_ci, j_comp_ci: optional (lo, hi) confidence intervals
        cc_max: J_comp / max(J_A, J_B) (reporting normalization)
        delta_add: J_comp - (J_A + J_B - J_A*J_B) (heuristic additive delta)
        comp_label: 'empirical' or 'UPPER BOUND' (shown in log)
        fig_path: path to the figure for reference

    Returns:
        (entry_line, decision_line)
    """
    eps = cfg.get("epsilon", "?")
    T = cfg.get("T", "?")
    name_a = str(cfg.get("A", "A"))
    name_b = str(cfg.get("B", "B"))
    comp = str(cfg.get("comp", "AND"))

    def _fmt_ci(ci: Optional[Tuple[Optional[float], Optional[float]]]) -> str:
        # Show CI only when both bounds are present
        if ci is None:
            return ""
        lo, hi = ci
        if lo is None or hi is None:
            return ""
        return f" [{lo:.2f}, {hi:.2f}]"

    entry = (
        f"ε={eps}, T={T}; {name_a} ⊕ {name_b} ({comp}) — "
        f"J_A={j_a:.2f}{_fmt_ci(j_a_ci)}, "
        f"J_B={j_b:.2f}{_fmt_ci(j_b_ci)}, "
        f"J_comp({comp_label})={j_comp:.2f}{_fmt_ci(j_comp_ci)} ⇒ "
        f"CC_max={cc_max:.2f}, Δ_add={delta_add:+.2f}. "
        f"{_region(cc_max)}. Next: probe nearby (ε,T)."
    )
    decision = f"DECISION: {_decision(cc_max)} — reason: CC_max={cc_max:.2f}; refs: {fig_path}"
    return entry, decision


# --- Internals ---------------------------------------------------------------

def _region(cc_max: float) -> str:
    if cc_max < _CC_CONSTRUCTIVE_MAX:
        return "Constructive Valley"
    if cc_max <= _CC_INDEPENDENT_MAX:
        return "Independent Plateau"
    return "Red Wedge (Destructive)"


def _decision(cc_max: float) -> str:
    if cc_max < _CC_CONSTRUCTIVE_MAX:
        return "ADOPT HYBRID"
    if cc_max <= _CC_INDEPENDENT_MAX:
        return "PREFER SINGLE"
    return "REDESIGN"