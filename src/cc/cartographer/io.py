# src/cc/cartographer/io.py
"""
Module: io
Purpose: Typed config loading with mapping validation and helpers for Cartographer
Dependencies: yaml, typing, numpy
Author: Pranav Bhave
Date: 2025-08-31 (refined 2025-09-03)

Capabilities
------------
- Safe YAML loader with strict "mapping required" validation.
- Robust parsers for integer-like fields (e.g., samples).
- Lightweight ROC construction from score vectors (deterministic, CPU-cheap).
- Deterministic toy score generators for smoke/e2e runs.

Conventions
-----------
- World 0 (benign) scores: s0
- World 1 (attack)  scores: s1
- ROC is an array of shape (K, 2) with columns [FPR, TPR], clipped to [0,1].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import yaml

# =============================================================================
# Config loading
# =============================================================================


def load_config(path: str) -> Mapping[str, Any]:
    """
    Load a YAML config file and validate it's a mapping.

    Raises:
        FileNotFoundError: if the file does not exist.
        TypeError: if the root YAML node is not a mapping.
        yaml.YAMLError: if YAML parsing fails.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} is not a mapping")
    return cast(Mapping[str, Any], data)


# =============================================================================
# Parsers
# =============================================================================


def parse_samples(n: int | None, cfg: Mapping[str, Any], default: int = 200) -> int:
    """
    Robustly resolve sample size from CLI integer or cfg['samples'].

    Rules:
      - If `n` is an int, use it.
      - Else, try cfg['samples'] as int-like (int or numeric string).
      - Ignore booleans (tend to pass mypy checks but are not valid here).
      - Fallback to `default` on any failure.

    Args:
        n: Optional override from CLI.
        cfg: Config mapping.
        default: Fallback value.

    Returns:
        Integer number of samples (>= 1).
    """
    if isinstance(n, int):
        return max(1, n)

    val: Any = cfg.get("samples", default)
    if isinstance(val, bool):
        return max(1, default)
    if isinstance(val, int):
        return max(1, val)
    if isinstance(val, str):
        try:
            return max(1, int(val.strip()))
        except ValueError:
            return max(1, default)
    return max(1, default)


# =============================================================================
# ROC helpers
# =============================================================================


def _roc_from_scores(s0: np.ndarray, s1: np.ndarray, max_pts: int = 256) -> np.ndarray:
    """
    Construct a ROC array [[FPR, TPR], ...] from score vectors.

    Conventions:
      - Higher score => more likely to predict "attack"/positive.
      - Threshold rule: predict positive if score >= t.
      - Uses pooled unique thresholds; subsamples if too many.

    Args:
        s0: World-0 (benign) scores, shape (N0,).
        s1: World-1 (attack) scores, shape (N1,).
        max_pts: Max number of thresholds to evaluate.

    Returns:
        np.ndarray of shape (K, 2), columns [FPR, TPR], clipped to [0,1] and
        including anchors (0,0) and (1,1).
    """
    s0 = np.asarray(s0, dtype=float).ravel()
    s1 = np.asarray(s1, dtype=float).ravel()

    if s0.size == 0 or s1.size == 0:
        return np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)

    thr = np.unique(np.concatenate([s0, s1], axis=0))
    if thr.size > max_pts:
        idx = np.linspace(0, thr.size - 1, max_pts).round().astype(int)
        thr = thr[idx]

    # Evaluate at thresholds in descending order (strict→lenient)
    thr = thr[::-1]
    # Vectorized comparisons to build TPR/FPR arrays
    tpr = (s1[:, None] >= thr[None, :]).mean(axis=0)  # P(pred=1 | world=1)
    fpr = (s0[:, None] >= thr[None, :]).mean(axis=0)  # P(pred=1 | world=0)

    roc_core = np.stack([fpr, tpr], axis=1)

    # Add anchors and sort by FPR asc for monotone plotting/interp
    anchors = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    roc = np.vstack([anchors[0:1], roc_core, anchors[1:2]])
    # Sort + de-duplicate on FPR to avoid small numeric noise
    order = np.argsort(roc[:, 0], kind="mergesort")
    roc = roc[order]
    # Clip numerical stragglers
    roc = np.clip(roc, 0.0, 1.0)
    return roc


# =============================================================================
# Toy score generator for smoke/e2e
# =============================================================================


def load_scores(cfg: Mapping[str, Any], n: int | None = None) -> Mapping[str, Any]:
    """
    Generate deterministic toy A/B score sets and their ROC curves for smoke runs.

    Config keys (optional):
      - seed: RNG seed (int; default 1337)
      - mu_A, mu_B: mean shift for world-1 vs world-0 (floats; defaults 1.0, 0.7)
      - sigma: standard deviation for both worlds (float; default 1.0)
      - samples: number of samples per world (int; default 200 or CLI `n`)

    Returns keys expected by pipeline/CLI:
      - A0, A1: world0/world1 scores for component A (np.ndarray)
      - B0, B1: world0/world1 scores for component B (np.ndarray)
      - Comp0, Comp1: composed empirical scores (None for smoke)
      - rocA, rocB: ROC arrays [[FPR, TPR], ...] for A and B
    """
    size = parse_samples(n, cfg, default=200)
    seed = int(cfg.get("seed", 1337) or 1337)
    mu_A = float(cfg.get("mu_A", 1.0) or 1.0)
    mu_B = float(cfg.get("mu_B", 0.7) or 0.7)
    sigma = float(cfg.get("sigma", 1.0) or 1.0)

    rng = np.random.default_rng(seed)

    # Component A
    A0 = rng.normal(loc=0.0, scale=sigma, size=size)
    A1 = rng.normal(loc=mu_A, scale=sigma, size=size)
    rocA = _roc_from_scores(A0, A1)

    # Component B
    B0 = rng.normal(loc=0.0, scale=sigma, size=size)
    B1 = rng.normal(loc=mu_B, scale=sigma, size=size)
    rocB = _roc_from_scores(B0, B1)

    out: dict[str, Any] = {
        "A0": A0,
        "A1": A1,
        "B0": B0,
        "B1": B1,
        "rocA": rocA,
        "rocB": rocB,
        # No empirical composed scores for smoke
        "Comp0": None,
        "Comp1": None,
    }
    return out
