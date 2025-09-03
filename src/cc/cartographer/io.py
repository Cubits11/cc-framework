"""
Module: io
Purpose: Typed config loading with mapping validation and helpers for Cartographer
Dependencies: yaml, typing, numpy
Author: Pranav Bhave
Date: 2025-08-31
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, cast

import yaml
import numpy as np


def load_config(path: str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} is not a mapping")
    return cast(Mapping[str, Any], data)


def parse_samples(n: Optional[int], cfg: Mapping[str, Any], default: int = 200) -> int:
    """
    Robustly resolve sample size from CLI integer or cfg['samples'].
    Avoids mypy 'Any' leaks and None issues.
    """
    if isinstance(n, int):
        return n

    val: Any = cfg.get("samples", default)
    if isinstance(val, bool):
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val.strip())
        except ValueError:
            return default
    return default


def _roc_from_scores(s0: np.ndarray, s1: np.ndarray, max_pts: int = 256) -> np.ndarray:
    """Return ROC as array [[FPR, TPR], ...] from score vectors."""
    s0 = np.asarray(s0, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    thr = np.unique(np.concatenate([s0, s1], axis=0))
    if thr.size > max_pts:
        idx = np.linspace(0, thr.size - 1, max_pts).round().astype(int)
        thr = thr[idx]
    tpr = (s1[:, None] >= thr[None, :]).mean(axis=0)
    fpr = (s0[:, None] >= thr[None, :]).mean(axis=0)
    roc = np.stack([fpr, tpr], axis=1)
    return np.clip(roc, 0.0, 1.0)


def load_scores(cfg: Mapping[str, Any], n: Optional[int] = None) -> Mapping[str, Any]:
    """
    Generate deterministic toy A/B score sets and their ROC curves for smoke runs.

    Returns keys expected by CLI:
      - A0, A1: world0/world1 scores for component A
      - B0, B1: world0/world1 scores for component B
      - (optional) Comp0, Comp1: composed empirical scores (None in smoke)
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

    out: Dict[str, Any] = {
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