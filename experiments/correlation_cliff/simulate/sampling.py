from __future__ import annotations

"""
simulate.sampling
=================

Probability validation + multinomial sampling + empirical metric extraction.

Key policies:
- NO renormalization. Sum-to-1 must hold within prob_tol.
- Only tiny floating jitter clipping is allowed (diagnostic convenience).
- Strict int handling for counts to prevent silent truncation.
"""

from typing import Any, Dict, Tuple

import math
import numpy as np

from .config import Rule
from . import utils as U


def rng_for_cell(seed: int, rep: int, lam_index: int, world: int) -> np.random.Generator:
    """
    Stable-per-cell RNG keyed by (seed, rep, lambda_index, world).

    Order-invariant guarantee depends on:
      - caller using canonical lambda_index (cfg.lambda_index_for_seed(lam))
    """
    if not all(isinstance(x, int) for x in (seed, rep, lam_index, world)):
        raise TypeError("seed/rep/lam_index/world must all be ints.")
    ss = np.random.SeedSequence([seed, rep, lam_index, world])
    return np.random.default_rng(ss)


def validate_cell_probs(
    p: np.ndarray,
    *,
    prob_tol: float,
    allow_tiny_negative: bool,
    tiny_negative_eps: float,
    context: str = "",
) -> np.ndarray:
    """
    Validate (and optionally tiny-clip) a 4-cell probability vector for a 2×2 Bernoulli joint.

    Contract:
    - No renormalization.
    - Only tiny out-of-bounds jitter clipping is allowed.
    - Enforces: finite, in [0,1], shape (4,), sum within prob_tol of 1.
    """
    try:
        prob_tol_f = float(prob_tol)
        eps_f = float(tiny_negative_eps)
    except Exception as e:
        raise ValueError(
            f"prob_tol and tiny_negative_eps must be floats. got prob_tol={prob_tol!r}, eps={tiny_negative_eps!r}"
        ) from e

    if not (math.isfinite(prob_tol_f) and 0.0 <= prob_tol_f <= 1e-3):
        raise ValueError(f"prob_tol must be finite and reasonably small, got {prob_tol_f}")
    if allow_tiny_negative:
        if not (math.isfinite(eps_f) and 0.0 < eps_f <= 1e-6):
            raise ValueError(f"tiny_negative_eps must be finite in (0, 1e-6], got {eps_f}")

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.shape != (4,):
        if p_arr.size == 4:
            p_arr = p_arr.reshape(4,)
        else:
            raise ValueError(f"Expected p shape (4,), got {p_arr.shape} (size={p_arr.size}). {context}".strip())

    if not np.all(np.isfinite(p_arr)):
        raise ValueError(f"Non-finite probabilities: {p_arr.tolist()}. {context}".strip())

    pmin = float(p_arr.min())
    pmax = float(p_arr.max())
    clipped_any = False

    if pmin < 0.0:
        if allow_tiny_negative and pmin >= -eps_f:
            p_arr = p_arr.copy()
            p_arr[p_arr < 0.0] = 0.0
            clipped_any = True
        else:
            raise ValueError(f"Negative cell probability encountered: min={pmin}, p={p_arr.tolist()}. {context}".strip())

    if pmax > 1.0:
        if allow_tiny_negative and pmax <= 1.0 + eps_f:
            if not clipped_any:
                p_arr = p_arr.copy()
            p_arr[p_arr > 1.0] = 1.0
            clipped_any = True
        else:
            raise ValueError(f"Cell probability > 1 encountered: max={pmax}, p={p_arr.tolist()}. {context}".strip())

    if float(p_arr.min()) < 0.0 or float(p_arr.max()) > 1.0:
        raise ValueError(f"Probabilities out of bounds after clipping: p={p_arr.tolist()}. {context}".strip())

    s = float(p_arr.sum())
    if not math.isfinite(s):
        raise ValueError(f"Probability sum is non-finite: sum={s}, p={p_arr.tolist()}. {context}".strip())

    err = abs(s - 1.0)
    if err > prob_tol_f:
        clip_note = " (after tiny clipping)" if clipped_any else ""
        raise ValueError(
            f"Cell probabilities do not sum to 1 within tol: sum={s} (|Δ|={err}, tol={prob_tol_f}){clip_note}. "
            f"p={p_arr.tolist()}. {context}".strip()
        )

    return p_arr


def draw_joint_counts(
    rng: np.random.Generator,
    *,
    n: int,
    p00: float,
    p01: float,
    p10: float,
    p11: float,
    prob_tol: float,
    allow_tiny_negative: bool,
    tiny_negative_eps: float,
    context: str = "",
) -> Tuple[int, int, int, int]:
    """
    Draw multinomial joint counts (N00, N01, N10, N11) for a 2×2 Bernoulli joint.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}. {context}".strip())

    if isinstance(n, bool):
        raise TypeError(f"n must be an int > 0, got bool {n}. {context}".strip())
    try:
        n_int = int(n)
    except Exception as e:
        raise TypeError(f"n must be an int > 0, got {n!r}. {context}".strip()) from e
    if n_int <= 0:
        raise ValueError(f"n must be positive, got {n_int}. {context}".strip())
    if n_int != n:
        raise TypeError(f"n must be an integer (no silent coercion), got {n!r}. {context}".strip())

    p = np.array([p00, p01, p10, p11], dtype=np.float64)
    p = validate_cell_probs(
        p,
        prob_tol=prob_tol,
        allow_tiny_negative=allow_tiny_negative,
        tiny_negative_eps=tiny_negative_eps,
        context=context,
    )

    counts = rng.multinomial(n_int, pvals=p, size=None)
    if counts.shape != (4,):
        raise RuntimeError(f"Unexpected multinomial output shape: {counts.shape}, expected (4,). {context}".strip())

    c0, c1, c2, c3 = (int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]))
    s = c0 + c1 + c2 + c3
    if s != n_int:
        raise RuntimeError(f"Multinomial draw inconsistent: sum(counts)={s} != n={n_int}. {context}".strip())

    return c0, c1, c2, c3


def empirical_from_counts(
    *,
    n: int,
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    rule: Rule,
    context: str = "",
) -> Dict[str, float]:
    """
    Compute empirical probabilities from joint counts, plus dependence summaries.
    """
    ctx = f" {context}" if context else ""

    if rule not in ("OR", "AND"):
        raise ValueError(f"Invalid rule: {rule!r}.{ctx}")

    def _as_strict_int(x: Any, name: str) -> int:
        if isinstance(x, bool):
            raise TypeError(f"{name} must be an int, got bool {x}.{ctx}")
        if isinstance(x, (np.integer, int)):
            return int(x)
        raise TypeError(f"{name} must be an int, got {type(x).__name__}={x!r}.{ctx}")

    n_i = _as_strict_int(n, "n")
    n00_i = _as_strict_int(n00, "n00")
    n01_i = _as_strict_int(n01, "n01")
    n10_i = _as_strict_int(n10, "n10")
    n11_i = _as_strict_int(n11, "n11")

    if n_i <= 0:
        raise ValueError(f"n must be positive, got {n_i}.{ctx}")

    if (n00_i < 0) or (n01_i < 0) or (n10_i < 0) or (n11_i < 0):
        raise ValueError(
            f"Counts must be non-negative. Got (n00,n01,n10,n11)=({n00_i},{n01_i},{n10_i},{n11_i}).{ctx}"
        )

    s = n00_i + n01_i + n10_i + n11_i
    if s != n_i:
        raise ValueError(f"Counts do not sum to n. sum={s}, n={n_i}.{ctx}")

    inv_n = 1.0 / float(n_i)
    p00 = float(n00_i) * inv_n
    p01 = float(n01_i) * inv_n
    p10 = float(n10_i) * inv_n
    p11 = float(n11_i) * inv_n

    pA = float(n10_i + n11_i) * inv_n
    pB = float(n01_i + n11_i) * inv_n

    cells = {"p00": p00, "p01": p01, "p10": p10, "p11": p11}

    pC = float(U.pC_from_joint(rule, cells, pA=pA, pB=pB))
    phi = float(U.phi_from_joint(pA, pB, p11))
    tau = float(U.kendall_tau_a_from_joint(cells))

    degA = 1.0 if (pA <= 0.0 or pA >= 1.0) else 0.0
    degB = 1.0 if (pB <= 0.0 or pB >= 1.0) else 0.0
    phi_finite = 1.0 if math.isfinite(phi) else 0.0
    tau_finite = 1.0 if math.isfinite(tau) else 0.0

    return {
        "n": float(n_i),
        "n00": float(n00_i),
        "n01": float(n01_i),
        "n10": float(n10_i),
        "n11": float(n11_i),
        "p00_hat": float(p00),
        "p01_hat": float(p01),
        "p10_hat": float(p10),
        "p11_hat": float(p11),
        "pA_hat": float(pA),
        "pB_hat": float(pB),
        "pC_hat": float(pC),
        "phi_hat": float(phi),
        "tau_hat": float(tau),
        "degenerate_A": float(degA),
        "degenerate_B": float(degB),
        "phi_finite": float(phi_finite),
        "tau_finite": float(tau_finite),
    }
