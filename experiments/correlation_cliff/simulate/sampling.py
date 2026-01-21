from __future__ import annotations

"""
simulate.sampling
=================

Probability validation + multinomial sampling + empirical metric extraction.

Key policies:
- NO renormalization: sum-to-1 must hold within prob_tol.
- Only tiny out-of-bounds (OOB) floating jitter clipping is allowed (diagnostic convenience).
- Strict int handling for counts to prevent silent truncation.
- IMPORTANT: NumPy multinomial historically treats the last pval as "leftover mass".
  We therefore enforce consistency between p11 and 1 - (p00+p01+p10) and sample using
  the remainder explicitly to avoid silent drift.

See:
- numpy.random.Generator.multinomial docs (last entry ignored as leftover mass)  # documented behavior
"""

import math
from typing import Any

import numpy as np

from . import utils as U
from .config import Rule

_FLOAT64_EPS = float(np.finfo(np.float64).eps)
# A "rounding-only" mismatch bound between p11 and 1 - sum(p00,p01,p10).
# This is intentionally independent of prob_tol; it protects "what distribution do we sample".
_LAST_MISMATCH_EPS = 64.0 * _FLOAT64_EPS


class ProbabilityValidationError(ValueError):
    """Raised when a probability vector violates the sampling contract."""


def rng_for_cell(seed: int, rep: int, lam_index: int, world: int) -> np.random.Generator:
    """
    Stable-per-cell RNG keyed by (seed, rep, lambda_index, world).

    Order-invariant guarantee depends on:
      - caller using canonical lam_index (cfg.lambda_index_for_seed(lam))
    """
    if not all(isinstance(x, int) for x in (seed, rep, lam_index, world)):
        raise TypeError("seed/rep/lam_index/world must all be ints.")
    ss = np.random.SeedSequence([seed, rep, lam_index, world])
    return np.random.default_rng(ss)


def validate_cell_probs_with_meta(
    p: np.ndarray,
    *,
    prob_tol: float,
    allow_tiny_negative: bool,
    tiny_negative_eps: float,
    context: str = "",
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Validate (and optionally tiny-clip) a 4-cell probability vector with diagnostics.

    Returns (p_arr, meta) where meta includes sum/error and clipping information.
    """
    ctx = f" {context}".strip()

    try:
        prob_tol_f = float(prob_tol)
        eps_f = float(tiny_negative_eps)
    except Exception as e:
        raise ProbabilityValidationError(
            f"prob_tol and tiny_negative_eps must be floats. got prob_tol={prob_tol!r}, eps={tiny_negative_eps!r}. {ctx}"
        ) from e

    if not (math.isfinite(prob_tol_f) and prob_tol_f >= 0.0):
        raise ProbabilityValidationError(
            f"prob_tol must be finite and >= 0, got {prob_tol_f}. {ctx}"
        )
    if prob_tol_f > 1e-2:
        raise ProbabilityValidationError(
            f"prob_tol is suspiciously large (>1e-2): {prob_tol_f}. {ctx}"
        )

    if allow_tiny_negative and not (math.isfinite(eps_f) and 0.0 < eps_f <= 1e-4):
        raise ProbabilityValidationError(
            f"tiny_negative_eps must be finite in (0, 1e-4], got {eps_f}. {ctx}"
        )

    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.shape != (4,):
        if p_arr.size == 4:
            p_arr = p_arr.reshape(
                4,
            )
        else:
            raise ProbabilityValidationError(
                f"Expected p shape (4,), got {p_arr.shape} (size={p_arr.size}). {ctx}".strip()
            )

    if not np.all(np.isfinite(p_arr)):
        raise ProbabilityValidationError(
            f"Non-finite probabilities: {p_arr.tolist()}. {ctx}".strip()
        )

    pmin = float(p_arr.min())
    pmax = float(p_arr.max())
    clipped_any = False
    clipped_low = False
    clipped_high = False

    if pmin < 0.0:
        if allow_tiny_negative and pmin >= -eps_f:
            p_arr = p_arr.copy()
            p_arr[p_arr < 0.0] = 0.0
            clipped_any = True
            clipped_low = True
        else:
            raise ProbabilityValidationError(
                f"Negative cell probability encountered: min={pmin}, p={p_arr.tolist()}. {ctx}".strip()
            )

    if pmax > 1.0:
        if allow_tiny_negative and pmax <= 1.0 + eps_f:
            if not clipped_any:
                p_arr = p_arr.copy()
            p_arr[p_arr > 1.0] = 1.0
            clipped_any = True
            clipped_high = True
        else:
            raise ProbabilityValidationError(
                f"Cell probability > 1 encountered: max={pmax}, p={p_arr.tolist()}. {ctx}".strip()
            )

    if float(p_arr.min()) < 0.0 or float(p_arr.max()) > 1.0:
        raise ProbabilityValidationError(
            f"Probabilities out of bounds after clipping: p={p_arr.tolist()}. {ctx}".strip()
        )

    s = float(p_arr.sum())
    if not math.isfinite(s):
        raise ProbabilityValidationError(
            f"Probability sum is non-finite: sum={s}, p={p_arr.tolist()}. {ctx}".strip()
        )

    err = abs(s - 1.0)
    if err > prob_tol_f:
        clip_note = " (after tiny clipping)" if clipped_any else ""
        raise ProbabilityValidationError(
            f"Cell probabilities do not sum to 1 within tol: sum={s} (|Δ|={err}, tol={prob_tol_f}){clip_note}. "
            f"p={p_arr.tolist()}. {ctx}".strip()
        )

    meta = {
        "sum_p": float(s),
        "sum_error": float(err),
        "pmin": float(p_arr.min()),
        "pmax": float(p_arr.max()),
        "clipped_any": float(clipped_any),
        "clipped_low": float(clipped_low),
        "clipped_high": float(clipped_high),
        "prob_tol": float(prob_tol_f),
        "tiny_negative_eps": float(eps_f) if allow_tiny_negative else float("nan"),
    }
    return p_arr, meta


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
    - Only tiny OOB jitter clipping is allowed.
    - Enforces: finite, in [0,1], shape (4,), sum within prob_tol of 1.
    """
    p_arr, _ = validate_cell_probs_with_meta(
        p,
        prob_tol=prob_tol,
        allow_tiny_negative=allow_tiny_negative,
        tiny_negative_eps=tiny_negative_eps,
        context=context,
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
    return_meta: bool = False,
) -> tuple[int, int, int, int] | tuple[tuple[int, int, int, int], dict[str, float]]:
    """
    Draw multinomial joint counts (N00, N01, N10, N11) for a 2×2 Bernoulli joint.

    Implementation note:
    NumPy multinomial treats the last probability entry as leftover mass in common implementations.
    We therefore enforce p11 consistency with 1 - (p00+p01+p10) up to rounding noise, then sample
    using the explicit remainder so "what we sample" is unambiguous.
    """
    ctx = f" {context}".strip()

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}. {ctx}".strip()
        )

    if isinstance(n, bool):
        raise TypeError(f"n must be an int > 0, got bool {n}. {ctx}".strip())
    if isinstance(n, (float, np.floating)):
        raise TypeError(f"n must be an integer (no silent coercion), got {n!r}. {ctx}".strip())
    try:
        n_int = int(n)
    except Exception as e:
        raise TypeError(f"n must be an int > 0, got {n!r}. {ctx}".strip()) from e
    if n_int <= 0:
        raise ValueError(f"n must be positive, got {n_int}. {ctx}".strip())
    if n_int != n:
        raise TypeError(f"n must be an integer (no silent coercion), got {n!r}. {ctx}".strip())

    p = np.array([p00, p01, p10, p11], dtype=np.float64)
    p, meta = validate_cell_probs_with_meta(
        p,
        prob_tol=prob_tol,
        allow_tiny_negative=allow_tiny_negative,
        tiny_negative_eps=tiny_negative_eps,
        context=context,
    )

    sum3 = float(p[0] + p[1] + p[2])
    if not math.isfinite(sum3):
        raise RuntimeError(
            f"Non-finite partial sum for p00+p01+p10: {sum3}. p={p.tolist()}. {ctx}".strip()
        )

    remainder = 1.0 - sum3

    # If remainder is slightly negative due to rounding, allow tiny clipping if enabled.
    if remainder < 0.0:
        if allow_tiny_negative and remainder >= -float(tiny_negative_eps):
            remainder = 0.0
        else:
            raise ProbabilityValidationError(
                f"Invalid remainder for last cell: 1 - (p00+p01+p10) = {remainder}. p={p.tolist()}. {ctx}".strip()
            )

    if remainder > 1.0 + _LAST_MISMATCH_EPS:
        raise ProbabilityValidationError(
            f"Invalid remainder for last cell (>1): remainder={remainder}. p={p.tolist()}. {ctx}".strip()
        )

    # Enforce "distribution identity": provided p11 must match implied remainder up to float rounding.
    mismatch = abs(float(p[3]) - float(remainder))
    mismatch_tol = max(_LAST_MISMATCH_EPS, float(prob_tol))
    if mismatch > mismatch_tol:
        raise ProbabilityValidationError(
            "Provided p11 is inconsistent with 1 - (p00+p01+p10) beyond tolerance. "
            f"p11={float(p[3])}, remainder={float(remainder)}, |Δ|={mismatch}, tol=max({_LAST_MISMATCH_EPS},{float(prob_tol)}). "
            f"p={p.tolist()}. {ctx}".strip()
        )

    # Sample with explicit remainder for clarity.
    pvals = np.array([float(p[0]), float(p[1]), float(p[2]), float(remainder)], dtype=np.float64)
    counts = rng.multinomial(n_int, pvals=pvals, size=None)

    if counts.shape != (4,):
        raise RuntimeError(
            f"Unexpected multinomial output shape: {counts.shape}, expected (4,). {ctx}".strip()
        )

    c0, c1, c2, c3 = (int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]))
    s = c0 + c1 + c2 + c3
    if s != n_int:
        raise RuntimeError(
            f"Multinomial draw inconsistent: sum(counts)={s} != n={n_int}. {ctx}".strip()
        )

    meta.update(
        {
            "p00_used": float(p[0]),
            "p01_used": float(p[1]),
            "p10_used": float(p[2]),
            "p11_used": float(remainder),
            "remainder": float(remainder),
            "p11_provided": float(p[3]),
            "p11_mismatch": float(mismatch),
            "p11_mismatch_tol": float(mismatch_tol),
        }
    )

    if return_meta:
        return (c0, c1, c2, c3), meta
    return c0, c1, c2, c3


def draw_joint_counts_batch(
    rng: np.random.Generator,
    *,
    n: int,
    p00: float,
    p01: float,
    p10: float,
    p11: float,
    size: int,
    prob_tol: float,
    allow_tiny_negative: bool,
    tiny_negative_eps: float,
    context: str = "",
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """
    Vectorized multinomial draws for repeated samples at fixed joint probabilities.
    """
    ctx = f" {context}".strip()

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}. {ctx}".strip()
        )
    if isinstance(size, bool):
        raise TypeError(f"size must be an int > 0, got bool {size}. {ctx}".strip())
    try:
        size_int = int(size)
    except Exception as e:
        raise TypeError(f"size must be an int > 0, got {size!r}. {ctx}".strip()) from e
    if size_int <= 0:
        raise ValueError(f"size must be positive, got {size_int}. {ctx}".strip())
    if size_int != size:
        raise TypeError(
            f"size must be an integer (no silent coercion), got {size!r}. {ctx}".strip()
        )

    if isinstance(n, bool):
        raise TypeError(f"n must be an int > 0, got bool {n}. {ctx}".strip())
    try:
        n_int = int(n)
    except Exception as e:
        raise TypeError(f"n must be an int > 0, got {n!r}. {ctx}".strip()) from e
    if n_int <= 0:
        raise ValueError(f"n must be positive, got {n_int}. {ctx}".strip())
    if n_int != n:
        raise TypeError(f"n must be an integer (no silent coercion), got {n!r}. {ctx}".strip())

    p = np.array([p00, p01, p10, p11], dtype=np.float64)
    p, meta = validate_cell_probs_with_meta(
        p,
        prob_tol=prob_tol,
        allow_tiny_negative=allow_tiny_negative,
        tiny_negative_eps=tiny_negative_eps,
        context=context,
    )

    sum3 = float(p[0] + p[1] + p[2])
    if not math.isfinite(sum3):
        raise RuntimeError(
            f"Non-finite partial sum for p00+p01+p10: {sum3}. p={p.tolist()}. {ctx}".strip()
        )

    remainder = 1.0 - sum3
    if remainder < 0.0:
        if allow_tiny_negative and remainder >= -float(tiny_negative_eps):
            remainder = 0.0
        else:
            raise ProbabilityValidationError(
                f"Invalid remainder for last cell: 1 - (p00+p01+p10) = {remainder}. p={p.tolist()}. {ctx}".strip()
            )
    if remainder > 1.0 + _LAST_MISMATCH_EPS:
        raise ProbabilityValidationError(
            f"Invalid remainder for last cell (>1): remainder={remainder}. p={p.tolist()}. {ctx}".strip()
        )

    mismatch = abs(float(p[3]) - float(remainder))
    mismatch_tol = max(_LAST_MISMATCH_EPS, float(prob_tol))
    if mismatch > mismatch_tol:
        raise ProbabilityValidationError(
            "Provided p11 is inconsistent with 1 - (p00+p01+p10) beyond tolerance. "
            f"p11={float(p[3])}, remainder={float(remainder)}, |Δ|={mismatch}, tol=max({_LAST_MISMATCH_EPS},{float(prob_tol)}). "
            f"p={p.tolist()}. {ctx}".strip()
        )

    meta.update(
        {
            "p00_used": float(p[0]),
            "p01_used": float(p[1]),
            "p10_used": float(p[2]),
            "p11_used": float(remainder),
            "remainder": float(remainder),
            "p11_provided": float(p[3]),
            "p11_mismatch": float(mismatch),
            "p11_mismatch_tol": float(mismatch_tol),
        }
    )

    pvals = np.array([float(p[0]), float(p[1]), float(p[2]), float(remainder)], dtype=np.float64)
    counts_arr = rng.multinomial(n_int, pvals=pvals, size=size_int)

    if counts_arr.shape != (size_int, 4):
        raise RuntimeError(
            f"Unexpected multinomial output shape: {counts_arr.shape}, expected ({size_int}, 4). {ctx}".strip()
        )
    if not np.all(counts_arr.sum(axis=1) == n_int):
        raise RuntimeError(
            f"Multinomial batch draw inconsistent: some rows do not sum to n={n_int}. {ctx}".strip()
        )
    if np.any(counts_arr < 0):
        raise RuntimeError(
            f"Multinomial batch draw inconsistent: negative counts detected. {ctx}".strip()
        )

    if return_meta:
        return counts_arr.astype(int), meta
    return counts_arr.astype(int)


def empirical_from_counts(
    *,
    n: int,
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    rule: Rule,
    context: str = "",
) -> dict[str, float]:
    """
    Compute empirical probabilities from joint counts, plus dependence summaries.
    """
    ctx = f" {context}".strip()

    if rule not in ("OR", "AND"):
        raise ValueError(f"Invalid rule: {rule!r}. {ctx}".strip())

    def _as_strict_int(x: Any, name: str) -> int:
        if isinstance(x, bool):
            raise TypeError(f"{name} must be an int, got bool {x}. {ctx}".strip())
        if isinstance(x, (np.integer, int)):
            return int(x)
        raise TypeError(f"{name} must be an int, got {type(x).__name__}={x!r}. {ctx}".strip())

    n_i = _as_strict_int(n, "n")
    n00_i = _as_strict_int(n00, "n00")
    n01_i = _as_strict_int(n01, "n01")
    n10_i = _as_strict_int(n10, "n10")
    n11_i = _as_strict_int(n11, "n11")

    if n_i <= 0:
        raise ValueError(f"n must be positive, got {n_i}. {ctx}".strip())

    if (n00_i < 0) or (n01_i < 0) or (n10_i < 0) or (n11_i < 0):
        raise ValueError(
            f"Counts must be non-negative. Got (n00,n01,n10,n11)=({n00_i},{n01_i},{n10_i},{n11_i}). {ctx}".strip()
        )

    s = n00_i + n01_i + n10_i + n11_i
    if s != n_i:
        raise ValueError(f"Counts do not sum to n. sum={s}, n={n_i}. {ctx}".strip())

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

    # Degeneracy computed from counts => exact, no float edge ambiguity.
    a_ones = n10_i + n11_i
    b_ones = n01_i + n11_i
    degA = 1.0 if (a_ones == 0 or a_ones == n_i) else 0.0
    degB = 1.0 if (b_ones == 0 or b_ones == n_i) else 0.0

    phi_finite = 1.0 if math.isfinite(phi) else 0.0
    tau_finite = 1.0 if math.isfinite(tau) else 0.0

    return {
        "n": float(n_i),
        "n00": float(n00_i),
        "n01": float(n01_i),
        "n10": float(n10_i),
        "n11": float(n11_i),
        "p00_hat": p00,
        "p01_hat": p01,
        "p10_hat": p10,
        "p11_hat": p11,
        "pA_hat": pA,
        "pB_hat": pB,
        "pC_hat": pC,
        "phi_hat": phi,
        "tau_hat": tau,
        "degenerate_A": float(degA),
        "degenerate_B": float(degB),
        "phi_finite": float(phi_finite),
        "tau_finite": float(tau_finite),
    }
