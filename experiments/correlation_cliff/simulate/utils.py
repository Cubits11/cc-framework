from __future__ import annotations

"""
simulate.utils
==============

Shared utilities for the correlation_cliff simulation harness.

Primary responsibility:
- Import the local `experiments.correlation_cliff.theory` façade without hiding bugs.
- Export the required theory symbols as a stable surface for simulate/* modules.
- Provide small deterministic helpers (like lambda grid builder).
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import logging
import math

import numpy as np

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Import theory without hiding bugs (and without trusting a wrong shadow module)
# -----------------------------------------------------------------------------
def _import_theory_module():
    """
    Import the local correlation_cliff theory module without hiding real defects.

    Strategy:
    - Prefer package-relative import: from .. import theory  (when imported as a package)
    - If and only if that fails with ImportError, fall back to script import: import theory

    Guardrail:
    - After import, validate expected symbols exist to reduce sys.path shadowing risks.
    """
    # 1) Prefer package-relative import
    try:
        from .. import theory as T  # type: ignore
        mode = "package"
    except ImportError:
        # 2) Script-style fallback (only on ImportError)
        try:
            import theory as T  # type: ignore
            mode = "script"
        except ImportError as e:
            raise ImportError(
                "Could not import correlation_cliff theory module.\n"
                "Recommended invocation from repo root:\n"
                "  python -m experiments.correlation_cliff.simulate --config path/to/config.yaml\n"
                "Or install the package so relative imports work."
            ) from e

    required = [
        "FHBounds",
        "TwoWorldMarginals",
        "WorldMarginals",
        "compute_fh_jc_envelope",
        "fh_bounds",
        "joint_cells_from_marginals",
        "kendall_tau_a_from_joint",
        "phi_from_joint",
        "p11_fh_linear",
        "pC_from_joint",
    ]
    missing = [name for name in required if not hasattr(T, name)]
    if missing:
        origin = getattr(T, "__file__", "<unknown>")
        raise ImportError(
            "Imported a module named 'theory' but it does not look like the "
            "correlation_cliff theory surface.\n"
            f"Import mode: {mode}\n"
            f"Module origin: {origin}\n"
            f"Missing expected exports: {missing}\n"
            "This usually means sys.path shadowing (imported the wrong module), "
            "or your correlation_cliff/theory.py is incomplete."
        )

    LOG.debug("Imported theory module in %s mode from %s", mode, getattr(T, "__file__", "<unknown>"))
    return T, mode


T, THEORY_IMPORT_MODE = _import_theory_module()

# Bind required symbols explicitly (crash early if theory surface changes).
FHBounds = T.FHBounds
TwoWorldMarginals = T.TwoWorldMarginals
WorldMarginals = T.WorldMarginals
compute_fh_jc_envelope = T.compute_fh_jc_envelope
fh_bounds = T.fh_bounds
joint_cells_from_marginals = T.joint_cells_from_marginals
kendall_tau_a_from_joint = T.kendall_tau_a_from_joint
phi_from_joint = T.phi_from_joint
p11_fh_linear = T.p11_fh_linear
pC_from_joint = T.pC_from_joint

# Optional: reference overlay helper (never assumed path-consistent).
compute_metrics_for_lambda: Optional[Callable[..., Dict[str, float]]] = getattr(T, "compute_metrics_for_lambda", None)


# -----------------------------------------------------------------------------
# Convenience: deterministic lambda grid builder
# -----------------------------------------------------------------------------
def build_linear_lambda_grid(
    num: int,
    *,
    start: float = 0.0,
    stop: float = 1.0,
    closed: Literal["both", "neither", "left", "right"] = "both",
    dtype: Any = float,
    snap_eps: float = 0.0,
) -> np.ndarray:
    """
    Build a deterministic, validated lambda grid on [start, stop] (default [0,1]).

    Contract:
    - `num` is the number of points returned (after endpoint handling).
    - validates finiteness + interval ordering; optional endpoint snapping.

    Endpoint control via `closed`:
      * "both"    -> include start and stop (standard linspace)
      * "neither" -> exclude both endpoints
      * "left"    -> include start, exclude stop
      * "right"   -> exclude start, include stop

    Returns
    -------
    np.ndarray
        1D array, strictly increasing (unless num==1), of length `num`.
    """
    try:
        num_i = int(num)
    except Exception as e:
        raise ValueError(f"num must be an int, got {num!r}") from e
    if num_i < 1:
        raise ValueError(f"num must be >= 1, got {num_i}")

    a = float(start)
    b = float(stop)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"start/stop must be finite, got start={start!r}, stop={stop!r}")
    if not (a < b):
        raise ValueError(f"Require start < stop, got start={a}, stop={b}")
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError(f"Lambda grid expects start/stop in [0,1], got start={a}, stop={b}")

    if closed not in ("both", "neither", "left", "right"):
        raise ValueError(f"closed must be one of {{both, neither, left, right}}, got {closed!r}")

    if closed == "both" and num_i < 2:
        raise ValueError("closed='both' requires num >= 2.")

    if closed == "both":
        grid = np.linspace(a, b, num=num_i, endpoint=True, dtype=float)
    elif closed == "neither":
        grid = np.linspace(a, b, num=num_i + 2, endpoint=True, dtype=float)[1:-1]
    elif closed == "left":
        grid = np.linspace(a, b, num=num_i + 1, endpoint=True, dtype=float)[:-1]
    else:  # right
        grid = np.linspace(a, b, num=num_i + 1, endpoint=True, dtype=float)[1:]

    if snap_eps:
        eps = float(snap_eps)
        if not (math.isfinite(eps) and eps >= 0.0):
            raise ValueError(f"snap_eps must be finite and >= 0, got {snap_eps!r}")
        if eps > 0.0 and grid.size > 0:
            grid = grid.copy()
            grid[np.abs(grid - a) <= eps] = a
            grid[np.abs(grid - b) <= eps] = b

    grid = np.asarray(grid, dtype=dtype)
    if grid.ndim != 1 or grid.size != num_i:
        raise RuntimeError(f"Internal error: expected 1D grid of length {num_i}, got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise RuntimeError("Internal error: produced non-finite grid values.")
    if grid.size >= 2 and not np.all(np.diff(grid) > 0):
        raise RuntimeError("Internal error: grid is not strictly increasing.")

    return grid
