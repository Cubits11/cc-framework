from __future__ import annotations

"""
simulate.utils
==============

Shared utilities for the correlation_cliff simulation harness.

Primary responsibility:
- Import the local `experiments.correlation_cliff.theory` façade without hiding bugs.
- Export the required theory symbols as a stable surface for simulate/* modules.
- Provide small deterministic helpers (like lambda grid builder).

Research-OS invariants:
- Never "fix" import problems by falling back when the real error is inside the target module.
- Validate theory surface eagerly: crash early, crash loud.
"""

import logging
import math
from importlib import import_module
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Import theory without hiding bugs (and without trusting wrong shadow modules)
# -----------------------------------------------------------------------------
_REQUIRED_EXPORTS = (
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
)


def _can_fallback_to_script_import(err: ImportError, attempted: str) -> bool:
    """
    Decide whether it is safe to fall back to script import.
    We only fall back when the package import fails because the *package/module*
    couldn't be found, not because an internal dependency inside theory failed.
    """
    # ModuleNotFoundError has .name; ImportError may too.
    name = getattr(err, "name", None)

    # If the missing module is scipy/numpy/etc, do NOT fall back (that would mask a real defect).
    if isinstance(name, str) and name and (name != attempted) and not name.startswith(attempted):
        return False

    # If the missing module is the attempted theory module (or its parent), fallback is reasonable.
    return True


def _import_theory_module() -> tuple[Any, str]:
    """
    Import correlation_cliff theory façade without masking real defects.

    Strategy:
    1) Prefer explicit package import via computed parent package:
         import_module("<parent>.theory")
    2) Only if that fails because the package/module can't be found (NOT internal deps),
       fall back to `import theory` for script-like execution contexts.

    Guardrail:
    - After import, validate expected symbols exist to reduce sys.path shadowing risks.
    """
    # Compute parent package robustly.
    pkg = __package__ or ""
    parent = pkg.rsplit(".", 1)[0] if "." in pkg else ""

    # 1) Package import
    if parent:
        try:
            T = import_module(f"{parent}.theory")
            mode = "package"
        except ImportError as e:
            # Only fallback if this is genuinely "module not found" for the attempted target.
            if not _can_fallback_to_script_import(e, attempted=f"{parent}.theory"):
                raise
            T = None
            mode = "package_failed"
    else:
        T = None
        mode = "no_parent_package"

    # 2) Script fallback (only when package import context isn't available)
    if T is None:
        try:
            T = import_module("theory")
            mode = "script"
        except ImportError as e:
            raise ImportError(
                "Could not import correlation_cliff theory module.\n"
                "Recommended invocation from repo root:\n"
                "  python -m experiments.correlation_cliff.simulate --config path/to/config.yaml\n"
                "Or install the package so relative imports work."
            ) from e

    # Validate surface
    missing = [name for name in _REQUIRED_EXPORTS if not hasattr(T, name)]
    if missing:
        origin = getattr(T, "__file__", "<unknown>")
        raise ImportError(
            "Imported a module named 'theory' but it does not look like the "
            "correlation_cliff theory façade.\n"
            f"Import mode: {mode}\n"
            f"Module origin: {origin}\n"
            f"Missing expected exports: {missing}\n"
            "This usually means sys.path shadowing (imported the wrong module), "
            "or your correlation_cliff/theory.py is incomplete."
        )

    # Optional: extra guard against bizarre shadow modules
    origin = getattr(T, "__file__", "") or ""
    if mode == "script" and origin and ("correlation_cliff" not in origin.replace("\\", "/")):
        LOG.warning(
            "simulate.utils imported 'theory' in script mode from unexpected location: %s", origin
        )

    LOG.debug(
        "Imported theory module in %s mode from %s", mode, getattr(T, "__file__", "<unknown>")
    )
    return T, mode


T, THEORY_IMPORT_MODE = _import_theory_module()

# Bind required symbols explicitly (crash early if theory surface changes)
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

# Optional: reference overlay helper (never assumed path-consistent)
compute_metrics_for_lambda: Optional[Callable[..., Dict[str, float]]] = getattr(
    T, "compute_metrics_for_lambda", None
)


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
    - Strict typing: bool is rejected for num.
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
    if isinstance(num, bool):
        raise TypeError(f"num must be an int, got bool {num!r}")
    if isinstance(num, (float, np.floating)):
        raise TypeError(f"num must be an integer (no silent coercion), got {num!r}")
    try:
        num_i = int(num)
    except Exception as e:
        raise TypeError(f"num must be an int, got {num!r}") from e
    if num_i < 1:
        raise ValueError(f"num must be >= 1, got {num_i}")
    if num_i != num:
        raise TypeError(f"num must be an integer (no silent coercion), got {num!r}")

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

    eps = float(snap_eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError(f"snap_eps must be finite and >= 0, got {snap_eps!r}")
    if eps > 0.0 and grid.size > 0:
        grid = grid.copy()
        grid[np.abs(grid - a) <= eps] = a
        grid[np.abs(grid - b) <= eps] = b

    grid = np.asarray(grid, dtype=dtype)
    if grid.ndim != 1 or grid.size != num_i:
        raise RuntimeError(
            f"Internal error: expected 1D grid of length {num_i}, got shape {grid.shape}"
        )
    if not np.all(np.isfinite(grid)):
        raise RuntimeError("Internal error: produced non-finite grid values.")
    if grid.size >= 2 and not np.all(np.diff(grid) > 0):
        raise RuntimeError("Internal error: grid is not strictly increasing.")

    return grid


__all__ = [
    "THEORY_IMPORT_MODE",
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
    "compute_metrics_for_lambda",
    "build_linear_lambda_grid",
]
