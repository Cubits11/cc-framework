# experiments/correlation_cliff/simulate.py
from __future__ import annotations

"""
Correlation Cliff — Simulation Module (research-grade harness)
=============================================================

This module generates *finite-sample* empirical estimates for the correlation-cliff
experiment under a *fixed-marginals, variable-dependence* model.

Design principles (non-negotiable)
----------------------------------
1) No silent import fallbacks:
   - We only fall back to script-style imports on ImportError.
   - Any other exception is a real bug and must surface.

2) No silent probability “fixing”:
   - We do NOT silently renormalize multinomial probabilities by default.
   - If a joint is invalid, we raise (default) or flag explicitly with diagnostics.

3) Reproducibility that survives refactors:
   - Default RNG policy is *order-invariant*: results do not change if you reorder lambdas.
   - Seed streams are keyed by (seed, replicate, lambda_index, world).

4) Honest overlays:
   - “Population overlay” columns (CC_pop, JC_pop, …) match the *configured simulation path*.
   - Optional reference overlays from theory (CC_theory_ref, …) are stored separately and
     mismatch diagnostics are recorded. You should never compare mismatched overlays
     without explicitly stating assumptions.

Scope
-----
This file is an experiment harness: construct joint → sample → compute estimates → log diagnostics.
Dependence-path math is isolated, and the long-term end state is to move all dependence primitives
into theory_core.py (single source of truth).

Dependencies
------------
numpy, pandas. Optional:
- scipy for gaussian_tau path
- pyyaml for CLI config loading

"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

Rule = Literal["OR", "AND"]
Path = Literal["fh_linear", "fh_power", "fh_scurve", "gaussian_tau"]
SeedPolicy = Literal["sequential", "stable_per_cell"]


def _rng_for_cell(seed: int, rep: int, lam_index: int, world: int) -> np.random.Generator:
    """
    Stable-per-cell RNG keyed by (seed, rep, lambda_index, world).
    """
    if not all(isinstance(x, int) for x in (seed, rep, lam_index, world)):
        raise TypeError("seed/rep/lam_index/world must all be ints.")
    ss = np.random.SeedSequence([seed, rep, lam_index, world])
    return np.random.default_rng(ss)


# -----------------------------------------------------------------------------
# Import theory without hiding bugs (and without trusting a wrong shadow module)
# -----------------------------------------------------------------------------
def _import_theory_module():
    """
    Import the local correlation_cliff theory module without hiding real defects.

    Strategy:
    - Prefer package-relative import: from . import theory
    - If and only if that fails with ImportError, fall back to script import: import theory

    Guardrail:
    - After import, we validate that expected symbols exist. This reduces the chance of
      accidentally importing an unrelated `theory` module due to sys.path shadowing.
    """
    # 1) Prefer package-relative import
    try:
        from . import theory as T  # type: ignore

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

    # Validate exports to avoid wrong-module surprises.
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

    LOG.debug(
        "Imported theory module in %s mode from %s", mode, getattr(T, "__file__", "<unknown>")
    )
    return T, mode


T, _THEORY_IMPORT_MODE = _import_theory_module()

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

# Optional: may or may not exist in the current theory façade.
# This is treated as a *reference overlay* only (never silently assumed path-consistent).
_compute_metrics_for_lambda: Optional[Callable[..., Dict[str, float]]] = getattr(
    T, "compute_metrics_for_lambda", None
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SimConfig:
    """
    Simulation configuration (research-grade, refactor-stable).

    Key guarantees
    --------------
    - Fail-fast validation with explicit error messages (no silent fixing).
    - Reproducibility contract:
        * stable_per_cell: order-invariant streams keyed by (seed, rep, lambda_index, world)
          where lambda_index is defined on a *canonical* lambda grid (sorted unique lambdas).
        * sequential: single RNG stream; results depend on loop order (explicitly allowed).
    - Honest overlays:
        * Path-consistent population overlays should be computed from this cfg.path.
        * Optional theory reference overlays are tagged separately.

    Parameters
    ----------
    marginals:
        TwoWorldMarginals with fixed pA/pB per world.
    rule:
        "OR" or "AND".
    lambdas:
        Dependence grid in [0,1]. Must be finite and unique (duplicates are ambiguous).
    n:
        Sample size per world per lambda per replicate.
    n_reps:
        Number of Monte Carlo replicates per lambda.
    seed:
        Base seed (integer). Used to derive independent streams under stable_per_cell.

    path:
        Dependence path name.
    path_params:
        Path-specific parameters (validated lightly here; deep validation belongs to path logic).

    seed_policy:
        - "stable_per_cell" (default): RNG keyed by (seed, rep, lambda_index, world). Order-invariant.
        - "sequential": single RNG stream; results depend on loop order.

    envelope_tol:
        Nonnegative tolerance for flagging empirical JC_hat outside FH envelope [jmin, jmax].

    hard_fail_on_invalid:
        If True, raise on invalid joint construction; else mark invalid and fill NaNs.

    prob_tol:
        Tolerance for probability sum-to-one checks (>=0).

    allow_tiny_negative / tiny_negative_eps:
        If allow_tiny_negative, clamp probabilities in [-tiny_negative_eps, 0) to 0.
        NOTE: this is a *diagnostic convenience*, not a license to ignore invalid constructions.

    include_theory_reference:
        If True and theory exposes compute_metrics_for_lambda, add *_theory_ref overlays.
        These are *reference overlays* and may not match the simulation path.
    """

    marginals: TwoWorldMarginals
    rule: Rule
    lambdas: Sequence[float]
    n: int
    n_reps: int = 1
    seed: int = 0

    path: Path = "fh_linear"
    path_params: Dict[str, Any] = field(default_factory=dict)

    seed_policy: SeedPolicy = "stable_per_cell"

    envelope_tol: float = 5e-3
    hard_fail_on_invalid: bool = True

    prob_tol: float = 1e-12
    allow_tiny_negative: bool = True
    tiny_negative_eps: float = 1e-15

    include_theory_reference: bool = True

    # ---- computed / normalized (do not pass at init) ----
    lambdas_canonical: Tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # ---- normalize + validate rule ----
        rr = str(self.rule).upper()
        if rr not in ("OR", "AND"):
            raise ValueError(f"rule must be 'OR' or 'AND', got {self.rule!r}")
        object.__setattr__(self, "rule", rr)  # type: ignore

        # ---- validate basic integers ----
        try:
            n = int(self.n)
        except Exception as e:
            raise ValueError(f"n must be an int, got {self.n!r}") from e
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        object.__setattr__(self, "n", n)

        try:
            n_reps = int(self.n_reps)
        except Exception as e:
            raise ValueError(f"n_reps must be an int, got {self.n_reps!r}") from e
        if n_reps <= 0:
            raise ValueError(f"n_reps must be positive, got {n_reps}")
        object.__setattr__(self, "n_reps", n_reps)

        try:
            seed = int(self.seed)
        except Exception as e:
            raise ValueError(f"seed must be an int, got {self.seed!r}") from e
        object.__setattr__(self, "seed", seed)

        # ---- validate policy-ish floats ----
        tol = float(self.envelope_tol)
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError(f"envelope_tol must be finite and >= 0, got {self.envelope_tol!r}")
        object.__setattr__(self, "envelope_tol", tol)

        prob_tol = float(self.prob_tol)
        if prob_tol < 0.0 or not math.isfinite(prob_tol):
            raise ValueError(f"prob_tol must be finite and >= 0, got {self.prob_tol!r}")
        object.__setattr__(self, "prob_tol", prob_tol)

        tiny_eps = float(self.tiny_negative_eps)
        if tiny_eps < 0.0 or not math.isfinite(tiny_eps):
            raise ValueError(
                f"tiny_negative_eps must be finite and >= 0, got {self.tiny_negative_eps!r}"
            )
        object.__setattr__(self, "tiny_negative_eps", tiny_eps)

        # ---- validate path + params container ----
        pp = str(self.path)
        if pp not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
            raise ValueError(
                f"path must be one of fh_linear/fh_power/fh_scurve/gaussian_tau, got {self.path!r}"
            )
        object.__setattr__(self, "path", pp)  # type: ignore

        if self.path_params is None:
            object.__setattr__(self, "path_params", {})
        elif not isinstance(self.path_params, dict):
            raise ValueError(f"path_params must be a dict, got {type(self.path_params).__name__}")

        sp = str(self.seed_policy)
        if sp not in ("stable_per_cell", "sequential"):
            raise ValueError(
                f"seed_policy must be 'stable_per_cell' or 'sequential', got {self.seed_policy!r}"
            )
        object.__setattr__(self, "seed_policy", sp)  # type: ignore

        # ---- normalize + validate lambdas ----
        if self.lambdas is None:
            raise ValueError("lambdas must be a non-empty sequence of floats in [0,1].")

        try:
            lambdas_tuple = tuple(float(x) for x in self.lambdas)
        except Exception as e:
            raise ValueError("lambdas must be a sequence of numbers convertible to float.") from e

        if len(lambdas_tuple) == 0:
            raise ValueError("lambdas must be non-empty.")

        for i, lam in enumerate(lambdas_tuple):
            if not math.isfinite(lam):
                raise ValueError(f"lambda[{i}] is not finite: {lam!r}")
            if not (0.0 <= lam <= 1.0):
                raise ValueError(f"lambda[{i}] must be in [0,1], got {lam}")

        # Enforce uniqueness (duplicates make seeding + grouping ambiguous).
        # If you truly need duplicates for debugging, create distinct grids explicitly.
        if len(set(lambdas_tuple)) != len(lambdas_tuple):
            raise ValueError(
                "lambdas contains duplicates. This is ambiguous for seeding/grouping.\n"
                "Make lambdas unique (or explicitly dedupe before constructing SimConfig)."
            )

        object.__setattr__(self, "lambdas", lambdas_tuple)  # type: ignore

        # Canonical lambda order defines lambda_index for stable_per_cell seeding (order-invariant).
        lambdas_canon = tuple(sorted(lambdas_tuple))
        object.__setattr__(self, "lambdas_canonical", lambdas_canon)

    # ---- small utilities used by later segments (kept here to enforce the contract) ----
    def lambda_index_for_seed(self, lam: float, *, tol: float = 0.0) -> int:
        """
        Map a lambda value to its *canonical* index (used for stable_per_cell seeding).

        Exact match is required unless tol > 0, in which case the nearest canonical lambda
        within tol is accepted.
        """
        v = float(lam)
        try:
            return self.lambdas_canonical.index(v)
        except ValueError:
            if tol <= 0.0:
                raise KeyError(
                    f"lambda={v} not found in canonical grid. "
                    "Pass a lambda from cfg.lambdas (or use tol if appropriate)."
                )
            # nearest-within-tol fallback (explicitly opt-in)
            best_i = -1
            best_d = float("inf")
            for i, u in enumerate(self.lambdas_canonical):
                d = abs(u - v)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_d <= float(tol) and best_i >= 0:
                return int(best_i)
            raise KeyError(
                f"lambda={v} not found in canonical grid within tol={tol}. "
                f"Nearest distance was {best_d}."
            )


def _validate_cfg(cfg: SimConfig) -> None:
    """
    Enterprise-grade config validation.

    Philosophy:
    - No silent fixing.
    - Fail fast with actionable errors.
    - Validate cross-field contracts (path_params vs path, gaussian ppf stability, etc.).

    NOTE: SimConfig.__post_init__ already normalizes many fields. This function is an
    extra guardrail and should remain *pure validation* (no mutation).
    """
    # -----------------------------
    # Basic type / enum validation
    # -----------------------------
    if cfg.rule not in ("OR", "AND"):
        raise ValueError(f"Invalid rule: {cfg.rule!r} (expected 'OR' or 'AND').")

    if cfg.path not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
        raise ValueError(
            f"Invalid path: {cfg.path!r} "
            "(expected 'fh_linear', 'fh_power', 'fh_scurve', or 'gaussian_tau')."
        )

    if cfg.seed_policy not in ("sequential", "stable_per_cell"):
        raise ValueError(
            f"Invalid seed_policy: {cfg.seed_policy!r} "
            "(expected 'sequential' or 'stable_per_cell')."
        )

    # -----------------------------
    # Numeric sanity
    # -----------------------------
    if not isinstance(cfg.n, int):
        raise ValueError(f"n must be an int, got {type(cfg.n).__name__}.")
    if cfg.n <= 0:
        raise ValueError(f"n must be positive, got {cfg.n}.")

    if not isinstance(cfg.n_reps, int):
        raise ValueError(f"n_reps must be an int, got {type(cfg.n_reps).__name__}.")
    if cfg.n_reps <= 0:
        raise ValueError(f"n_reps must be positive, got {cfg.n_reps}.")

    if not isinstance(cfg.seed, int):
        raise ValueError(f"seed must be an int, got {type(cfg.seed).__name__}.")

    # -----------------------------
    # Lambda grid validation
    # -----------------------------
    if cfg.lambdas is None or len(cfg.lambdas) < 1:
        raise ValueError("lambdas must be non-empty.")

    lambdas = [float(x) for x in cfg.lambdas]
    for i, lam in enumerate(lambdas):
        if not math.isfinite(lam):
            raise ValueError(f"lambda[{i}] is not finite: {lam!r}")
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lambda[{i}] must be in [0,1], got {lam}")

    if len(set(lambdas)) != len(lambdas):
        raise ValueError(
            "lambdas contains duplicates. This is ambiguous for seeding/grouping.\n"
            "Make lambdas unique."
        )

    # If SimConfig provides canonical lambdas, verify consistency.
    if hasattr(cfg, "lambdas_canonical"):
        canon = list(getattr(cfg, "lambdas_canonical"))
        if canon != sorted(lambdas):
            raise ValueError(
                "cfg.lambdas_canonical does not match sorted(cfg.lambdas). "
                "This breaks stable_per_cell seeding invariants."
            )

    # -----------------------------
    # Probability / tolerance policies
    # -----------------------------
    if not (isinstance(cfg.prob_tol, float) or isinstance(cfg.prob_tol, int)):
        raise ValueError(f"prob_tol must be a float, got {type(cfg.prob_tol).__name__}.")
    if not math.isfinite(float(cfg.prob_tol)) or float(cfg.prob_tol) < 0.0:
        raise ValueError(f"prob_tol must be finite and >= 0, got {cfg.prob_tol!r}.")
    # Upper bound is a *policy* check: too-large tolerances hide real bugs.
    if float(cfg.prob_tol) > 1e-6:
        raise ValueError(
            f"prob_tol={cfg.prob_tol} seems unreasonably large. "
            "Recommended: 1e-12 to 1e-9; hard max: 1e-6."
        )

    if not (isinstance(cfg.envelope_tol, float) or isinstance(cfg.envelope_tol, int)):
        raise ValueError(f"envelope_tol must be a float, got {type(cfg.envelope_tol).__name__}.")
    if not math.isfinite(float(cfg.envelope_tol)) or float(cfg.envelope_tol) < 0.0:
        raise ValueError(f"envelope_tol must be finite and >= 0, got {cfg.envelope_tol!r}.")
    if float(cfg.envelope_tol) > 0.1:
        raise ValueError(
            f"envelope_tol={cfg.envelope_tol} is suspiciously large. "
            "Envelope is a feasibility guardrail; large tol defeats the check."
        )

    if cfg.allow_tiny_negative:
        if not (isinstance(cfg.tiny_negative_eps, float) or isinstance(cfg.tiny_negative_eps, int)):
            raise ValueError(
                f"tiny_negative_eps must be a float, got {type(cfg.tiny_negative_eps).__name__}."
            )
        eps = float(cfg.tiny_negative_eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                f"tiny_negative_eps must be finite and > 0, got {cfg.tiny_negative_eps!r}."
            )
        if eps > 1e-6:
            raise ValueError(
                f"tiny_negative_eps={eps} seems unreasonably large. "
                "This feature is meant for floating-point dust only (<=1e-6, ideally <=1e-12)."
            )
        # If you allow tiny negatives, it should not exceed your general prob tolerance by orders of magnitude.
        if eps > max(1e3 * float(cfg.prob_tol), 1e-12):
            raise ValueError(
                f"tiny_negative_eps={eps} is too large relative to prob_tol={cfg.prob_tol}. "
                "This risks masking genuinely invalid joints."
            )

    # -----------------------------
    # Marginals sanity (+ FH feasibility)
    # -----------------------------
    def _vp(v: float, name: str) -> float:
        vv = float(v)
        if not math.isfinite(vv):
            raise ValueError(f"{name} is not finite: {vv!r}")
        if not (0.0 <= vv <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {vv}")
        return vv

    for w, wm in (("w0", cfg.marginals.w0), ("w1", cfg.marginals.w1)):
        pA = _vp(wm.pA, f"marginals.{w}.pA")
        pB = _vp(wm.pB, f"marginals.{w}.pB")

        # FH bounds sanity (must exist and be ordered)
        b = fh_bounds(pA, pB)
        L = float(b.L)
        U = float(b.U)
        if not (math.isfinite(L) and math.isfinite(U)):
            raise ValueError(f"FH bounds not finite for {w}: L={L}, U={U}")
        if L > U + float(cfg.prob_tol):
            raise ValueError(f"FH bounds inverted for {w}: L={L} > U={U} (prob_tol={cfg.prob_tol})")

    # Jbest must be > 0 to define CC
    JA = abs(float(cfg.marginals.w1.pA) - float(cfg.marginals.w0.pA))
    JB = abs(float(cfg.marginals.w1.pB) - float(cfg.marginals.w0.pB))
    if max(JA, JB) <= 0.0:
        raise ValueError(
            "Degenerate setup: Jbest = max(|ΔA|,|ΔB|) is 0, so CC is undefined.\n"
            "Choose marginals with at least one nonzero delta between worlds."
        )

    # -----------------------------
    # Path-parameter contracts
    # -----------------------------
    pp = cfg.path_params or {}

    if cfg.path == "fh_power":
        gamma = float(pp.get("gamma", 1.0))
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError(f"fh_power requires gamma > 0, got {gamma!r} in path_params.")

    if cfg.path == "fh_scurve":
        k = float(pp.get("k", 8.0))
        if not math.isfinite(k) or k <= 0.0:
            raise ValueError(f"fh_scurve requires k > 0, got {k!r} in path_params.")

    if cfg.path == "gaussian_tau":
        # Gaussian copula uses norm.ppf(p); p in (0,1) is required.
        # If any pA/pB is too close to 0/1, require ppf_clip_eps (or ensure your code clips).
        ppf_clip_eps = pp.get("ppf_clip_eps", None)
        if ppf_clip_eps is None:
            # Still allow if marginals are comfortably away from boundaries.
            # (This keeps config ergonomic but still safe.)
            eps_needed = False
            for wm in (cfg.marginals.w0, cfg.marginals.w1):
                for v in (float(wm.pA), float(wm.pB)):
                    if v <= 1e-12 or v >= 1.0 - 1e-12:
                        eps_needed = True
            if eps_needed:
                raise ValueError(
                    "gaussian_tau path with marginals extremely close to 0 or 1 requires "
                    "path_params['ppf_clip_eps'] to avoid ±∞ in norm.ppf."
                )
        else:
            eps = float(ppf_clip_eps)
            if not math.isfinite(eps) or not (0.0 < eps < 0.5):
                raise ValueError(
                    f"ppf_clip_eps must be finite and in (0,0.5), got {ppf_clip_eps!r}."
                )


# -----------------------------------------------------------------------------
# Helpers: lambda transforms / paths
# -----------------------------------------------------------------------------
def _lam_power(lam: float, gamma: float) -> float:
    """
    Monotone power transform of lambda in [0,1].

    Purpose:
      - Provide a *shape control* for traversing the FH interval without leaving feasibility.
      - gamma > 1 slows early movement (more "flat" near 0).
      - 0 < gamma < 1 accelerates early movement (more "steep" near 0).

    Guarantees (for valid inputs):
      - maps [0,1] -> [0,1]
      - preserves endpoints: f(0)=0, f(1)=1
      - monotone increasing

    Notes:
      - We validate finiteness because NaNs/inf will quietly poison downstream tables.
      - We do not clip lam here: caller is responsible for lambda contract checks.
    """
    lam = float(lam)
    gamma = float(gamma)

    if not math.isfinite(lam):
        raise ValueError(f"lam must be finite, got {lam!r}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0,1], got {lam}")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and > 0, got {gamma!r}")

    # Fast path for common cases and exact endpoints.
    if lam == 0.0 or lam == 1.0:
        return lam
    if gamma == 1.0:
        return lam

    y = lam**gamma

    # Defensive: numeric noise should never push outside [0,1], but floating can be weird.
    if y < 0.0:
        y = 0.0
    elif y > 1.0:
        y = 1.0
    return float(y)


def _lam_scurve(lam: float, k: float) -> float:
    """
    Smooth S-curve (sigmoid-based) transform on [0,1] with *exact* endpoint preservation.

    Motivation:
      - A "soft" nonlinear traversal of the FH interval that changes sensitivity mid-range,
        while staying within feasibility via later affine mixing with [L,U].
      - Useful for stress-testing "where dependence changes fastest" without changing
        the FH bounds themselves.

    Definition:
      s_raw(lam) = sigmoid(k*(lam - 0.5))
      a = sigmoid(-0.5*k) = s_raw(0)
      b = sigmoid(+0.5*k) = s_raw(1)
      s(lam) = (s_raw(lam) - a) / (b - a)

    Guarantees (for valid inputs):
      - maps [0,1] -> [0,1]
      - preserves endpoints exactly: s(0)=0, s(1)=1 (up to floating roundoff)
      - monotone increasing
      - smooth (C-infinity)

    Numerical notes:
      - For very large k, exp() can overflow; we use a stable sigmoid implementation.
      - If k is tiny, b≈a and the rescale becomes ill-conditioned; we fall back to identity.
    """
    lam = float(lam)
    k = float(k)

    if not math.isfinite(lam):
        raise ValueError(f"lam must be finite, got {lam!r}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0,1], got {lam}")
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"k must be finite and > 0, got {k!r}")

    # Stable sigmoid: avoids overflow for large |z|
    def _sigmoid(z: float) -> float:
        z = float(z)
        if z >= 0.0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    # Exact endpoints are worth preserving to avoid drift in grids / root finding.
    if lam == 0.0:
        return 0.0
    if lam == 1.0:
        return 1.0

    # If k is extremely small, sigmoid is almost linear and rescaling gets numerically ugly.
    # This threshold is conservative; tweak if you want to allow super-smooth curves.
    if k < 1e-8:
        return lam

    a = _sigmoid(-0.5 * k)
    b = _sigmoid(+0.5 * k)

    denom = b - a
    if denom <= 0.0 or not math.isfinite(denom):
        # Shouldn't happen for valid k, but guard anyway.
        return lam

    s_raw = _sigmoid(k * (lam - 0.5))
    s = (s_raw - a) / denom

    # Clamp for microscopic numerical overshoots
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0

    return float(s)


def _bvn_cdf_scipy(x: float, y: float, rho: float) -> float:
    """
    Bivariate standard normal CDF: P(Z1 <= x, Z2 <= y) for Corr(Z1,Z2)=rho.

    Design:
      - SciPy is an *optional* dependency for the gaussian_tau path.
      - We raise ImportError if SciPy is unavailable (no silent fallbacks).
      - We validate rho to avoid passing invalid covariance to SciPy.

    Notes:
      - multivariate_normal.cdf uses numerical integration; it can be slow for large grids.
        That’s acceptable here because gaussian_tau is a sensitivity path, not the default.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    ValueError
        If rho is not finite or not in [-1, 1].
    """
    x = float(x)
    y = float(y)
    rho = float(rho)

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"x and y must be finite, got x={x!r}, y={y!r}")
    if not math.isfinite(rho) or not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be finite and in [-1,1], got {rho!r}")

    try:
        from scipy.stats import multivariate_normal  # type: ignore
    except ImportError as e:
        raise ImportError("SciPy not available for gaussian_tau path") from e

    mean = np.array([0.0, 0.0], dtype=float)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)

    # SciPy may throw if cov is not PSD due to tiny numerical issues near |rho|=1.
    # We'll let SciPy throw in that case, because the caller must not use rho outside [-1,1].
    val = float(multivariate_normal(mean=mean, cov=cov, allow_singular=True).cdf([x, y]))

    # Defensive: tiny out-of-range due to numerical integration noise.
    if val < 0.0:
        val = 0.0
    elif val > 1.0:
        val = 1.0
    return val


def _p11_gaussian_tau(
    pA: float,
    pB: float,
    tau: float,
    *,
    ppf_clip_eps: float = 1e-12,
) -> Tuple[float, Dict[str, float]]:
    """
    Gaussian copula overlap: p11 = C(u,v) where u=pA, v=pB and concordance is set by Kendall tau.

    Mapping:
      Gaussian copula has: tau = (2/pi) * arcsin(rho)  =>  rho = sin(pi*tau/2)
      Then: p11 = Phi2(Phi^{-1}(u), Phi^{-1}(v); rho)

    Why this exists:
      - Provides a *smooth* dependence family for sensitivity analysis.
      - IMPORTANT: For discrete margins, no copula family spans the full FH range.
        So this is a *modeling choice* (assumption), not an identifiability claim.

    Inputs
    ------
    pA, pB:
        Marginal probabilities in [0,1].
    tau:
        Kendall tau in [-1,1].
    ppf_clip_eps:
        Clip u,v into [eps, 1-eps] before applying norm.ppf to avoid ±∞.
        This does not “fix” probability validity; it only avoids undefined transforms.

    Returns
    -------
    (p11, meta)
      meta includes:
        - tau
        - rho
        - pA_used, pB_used (after clipping)
        - ppf_clip_eps

    Raises
    ------
    ImportError
        If SciPy is not installed.
    ValueError
        For invalid inputs.
    """
    pA = float(pA)
    pB = float(pB)
    tau = float(tau)
    ppf_clip_eps = float(ppf_clip_eps)

    if not (0.0 <= pA <= 1.0) or not math.isfinite(pA):
        raise ValueError(f"pA must be finite and in [0,1], got {pA!r}")
    if not (0.0 <= pB <= 1.0) or not math.isfinite(pB):
        raise ValueError(f"pB must be finite and in [0,1], got {pB!r}")
    if not math.isfinite(tau) or not (-1.0 <= tau <= 1.0):
        raise ValueError(f"tau must be finite and in [-1,1], got {tau!r}")
    if not math.isfinite(ppf_clip_eps) or not (0.0 < ppf_clip_eps < 0.5):
        raise ValueError(f"ppf_clip_eps must be in (0, 0.5), got {ppf_clip_eps!r}")

    # Convert tau -> rho for Gaussian copula
    rho = math.sin(math.pi * tau / 2.0)

    try:
        from scipy.stats import norm  # type: ignore
    except ImportError as e:
        raise ImportError("SciPy not available for gaussian_tau path") from e

    # Avoid ±∞ from norm.ppf at 0/1
    pA_used = min(max(pA, ppf_clip_eps), 1.0 - ppf_clip_eps)
    pB_used = min(max(pB, ppf_clip_eps), 1.0 - ppf_clip_eps)

    x = float(norm.ppf(pA_used))
    y = float(norm.ppf(pB_used))

    p11 = _bvn_cdf_scipy(x, y, rho)

    meta = {
        "tau": float(tau),
        "rho": float(rho),
        "pA_used": float(pA_used),
        "pB_used": float(pB_used),
        "ppf_clip_eps": float(ppf_clip_eps),
    }
    return float(p11), meta


def p11_from_path(
    pA: float,
    pB: float,
    lam: float,
    *,
    path: Path,
    path_params: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    V3: Research/enterprise-grade p11 constructor with explicit invariants and audit fields.

    Returns (p11, meta) where meta is NUMERIC-ONLY by design (downstream-safe).
    Required meta keys for ALL paths:
      - L, U, FH_width, lam, lam_eff
      - raw_p11, clip_amt, clipped   (clipped is 0.0/1.0)
    gaussian_tau adds:
      - tau, rho
      - ppf_clip_eps, pA_used, pB_used, pA_clip, pB_clip

    clip_policy (path_params):
      - "clip" (default): enforce p11 ∈ [L,U] and record clip_amt/clipped
      - "raise": if raw_p11 violates FH by > clip_tol, raise FeasibilityError
    """

    # ----------------------------
    # local exceptions (keep lightweight, but structured)
    # ----------------------------
    class P11FromPathError(ValueError):
        pass

    class InputValidationError(P11FromPathError):
        pass

    class FeasibilityError(P11FromPathError):
        pass

    class NumericalError(P11FromPathError):
        pass

    # ----------------------------
    # sanitize inputs
    # ----------------------------
    try:
        pA = float(pA)
        pB = float(pB)
        lam = float(lam)
    except Exception as e:
        raise InputValidationError(
            f"pA, pB, lam must be coercible to float. Got pA={pA!r}, pB={pB!r}, lam={lam!r}"
        ) from e

    if not math.isfinite(pA) or not (0.0 <= pA <= 1.0):
        raise InputValidationError(f"pA must be finite and in [0,1], got {pA!r}")
    if not math.isfinite(pB) or not (0.0 <= pB <= 1.0):
        raise InputValidationError(f"pB must be finite and in [0,1], got {pB!r}")
    if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
        raise InputValidationError(f"lam must be finite and in [0,1], got {lam!r}")

    if not isinstance(path_params, dict):
        # keep dict requirement to match your current typing + simplicity
        raise InputValidationError(f"path_params must be a dict, got {type(path_params).__name__}")

    # ----------------------------
    # FH bounds (always)
    # ----------------------------
    b = fh_bounds(pA, pB)
    L = float(b.L)
    U = float(b.U)

    if not (math.isfinite(L) and math.isfinite(U) and 0.0 <= L <= U <= 1.0):
        raise NumericalError(f"FH bounds invalid for pA={pA}, pB={pB}: L={L}, U={U}")

    width = float(U - L)
    meta: Dict[str, float] = {
        "L": L,
        "U": U,
        "FH_width": width,
        "lam": float(lam),
        # filled later:
        "lam_eff": float("nan"),
        "raw_p11": float("nan"),
        "clip_amt": 0.0,
        "clipped": 0.0,
    }

    # ----------------------------
    # clip policy (explicit)
    # ----------------------------
    clip_policy = str(path_params.get("clip_policy", "clip")).lower().strip()
    if clip_policy not in ("clip", "raise"):
        raise InputValidationError(f"clip_policy must be 'clip' or 'raise', got {clip_policy!r}")

    clip_tol = float(path_params.get("clip_tol", 0.0))
    if not math.isfinite(clip_tol) or clip_tol < 0.0 or clip_tol > 1e-6:
        raise InputValidationError(
            f"clip_tol must be finite, >=0, and small (<=1e-6). got {clip_tol!r}"
        )

    def _finalize(raw: float, lam_eff: float) -> Tuple[float, Dict[str, float]]:
        if not math.isfinite(raw):
            raise NumericalError(f"raw_p11 is non-finite: {raw!r}")

        meta["lam_eff"] = float(lam_eff)
        meta["raw_p11"] = float(raw)

        # enforce feasibility with explicit policy
        if raw < L - clip_tol or raw > U + clip_tol:
            if clip_policy == "raise":
                raise FeasibilityError(
                    f"{path}: raw_p11 violates FH bounds by more than clip_tol. "
                    f"raw={raw}, L={L}, U={U}, clip_tol={clip_tol}"
                )

        clipped = raw
        if clipped < L:
            clipped = L
        elif clipped > U:
            clipped = U

        meta["clip_amt"] = float(raw - clipped)  # signed
        meta["clipped"] = float(1.0 if clipped != raw else 0.0)

        # postcondition: always within [L,U]
        if not (L <= clipped <= U):
            raise NumericalError(
                f"Postcondition failed: p11 not in [L,U]. p11={clipped}, L={L}, U={U}"
            )

        return float(clipped), meta

    # ----------------------------
    # FH paths (guaranteed feasibility by construction)
    # ----------------------------
    if path == "fh_linear":
        raw = float(p11_fh_linear(pA, pB, lam))
        return _finalize(raw, lam_eff=lam)

    if path == "fh_power":
        gamma_raw = path_params.get("gamma", 1.0)
        gamma = float(gamma_raw)
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise InputValidationError(f"fh_power requires gamma > 0, got {gamma_raw!r}")
        lam_eff = _lam_power(lam, gamma)
        if not (0.0 <= lam_eff <= 1.0) or not math.isfinite(lam_eff):
            raise NumericalError(
                f"fh_power produced invalid lam_eff={lam_eff} for lam={lam}, gamma={gamma}"
            )
        raw = L + lam_eff * width
        meta["gamma"] = float(gamma)  # numeric-only metadata
        return _finalize(float(raw), lam_eff=float(lam_eff))

    if path == "fh_scurve":
        k_raw = path_params.get("k", 8.0)
        k = float(k_raw)
        if not math.isfinite(k) or k <= 0.0:
            raise InputValidationError(f"fh_scurve requires k > 0, got {k_raw!r}")
        lam_eff = _lam_scurve(lam, k)
        if not (0.0 <= lam_eff <= 1.0) or not math.isfinite(lam_eff):
            raise NumericalError(
                f"fh_scurve produced invalid lam_eff={lam_eff} for lam={lam}, k={k}"
            )
        raw = L + lam_eff * width
        meta["k"] = float(k)
        return _finalize(float(raw), lam_eff=float(lam_eff))

    # ----------------------------
    # Gaussian copula path (assumption-driven; may not span FH extremes)
    # ----------------------------
    if path == "gaussian_tau":
        tau = 2.0 * lam - 1.0
        if not (-1.0 <= tau <= 1.0) or not math.isfinite(tau):
            raise InputValidationError(f"tau derived from lam must be in [-1,1], got {tau!r}")

        # Degenerate marginals: exact solution, no SciPy, no PPF nonsense.
        # If pA or pB is 0/1, the overlap is fully determined.
        if pA in (0.0, 1.0) or pB in (0.0, 1.0):
            raw = min(pA, pB)
            # Also record tau/rho for completeness
            rho = math.sin(math.pi * tau / 2.0)
            meta.update({"tau": float(tau), "rho": float(rho)})
            meta.update(
                {
                    "ppf_clip_eps": 0.0,
                    "pA_used": float(pA),
                    "pB_used": float(pB),
                    "pA_clip": 0.0,
                    "pB_clip": 0.0,
                }
            )
            return _finalize(float(raw), lam_eff=lam)

        # tau extremes: avoid unstable BVN calls.
        # For Gaussian copula: tau=+1 => comonotone => p11 = U; tau=-1 => countermonotone => p11 = L
        tau_ext_tol = float(path_params.get("tau_extreme_tol", 1e-12))
        if not math.isfinite(tau_ext_tol) or tau_ext_tol < 0.0 or tau_ext_tol > 1e-6:
            raise InputValidationError(
                f"tau_extreme_tol must be finite and small (<=1e-6), got {tau_ext_tol!r}"
            )

        if tau >= 1.0 - tau_ext_tol:
            rho = 1.0
            meta.update({"tau": float(tau), "rho": float(rho)})
            meta.update(
                {
                    "ppf_clip_eps": 0.0,
                    "pA_used": float(pA),
                    "pB_used": float(pB),
                    "pA_clip": 0.0,
                    "pB_clip": 0.0,
                }
            )
            return _finalize(float(U), lam_eff=lam)

        if tau <= -1.0 + tau_ext_tol:
            rho = -1.0
            meta.update({"tau": float(tau), "rho": float(rho)})
            meta.update(
                {
                    "ppf_clip_eps": 0.0,
                    "pA_used": float(pA),
                    "pB_used": float(pB),
                    "pA_clip": 0.0,
                    "pB_clip": 0.0,
                }
            )
            return _finalize(float(L), lam_eff=lam)

        eps_raw = path_params.get("ppf_clip_eps", 1e-12)
        eps = float(eps_raw)
        if not math.isfinite(eps) or not (0.0 < eps < 0.5):
            raise InputValidationError(f"ppf_clip_eps must be in (0,0.5), got {eps_raw!r}")

        pA_eff = min(max(pA, eps), 1.0 - eps)
        pB_eff = min(max(pB, eps), 1.0 - eps)

        meta.update(
            {
                "ppf_clip_eps": float(eps),
                "pA_used": float(pA_eff),
                "pB_used": float(pB_eff),
                "pA_clip": float(pA - pA_eff),
                "pB_clip": float(pB - pB_eff),
            }
        )

        try:
            raw, m2 = _p11_gaussian_tau(pA_eff, pB_eff, tau)
        except ImportError as e:
            raise ImportError(
                "gaussian_tau requested but SciPy is unavailable. "
                "Install scipy or choose fh_linear/fh_power/fh_scurve."
            ) from e

        # m2 expected numeric-only: {"tau":..., "rho":...}
        for k, v in m2.items():
            meta[k] = float(v)

        return _finalize(float(raw), lam_eff=lam)

    raise InputValidationError(f"Unknown path: {path!r}")


# -----------------------------------------------------------------------------
# Probability validation & sampling
# -----------------------------------------------------------------------------
def _validate_cell_probs(
    p: np.ndarray,
    *,
    prob_tol: float,
    allow_tiny_negative: bool,
    tiny_negative_eps: float,
    context: str = "",
) -> np.ndarray:
    """
    Validate (and optionally tiny-clip) a 4-cell probability vector for a 2×2 Bernoulli joint.

    Contract (no lies):
    - We DO NOT renormalize probabilities. If the vector does not sum to 1 (within prob_tol),
      we raise.
    - The only "repair" permitted is clipping *tiny* out-of-bounds jitter caused by floating
      arithmetic (e.g., -1e-16 or 1+1e-16), controlled by allow_tiny_negative and tiny_negative_eps.
    - After this returns successfully:
        * p is float64 with shape (4,)
        * all entries are finite and in [0,1]
        * sum(p) is within prob_tol of 1

    Parameters
    ----------
    p:
        Array-like of shape (4,) in order [p00, p01, p10, p11].
    prob_tol:
        Allowed absolute deviation of sum(p) from 1.0 (e.g., 1e-12).
    allow_tiny_negative:
        If True, we clip tiny negatives (>= -tiny_negative_eps) up to 0.
        We also symmetrically clip tiny >1 values (<= 1+tiny_negative_eps) down to 1.
    tiny_negative_eps:
        Magnitude threshold for "tiny numerical jitter" clipping.
    context:
        Optional human-readable context appended to error messages (e.g., "w=0, lam=0.25, rep=7").

    Returns
    -------
    np.ndarray
        Validated probability vector (float64). May be a copy if clipping occurred.
    """
    # Defensive checks on tolerances (cfg should already validate, but this is a hard boundary)
    try:
        prob_tol_f = float(prob_tol)
        eps_f = float(tiny_negative_eps)
    except Exception as e:
        raise ValueError(
            f"prob_tol and tiny_negative_eps must be floats. got prob_tol={prob_tol!r}, eps={tiny_negative_eps!r}"
        ) from e

    if not (math.isfinite(prob_tol_f) and prob_tol_f >= 0.0 and prob_tol_f <= 1e-3):
        raise ValueError(f"prob_tol must be finite and reasonably small, got {prob_tol_f}")
    if allow_tiny_negative:
        if not (math.isfinite(eps_f) and eps_f > 0.0 and eps_f <= 1e-6):
            raise ValueError(f"tiny_negative_eps must be finite in (0, 1e-6], got {eps_f}")

    # Normalize input type/shape without mutating caller data unless we must clip.
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.shape != (4,):
        # allow (4,1), (1,4), etc. only if it is exactly 4 elements
        if p_arr.size == 4:
            p_arr = p_arr.reshape(
                4,
            )
        else:
            raise ValueError(
                f"Expected p shape (4,), got {p_arr.shape} (size={p_arr.size}).{(' ' + context) if context else ''}"
            )

    if not np.all(np.isfinite(p_arr)):
        raise ValueError(
            f"Non-finite probabilities: {p_arr.tolist()}.{(' ' + context) if context else ''}"
        )

    # Out-of-bounds handling: ONLY tiny jitter is clipped (no renormalization).
    pmin = float(p_arr.min())
    pmax = float(p_arr.max())
    clipped_any = False

    if pmin < 0.0:
        if allow_tiny_negative and pmin >= -eps_f:
            p_arr = p_arr.copy()
            p_arr[p_arr < 0.0] = 0.0
            clipped_any = True
        else:
            raise ValueError(
                f"Negative cell probability encountered: min={pmin}, p={p_arr.tolist()}.{(' ' + context) if context else ''}"
            )

    if pmax > 1.0:
        # Symmetric “tiny jitter” rule: if we're allowing tiny negatives, we also allow tiny >1
        if allow_tiny_negative and pmax <= 1.0 + eps_f:
            if not clipped_any:
                p_arr = p_arr.copy()
            p_arr[p_arr > 1.0] = 1.0
            clipped_any = True
        else:
            raise ValueError(
                f"Cell probability > 1 encountered: max={pmax}, p={p_arr.tolist()}.{(' ' + context) if context else ''}"
            )

    # Post-clipping hard bounds check (should now be clean)
    if float(p_arr.min()) < 0.0 or float(p_arr.max()) > 1.0:
        raise ValueError(
            f"Probabilities out of bounds after clipping: p={p_arr.tolist()}.{(' ' + context) if context else ''}"
        )

    s = float(p_arr.sum())
    if not math.isfinite(s):
        raise ValueError(
            f"Probability sum is non-finite: sum={s}, p={p_arr.tolist()}.{(' ' + context) if context else ''}"
        )

    # No renormalization: either it's valid within tolerance or it's an error.
    err = abs(s - 1.0)
    if err > prob_tol_f:
        # If clipping happened, call it out explicitly so debugging is honest.
        clip_note = " (after tiny clipping)" if clipped_any else ""
        raise ValueError(
            f"Cell probabilities do not sum to 1 within tol: sum={s} (|Δ|={err}, tol={prob_tol_f}){clip_note}. "
            f"p={p_arr.tolist()}.{(' ' + context) if context else ''}"
        )

    return p_arr


def _draw_joint_counts(
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

    Design intent (research-grade, no lies):
    - We validate p = [p00,p01,p10,p11] strictly (no renormalization).
    - We allow only *tiny* floating jitter repairs (clipping), governed by
      allow_tiny_negative / tiny_negative_eps.
    - We assert the draw is internally consistent: sum(counts) == n.

    Parameters
    ----------
    rng:
        numpy Generator used for sampling.
    n:
        Total sample size for the multinomial draw (must be a positive integer).
    p00, p01, p10, p11:
        Cell probabilities in the order matching `joint_cells_from_marginals`.
    prob_tol, allow_tiny_negative, tiny_negative_eps:
        Passed through to `_validate_cell_probs`.
    context:
        Optional string appended to error messages for traceability.

    Returns
    -------
    (N00, N01, N10, N11): Tuple[int, int, int, int]
    """
    # RNG sanity (helps catch accidental passing of RandomState / None)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}.{(' ' + context) if context else ''}"
        )

    # n must be a strict integer (no silent coercion)
    if isinstance(n, bool):
        raise TypeError(f"n must be an int > 0, got bool {n}.{(' ' + context) if context else ''}")
    if isinstance(n, (float, np.floating)):
        raise TypeError(
            f"n must be an integer (no silent coercion), got {n!r}.{(' ' + context) if context else ''}"
        )
    try:
        n_int = int(n)
    except Exception as e:
        raise TypeError(
            f"n must be an int > 0, got {n!r}.{(' ' + context) if context else ''}"
        ) from e
    if n_int <= 0:
        raise ValueError(f"n must be positive, got {n_int}.{(' ' + context) if context else ''}")
    if n_int != n:
        # If caller passed a float like 1000.0, we fail loudly; don't silently coerce.
        raise TypeError(
            f"n must be an integer (no silent coercion), got {n!r}.{(' ' + context) if context else ''}"
        )

    # Validate probabilities as a joint simplex point (no renormalization)
    p = np.array([p00, p01, p10, p11], dtype=np.float64)
    p = _validate_cell_probs(
        p,
        prob_tol=prob_tol,
        allow_tiny_negative=allow_tiny_negative,
        tiny_negative_eps=tiny_negative_eps,
        context=context,
    )

    # Draw: Generator.multinomial returns shape (k,) when size=None
    counts = rng.multinomial(n_int, pvals=p, size=None)

    if counts.shape != (4,):
        raise RuntimeError(
            f"Unexpected multinomial output shape: {counts.shape}, expected (4,).{(' ' + context) if context else ''}"
        )

    c0, c1, c2, c3 = (int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]))
    s = c0 + c1 + c2 + c3
    if s != n_int:
        # This should never happen; if it does, something is deeply wrong (dtype overflow, corrupted RNG, etc.)
        raise RuntimeError(
            f"Multinomial draw inconsistent: sum(counts)={s} != n={n_int}. counts={counts.tolist()}.{(' ' + context) if context else ''}"
        )

    return c0, c1, c2, c3


def _empirical_from_counts(
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

    Contract (explicit):
    - Inputs are integer counts for a single 2×2 contingency table.
    - No silent coercions: floats like 1.0 or numpy scalars are rejected unless they
      are true ints already. This prevents accidental truncation.
    - Counts must be non-negative and sum exactly to n.

    Notes:
    - phi_hat can be NaN when margins are degenerate (pA in {0,1} or pB in {0,1}).
      We record explicit degeneracy flags in the return dict.
    - tau_hat can be NaN for certain degenerate joints; we pass it through unchanged
      but record a flag when it's non-finite.

    Returns
    -------
    Dict[str, float]
      Core keys:
        p00_hat, p01_hat, p10_hat, p11_hat, pA_hat, pB_hat, pC_hat, phi_hat, tau_hat

      Diagnostics (float 0/1):
        degenerate_A, degenerate_B, phi_finite, tau_finite

      Also includes counts as floats (useful for long-form logging):
        n, n00, n01, n10, n11
    """
    ctx = f" {context}" if context else ""

    # Rule validation
    if rule not in ("OR", "AND"):
        raise ValueError(f"Invalid rule: {rule!r}.{ctx}")

    # Strict integer validation (reject bools and silent float coercion)
    def _as_strict_int(x: Any, name: str) -> int:
        if isinstance(x, bool):
            raise TypeError(f"{name} must be an int, got bool {x}.{ctx}")
        if isinstance(x, (np.integer, int)):
            return int(x)
        # Reject floats/strings/etc. explicitly (no silent truncation).
        raise TypeError(f"{name} must be an int, got {type(x).__name__}={x!r}.{ctx}")

    n_i = _as_strict_int(n, "n")
    n00_i = _as_strict_int(n00, "n00")
    n01_i = _as_strict_int(n01, "n01")
    n10_i = _as_strict_int(n10, "n10")
    n11_i = _as_strict_int(n11, "n11")

    if n_i <= 0:
        raise ValueError(f"n must be positive, got {n_i}.{ctx}")

    # Count sanity
    if (n00_i < 0) or (n01_i < 0) or (n10_i < 0) or (n11_i < 0):
        raise ValueError(
            f"Counts must be non-negative. "
            f"Got (n00,n01,n10,n11)=({n00_i},{n01_i},{n10_i},{n11_i}).{ctx}"
        )

    s = n00_i + n01_i + n10_i + n11_i
    if s != n_i:
        raise ValueError(
            f"Counts do not sum to n. sum={s}, n={n_i}. "
            f"(n00,n01,n10,n11)=({n00_i},{n01_i},{n10_i},{n11_i}).{ctx}"
        )

    # Convert to empirical probabilities (exact rationals -> float64)
    inv_n = 1.0 / float(n_i)
    p00 = float(n00_i) * inv_n
    p01 = float(n01_i) * inv_n
    p10 = float(n10_i) * inv_n
    p11 = float(n11_i) * inv_n

    # Empirical marginals
    pA = (float(n10_i + n11_i)) * inv_n
    pB = (float(n01_i + n11_i)) * inv_n

    # Joint cell dict for theory helpers
    cells = {"p00": p00, "p01": p01, "p10": p10, "p11": p11}

    # Composition probability (computed from empirical joint + empirical marginals)
    pC = float(pC_from_joint(rule, cells, pA=pA, pB=pB))

    # Dependence summaries (may be NaN for degenerate margins)
    phi = float(phi_from_joint(pA, pB, p11))
    tau = float(kendall_tau_a_from_joint(cells))

    # Degeneracy / finiteness diagnostics
    # (Exact equality is fine here because pA,pB are rational with denominator n.)
    degA = 1.0 if (pA <= 0.0 or pA >= 1.0) else 0.0
    degB = 1.0 if (pB <= 0.0 or pB >= 1.0) else 0.0
    phi_finite = 1.0 if math.isfinite(phi) else 0.0
    tau_finite = 1.0 if math.isfinite(tau) else 0.0

    # Return core estimates + audit-friendly extras
    return {
        # counts (float for DataFrame friendliness)
        "n": float(n_i),
        "n00": float(n00_i),
        "n01": float(n01_i),
        "n10": float(n10_i),
        "n11": float(n11_i),
        # empirical joint
        "p00_hat": float(p00),
        "p01_hat": float(p01),
        "p10_hat": float(p10),
        "p11_hat": float(p11),
        # empirical marginals + composed
        "pA_hat": float(pA),
        "pB_hat": float(pB),
        "pC_hat": float(pC),
        # dependence
        "phi_hat": float(phi),
        "tau_hat": float(tau),
        # diagnostics
        "degenerate_A": float(degA),
        "degenerate_B": float(degB),
        "phi_finite": float(phi_finite),
        "tau_finite": float(tau_finite),
    }


# -----------------------------------------------------------------------------
# Core simulation
# -----------------------------------------------------------------------------
def simulate_replicate_at_lambda(
    cfg: SimConfig,
    *,
    lam: float,
    lam_index: int,
    rep: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Simulate one replicate at a given lambda.

    Enterprise / research-grade guarantees:
    - Never silently “fixes” invalid probabilities or invalid joints.
    - If cfg.hard_fail_on_invalid is False, all failures are *explicitly recorded*
      with per-world stage + message, and all downstream numeric fields become NaN.
    - Stable seeding policy is honored exactly.

    Output schema:
    - Always emits per-world fields for w in {0,1} (NaNs if invalid).
    - Always emits derived empirical + population overlays (NaNs if insufficient data).
    - Always emits envelope flag diagnostics.
    """
    # --- Validate identifiers early (fail fast, deterministic errors) ---
    lam_f = float(lam)
    if not (0.0 <= lam_f <= 1.0) or not math.isfinite(lam_f):
        raise ValueError(f"lambda must be finite and in [0,1], got {lam!r}")
    li = int(lam_index)
    rp = int(rep)
    if li < 0:
        raise ValueError(f"lambda_index must be >=0, got {lam_index!r}")
    if rp < 0:
        raise ValueError(f"rep must be >=0, got {rep!r}")

    out: Dict[str, Any] = {
        "lambda": lam_f,
        "lambda_index": li,
        "rep": rp,
        "rule": cfg.rule,
        "path": cfg.path,
        "seed": int(cfg.seed),
        "seed_policy": cfg.seed_policy,
        "n_per_world": int(cfg.n),
        "hard_fail_on_invalid": bool(cfg.hard_fail_on_invalid),
    }

    # Population feasibility envelope for JC (path-independent; FH feasibility only)
    jmin, jmax = compute_fh_jc_envelope(cfg.marginals, cfg.rule)
    out["JC_env_min"] = float(jmin)
    out["JC_env_max"] = float(jmax)

    # Predeclare per-world fields so the DataFrame schema is stable even on failure.
    # (We keep booleans as float 0/1 only when it helps aggregation; otherwise bool is fine.)
    _WORLD_NUM_FIELDS = (
        # true marginals / joint
        "pA_true",
        "pB_true",
        "p00_true",
        "p01_true",
        "p10_true",
        "p11_true",
        # counts
        "n00",
        "n01",
        "n10",
        "n11",
        # hats (core)
        "p00_hat",
        "p01_hat",
        "p10_hat",
        "p11_hat",
        "pA_hat",
        "pB_hat",
        "pC_hat",
        "phi_hat",
        "tau_hat",
        # hats (diagnostics from _empirical_from_counts, if present)
        "degenerate_A",
        "degenerate_B",
        "phi_finite",
        "tau_finite",
        # path-consistent population overlays from constructed joint
        "pC_true",
        "phi_true",
        "tau_true",
    )

    def _prime_world_schema(w: int) -> None:
        out[f"world_valid_w{w}"] = True
        out[f"world_error_stage_w{w}"] = ""
        out[f"world_error_msg_w{w}"] = ""
        for base in _WORLD_NUM_FIELDS:
            out[f"{base}_w{w}"] = float("nan")

    def _mark_world_invalid(w: int, *, stage: str, msg: str) -> None:
        out[f"world_valid_w{w}"] = False
        out[f"world_error_stage_w{w}"] = str(stage)
        out[f"world_error_msg_w{w}"] = str(msg)

        # Ensure all numeric fields exist (already primed) and remain NaN.
        # Do NOT delete anything; stability > cleverness.

    for w in (0, 1):
        _prime_world_schema(w)

    # ----------------------------
    # Per-world: construct joint + sample + compute hats + pop overlays
    # ----------------------------
    for w, wm in ((0, cfg.marginals.w0), (1, cfg.marginals.w1)):
        out[f"pA_true_w{w}"] = float(wm.pA)
        out[f"pB_true_w{w}"] = float(wm.pB)

        # Choose RNG (order-invariant by default)
        rng_w = rng if cfg.seed_policy == "sequential" else _rng_for_cell(cfg.seed, rp, li, w)

        # Choose p11 per configured path
        try:
            p11, meta = p11_from_path(
                wm.pA, wm.pB, lam_f, path=cfg.path, path_params=cfg.path_params
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="p11_from_path", msg=f"{type(e).__name__}: {e}")
            continue

        for mk, mv in meta.items():
            # Preserve your existing column naming scheme for compatibility.
            out[f"{mk}_w{w}"] = float(mv)

        # Construct full joint
        try:
            cells = joint_cells_from_marginals(wm.pA, wm.pB, float(p11))
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(
                w, stage="joint_cells_from_marginals", msg=f"{type(e).__name__}: {e}"
            )
            continue

        # Validate joint probabilities (no silent normalization)
        try:
            _ = _validate_cell_probs(
                np.array([cells["p00"], cells["p01"], cells["p10"], cells["p11"]], dtype=float),
                prob_tol=cfg.prob_tol,
                allow_tiny_negative=cfg.allow_tiny_negative,
                tiny_negative_eps=cfg.tiny_negative_eps,
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="validate_joint_probs", msg=f"{type(e).__name__}: {e}")
            continue

        # Record true joint
        out[f"p00_true_w{w}"] = float(cells["p00"])
        out[f"p01_true_w{w}"] = float(cells["p01"])
        out[f"p10_true_w{w}"] = float(cells["p10"])
        out[f"p11_true_w{w}"] = float(cells["p11"])

        # Sample multinomial joint counts
        try:
            n00, n01, n10, n11 = _draw_joint_counts(
                rng_w,
                n=cfg.n,
                p00=float(cells["p00"]),
                p01=float(cells["p01"]),
                p10=float(cells["p10"]),
                p11=float(cells["p11"]),
                prob_tol=cfg.prob_tol,
                allow_tiny_negative=cfg.allow_tiny_negative,
                tiny_negative_eps=cfg.tiny_negative_eps,
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="draw_joint_counts", msg=f"{type(e).__name__}: {e}")
            continue

        out[f"n00_w{w}"] = int(n00)
        out[f"n01_w{w}"] = int(n01)
        out[f"n10_w{w}"] = int(n10)
        out[f"n11_w{w}"] = int(n11)

        # Empirical estimates from counts
        try:
            hats = _empirical_from_counts(
                n=cfg.n,
                n00=n00,
                n01=n01,
                n10=n10,
                n11=n11,
                rule=cfg.rule,
                context=f"(lam={lam_f:.6g}, idx={li}, rep={rp}, w={w})",
            )
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            _mark_world_invalid(w, stage="empirical_from_counts", msg=f"{type(e).__name__}: {e}")
            continue

        # Store hats, but protect against accidental key collisions
        # (Counts already stored as n00_w*, etc. If hats returns n00/n01/... keys, ignore them.)
        _COLLIDE = {"n", "n00", "n01", "n10", "n11"}
        for hk, hv in hats.items():
            if hk in _COLLIDE:
                continue
            out[f"{hk}_w{w}"] = float(hv)

        # Path-consistent population overlays (derived from the constructed joint)
        try:
            pC_true = float(pC_from_joint(cfg.rule, cells, pA=float(wm.pA), pB=float(wm.pB)))
            phi_true = float(phi_from_joint(float(wm.pA), float(wm.pB), float(cells["p11"])))
            tau_true = float(kendall_tau_a_from_joint(cells))
        except Exception as e:
            if cfg.hard_fail_on_invalid:
                raise
            # This is unusual, but we handle it cleanly.
            _mark_world_invalid(w, stage="population_overlays", msg=f"{type(e).__name__}: {e}")
            continue

        out[f"pC_true_w{w}"] = float(pC_true)
        out[f"phi_true_w{w}"] = float(phi_true)
        out[f"tau_true_w{w}"] = float(tau_true)

    # ----------------------------
    # Derived empirical metrics across worlds
    # ----------------------------
    w0_ok = bool(out.get("world_valid_w0", False))
    w1_ok = bool(out.get("world_valid_w1", False))
    out["worlds_valid"] = bool(w0_ok and w1_ok)

    pA0 = float(out.get("pA_hat_w0", float("nan")))
    pA1 = float(out.get("pA_hat_w1", float("nan")))
    pB0 = float(out.get("pB_hat_w0", float("nan")))
    pB1 = float(out.get("pB_hat_w1", float("nan")))
    pC0 = float(out.get("pC_hat_w0", float("nan")))
    pC1 = float(out.get("pC_hat_w1", float("nan")))

    JA_hat = abs(pA1 - pA0)
    JB_hat = abs(pB1 - pB0)
    Jbest_hat = max(JA_hat, JB_hat)
    dC_hat = pC1 - pC0
    JC_hat = abs(dC_hat)
    CC_hat = (
        (JC_hat / Jbest_hat) if (math.isfinite(Jbest_hat) and Jbest_hat > 0.0) else float("nan")
    )

    out["JA_hat"] = float(JA_hat)
    out["JB_hat"] = float(JB_hat)
    out["Jbest_hat"] = float(Jbest_hat)
    out["dC_hat"] = float(dC_hat)
    out["JC_hat"] = float(JC_hat)
    out["CC_hat"] = float(CC_hat)

    # Empirical dependence summaries averaged across worlds
    phi0 = float(out.get("phi_hat_w0", float("nan")))
    phi1 = float(out.get("phi_hat_w1", float("nan")))
    tau0 = float(out.get("tau_hat_w0", float("nan")))
    tau1 = float(out.get("tau_hat_w1", float("nan")))
    out["phi_hat_avg"] = float(0.5 * (phi0 + phi1))
    out["tau_hat_avg"] = float(0.5 * (tau0 + tau1))

    # ----------------------------
    # Path-consistent population overlays across worlds
    # ----------------------------
    pC0_true = float(out.get("pC_true_w0", float("nan")))
    pC1_true = float(out.get("pC_true_w1", float("nan")))
    dC_pop = pC1_true - pC0_true
    JC_pop = abs(dC_pop)

    JA_pop = abs(float(cfg.marginals.w1.pA) - float(cfg.marginals.w0.pA))
    JB_pop = abs(float(cfg.marginals.w1.pB) - float(cfg.marginals.w0.pB))
    Jbest_pop = max(JA_pop, JB_pop)
    CC_pop = (
        (JC_pop / Jbest_pop) if (Jbest_pop > 0.0 and math.isfinite(Jbest_pop)) else float("nan")
    )

    out["dC_pop"] = float(dC_pop)
    out["JC_pop"] = float(JC_pop)
    out["JA_pop"] = float(JA_pop)
    out["JB_pop"] = float(JB_pop)
    out["Jbest_pop"] = float(Jbest_pop)
    out["CC_pop"] = float(CC_pop)

    phi0_true = float(out.get("phi_true_w0", float("nan")))
    phi1_true = float(out.get("phi_true_w1", float("nan")))
    tau0_true = float(out.get("tau_true_w0", float("nan")))
    tau1_true = float(out.get("tau_true_w1", float("nan")))
    out["phi_pop_avg"] = float(0.5 * (phi0_true + phi1_true))
    out["tau_pop_avg"] = float(0.5 * (tau0_true + tau1_true))

    # ----------------------------
    # Optional: theory reference overlays (separate, explicitly labeled)
    # ----------------------------
    if cfg.include_theory_reference and callable(_compute_metrics_for_lambda):
        try:
            theory = _compute_metrics_for_lambda(cfg.marginals, cfg.rule, lam_f)  # type: ignore[misc]
            out["CC_theory_ref"] = float(theory.get("CC", float("nan")))
            out["JC_theory_ref"] = float(theory.get("JC", float("nan")))
            out["dC_theory_ref"] = float(theory.get("dC", float("nan")))
            out["phi_theory_ref_avg"] = float(theory.get("phi_avg", float("nan")))
            out["tau_theory_ref_avg"] = float(theory.get("tau_avg", float("nan")))
            out["CC_ref_minus_pop"] = float(out["CC_theory_ref"] - out["CC_pop"])
            out["JC_ref_minus_pop"] = float(out["JC_theory_ref"] - out["JC_pop"])
        except Exception as e:
            out["theory_ref_error"] = f"{type(e).__name__}: {e}"

    # ----------------------------
    # Envelope flagging: compare empirical JC_hat to population FH envelope [jmin,jmax]
    # ----------------------------
    tol = float(cfg.envelope_tol)
    if math.isfinite(JC_hat) and math.isfinite(float(jmin)) and math.isfinite(float(jmax)):
        low = float(jmin) - tol
        high = float(jmax) + tol
        violated_low = JC_hat < low
        violated_high = JC_hat > high
        violated = bool(violated_low or violated_high)
        out["JC_env_violation"] = violated
        out["JC_env_violation_low"] = bool(violated_low)
        out["JC_env_violation_high"] = bool(violated_high)
        if violated_low:
            out["JC_env_gap"] = float(low - JC_hat)
        elif violated_high:
            out["JC_env_gap"] = float(JC_hat - high)
        else:
            out["JC_env_gap"] = 0.0
    else:
        out["JC_env_violation"] = False
        out["JC_env_violation_low"] = False
        out["JC_env_violation_high"] = False
        out["JC_env_gap"] = float("nan")

    return out


def simulate_grid(cfg: SimConfig) -> pd.DataFrame:
    """
    Run simulation across all lambdas and replicates.

    Enterprise / PhD-grade behaviors:
    - Validates cfg and lambda grid up front (type/range/finite).
    - Deterministic row schema and deterministic ordering of identifiers.
    - Honors RNG policy exactly:
        * sequential: single stream seeded by cfg.seed (loop-order dependent by design)
        * stable_per_cell: per-cell streams keyed inside simulate_replicate_at_lambda
          (loop-order invariant by design)
    - Failure semantics:
        * If cfg.hard_fail_on_invalid: any exception bubbles immediately.
        * Else: we record a row-level error and continue (no silent drop).

    Returns
    -------
    pd.DataFrame
        One row per (lambda, rep), with rich per-world and diagnostic columns.
    """
    _validate_cfg(cfg)

    # Materialize lambdas once (handles generators, numpy arrays, etc.)
    lambdas_list = [float(x) for x in cfg.lambdas]
    if len(lambdas_list) == 0:
        raise ValueError("cfg.lambdas must be non-empty.")

    # Validate lambdas up-front so failures are consistent and fast.
    # (simulate_replicate_at_lambda also checks, but this avoids partial runs.)
    for i, lam in enumerate(lambdas_list):
        if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
            raise ValueError(f"Invalid lambda at index {i}: must be finite in [0,1], got {lam!r}")

    # RNG policy:
    # - sequential: a single RNG stream; depends on loop order (intentional)
    # - stable_per_cell: per-(rep,lambda_index,world) streams; order-invariant (intentional)
    if cfg.seed_policy == "sequential":
        base_rng = np.random.default_rng(int(cfg.seed))
    else:
        # Base RNG is intentionally unused for per-cell RNG policy; still deterministic.
        base_rng = np.random.default_rng(0)

    total = int(cfg.n_reps) * int(len(lambdas_list))
    rows: list[Dict[str, Any]] = []
    rows_extend = rows.append  # tiny speed win in hot loops

    for rep in range(int(cfg.n_reps)):
        for lam in lambdas_list:
            lam_index = cfg.lambda_index_for_seed(lam)
            try:
                row = simulate_replicate_at_lambda(
                    cfg,
                    lam=float(lam),
                    lam_index=int(lam_index),
                    rep=int(rep),
                    rng=base_rng,
                )
                # Row-level success marker (handy for filtering)
                row.setdefault("row_ok", True)
                row.setdefault("row_error_stage", "")
                row.setdefault("row_error_msg", "")
                rows_extend(row)
            except Exception as e:
                if cfg.hard_fail_on_invalid:
                    raise

                # Non-fatal row: record explicit diagnostics and continue.
                # Keep minimal stable identifiers so downstream aggregation can skip or diagnose.
                rows_extend(
                    {
                        "lambda": float(lam),
                        "lambda_index": int(lam_index),
                        "rep": int(rep),
                        "rule": cfg.rule,
                        "path": cfg.path,
                        "seed": int(cfg.seed),
                        "seed_policy": cfg.seed_policy,
                        "n_per_world": int(cfg.n),
                        "row_ok": False,
                        "row_error_stage": "simulate_replicate_at_lambda",
                        "row_error_msg": f"{type(e).__name__}: {e}",
                    }
                )

    df = pd.DataFrame.from_records(rows)

    # Deterministic output ordering for reproducibility and plotting.
    # (Sorting does NOT affect RNG, which has already been consumed.)
    sort_cols = [c for c in ("lambda", "rep") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Optional sanity check: confirm expected number of rows
    # (We avoid raising here because cfg.hard_fail_on_invalid=False means partial failures are allowed.)
    if len(df) != total and cfg.hard_fail_on_invalid:
        raise RuntimeError(f"simulate_grid produced {len(df)} rows, expected {total}")

    return df


def summarize_simulation(
    df_long: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """
    Aggregate replicate-level results to produce per-lambda summaries.

    Enterprise / PhD-grade behaviors:
    - Never mixes different (rule, path) experiments into one curve:
        * If 'rule'/'path' columns exist, summaries are computed per (rule, path, lambda).
    - Quantile hygiene:
        * validates quantiles are finite, unique, within [0,1].
        * stable column naming: qXXXX where XXXX = round(q*1000).
    - Robust to partial failures:
        * if 'row_ok' exists, reports ok/fail counts + error rates.
    - Population overlay consistency checks:
        * if pop columns vary across reps (should not), reports drift diagnostics.

    Returns
    -------
    pd.DataFrame
        One row per group (lambda [+ optional rule/path]).
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    if "lambda" not in df_long.columns:
        raise ValueError("df_long must contain a 'lambda' column.")

    # ---- quantile validation ----
    q_raw = [float(x) for x in quantiles]
    if len(q_raw) == 0:
        raise ValueError("quantiles must be non-empty.")
    for qq in q_raw:
        if not math.isfinite(qq) or not (0.0 <= qq <= 1.0):
            raise ValueError(f"Invalid quantile {qq!r}: must be finite and in [0,1].")
    # de-dup (with a little tolerance for float inputs), then sort
    q = sorted({round(qq, 12) for qq in q_raw})
    if len(q) != len(q_raw):
        # duplicates are usually accidental; force user to be explicit
        raise ValueError(f"quantiles contains duplicates after rounding: {quantiles!r}")

    def _q_label(qq: float) -> str:
        # round to avoid 0.975*1000 becoming 974 due to float representation
        return f"q{int(round(qq * 1000.0)):04d}"

    def _qcols(s: pd.Series, prefix: str) -> Dict[str, float]:
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        if s2.empty:
            return {f"{prefix}_{_q_label(qq)}": float("nan") for qq in q}
        qs = s2.quantile(q)
        out: Dict[str, float] = {}
        for qq in q:
            out[f"{prefix}_{_q_label(qq)}"] = float(qs.loc[qq])
        return out

    # ---- grouping keys (avoid mixing experiments) ----
    group_keys: List[str] = ["lambda"]
    for k in ("rule", "path"):
        if k in df_long.columns:
            group_keys.append(k)

    # Normalize lambda for stable grouping / sorting
    df = df_long.copy()
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    if df["lambda"].isna().any():
        bad = df_long.loc[df["lambda"].isna(), "lambda"].head(5).tolist()
        raise ValueError(f"Found non-numeric lambda values (first few): {bad!r}")

    # ---- columns we summarize ----
    core_cols = (
        "CC_hat",
        "JC_hat",
        "dC_hat",
        "JA_hat",
        "JB_hat",
        "Jbest_hat",
        "phi_hat_avg",
        "tau_hat_avg",
    )

    pop_cols = (
        "CC_pop",
        "JC_pop",
        "dC_pop",
        "phi_pop_avg",
        "tau_pop_avg",
        "JC_env_min",
        "JC_env_max",
    )

    theory_cols = (
        "CC_theory_ref",
        "JC_theory_ref",
        "dC_theory_ref",
        "phi_theory_ref_avg",
        "tau_theory_ref_avg",
    )

    groups: list[Dict[str, Any]] = []
    gb = df.groupby(group_keys, sort=True, dropna=False)

    for key, g in gb:
        # key is scalar if group_keys == ["lambda"], else tuple in group_keys order
        row: Dict[str, Any] = {}

        if isinstance(key, tuple):
            for kname, kval in zip(group_keys, key):
                row[kname] = (
                    float(kval) if kname == "lambda" else (None if pd.isna(kval) else str(kval))
                )
        else:
            row["lambda"] = float(key)

        # Replicate accounting
        row["n_rows"] = int(len(g))
        if "rep" in g.columns:
            row["n_reps"] = int(pd.to_numeric(g["rep"], errors="coerce").nunique(dropna=True))
        else:
            row["n_reps"] = int(len(g))

        # Row-level success/failure accounting (if present)
        if "row_ok" in g.columns:
            ok = g["row_ok"].fillna(False).astype(bool)
            row["row_ok_rate"] = float(ok.mean()) if len(ok) > 0 else float("nan")
            row["n_row_ok"] = int(ok.sum())
            row["n_row_fail"] = int((~ok).sum())
        else:
            row["row_ok_rate"] = float("nan")
            row["n_row_ok"] = int(len(g))
            row["n_row_fail"] = 0

        # Core empirical metrics: mean/std + missing rate
        for col in core_cols:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(s.mean())
                row[f"{col}_std"] = float(s.std(ddof=1)) if s.notna().sum() >= 2 else float("nan")
                row[f"{col}_n"] = int(s.notna().sum())
                row[f"{col}_nan_rate"] = float(s.isna().mean())

        # Quantiles for headline stats
        if "CC_hat" in g.columns:
            row.update(_qcols(g["CC_hat"], "CC_hat"))
        if "JC_hat" in g.columns:
            row.update(_qcols(g["JC_hat"], "JC_hat"))

        # Envelope violation rate (empirical)
        if "JC_env_violation" in g.columns:
            v = g["JC_env_violation"].fillna(False).astype(bool)
            row["JC_env_violation_rate"] = float(v.mean()) if len(v) > 0 else float("nan")
            row["JC_env_violation_n"] = int(v.sum())
        else:
            row["JC_env_violation_rate"] = float("nan")
            row["JC_env_violation_n"] = 0

        # Invalid joint rates (if present)
        inv_cols = [c for c in g.columns if c.startswith("invalid_joint_w")]
        for c in inv_cols:
            v = g[c].fillna(False).astype(bool)
            row[f"{c}_rate"] = float(v.mean()) if len(v) > 0 else float("nan")
            row[f"{c}_n"] = int(v.sum())

        # Population overlays (should be identical across reps; keep value + drift diagnostics)
        for col in pop_cols:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                if s.empty:
                    row[col] = float("nan")
                    row[f"{col}_drift"] = float("nan")
                    row[f"{col}_nonconstant"] = False
                else:
                    v0 = float(s.iloc[0])
                    drift = float(s.max() - s.min()) if len(s) > 1 else 0.0
                    row[col] = v0
                    row[f"{col}_drift"] = drift
                    row[f"{col}_nonconstant"] = bool(drift != 0.0)

        # Optional theory reference overlays (first non-null; also drift check)
        for col in theory_cols:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                if s.empty:
                    row[col] = float("nan")
                    row[f"{col}_drift"] = float("nan")
                    row[f"{col}_nonconstant"] = False
                else:
                    v0 = float(s.iloc[0])
                    drift = float(s.max() - s.min()) if len(s) > 1 else 0.0
                    row[col] = v0
                    row[f"{col}_drift"] = drift
                    row[f"{col}_nonconstant"] = bool(drift != 0.0)

        groups.append(row)

    df_sum = pd.DataFrame(groups)

    # Deterministic sort
    sort_cols = [c for c in ("rule", "path", "lambda") if c in df_sum.columns]
    if sort_cols:
        df_sum = df_sum.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df_sum = df_sum.sort_values("lambda", kind="mergesort").reset_index(drop=True)

    return df_sum


# -----------------------------------------------------------------------------
# Convenience: grid builder
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

    Design goals
    ------------
    - **Unambiguous contract**: `num` is the number of points returned (after endpoint handling).
    - **Research-safe**: validates finiteness + interval ordering; optional endpoint snapping.
    - **Endpoint control** via `closed`:
        * "both"    -> include start and stop (standard linspace)
        * "neither" -> exclude both endpoints
        * "left"    -> include start, exclude stop
        * "right"   -> exclude start, include stop

    Parameters
    ----------
    num:
        Number of points returned in the final grid (must be >= 1; stronger constraints
        apply depending on `closed`).
    start, stop:
        Interval endpoints. For lambda, you typically want 0 <= start < stop <= 1.
    closed:
        Endpoint inclusion policy.
    dtype:
        Output dtype (float recommended).
    snap_eps:
        If > 0, values within snap_eps of the endpoints are snapped exactly to start/stop.
        This helps avoid float edge artifacts when later doing comparisons or dictionary keys.

    Returns
    -------
    np.ndarray
        1D array, strictly increasing (unless num==1), of length `num`.

    Notes
    -----
    - For closed="neither"/"left"/"right", we internally build a slightly larger linspace
      and slice, ensuring you still get exactly `num` returned points.
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
        # Keep this strict because this is a *lambda* grid builder.
        raise ValueError(f"Lambda grid expects start/stop in [0,1], got start={a}, stop={b}")

    if closed not in ("both", "neither", "left", "right"):
        raise ValueError(f"closed must be one of {{both, neither, left, right}}, got {closed!r}")

    # Stronger "usability" constraints depending on endpoint policy.
    # (These prevent common accidental configs that produce degenerate/empty grids.)
    if closed == "both" and num_i < 2:
        # With both endpoints, num=1 would collapse to just 'start' or 'stop' ambiguity.
        raise ValueError("closed='both' requires num >= 2.")
    if closed in ("left", "right") and num_i < 1:
        raise ValueError(f"closed={closed!r} requires num >= 1.")
    if closed == "neither" and num_i < 1:
        raise ValueError("closed='neither' requires num >= 1.")

    # Build grid with guaranteed length == num_i after slicing.
    if closed == "both":
        grid = np.linspace(a, b, num=num_i, endpoint=True, dtype=float)
    elif closed == "neither":
        grid = np.linspace(a, b, num=num_i + 2, endpoint=True, dtype=float)[1:-1]
    elif closed == "left":
        # include start, exclude stop
        grid = np.linspace(a, b, num=num_i + 1, endpoint=True, dtype=float)[:-1]
    else:  # closed == "right"
        # exclude start, include stop
        grid = np.linspace(a, b, num=num_i + 1, endpoint=True, dtype=float)[1:]

    # Optional snapping for stable boundary behavior
    if snap_eps:
        eps = float(snap_eps)
        if not (math.isfinite(eps) and eps >= 0.0):
            raise ValueError(f"snap_eps must be finite and >= 0, got {snap_eps!r}")
        if eps > 0.0 and grid.size > 0:
            grid = grid.copy()
            grid[np.abs(grid - a) <= eps] = a
            grid[np.abs(grid - b) <= eps] = b

    # Final hard guarantees
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


# -----------------------------------------------------------------------------
# CLI config loading (enterprise-grade, schema-aware)
# -----------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML config into a plain Python dict with strong error messages.

    Guarantees:
      - returns {} for empty YAML
      - returns a *mapping* (dict) only; raises otherwise
      - never silently swallows YAML parse errors
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError("pyyaml is required for --config usage (pip install pyyaml).") from e

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        raise OSError(f"Could not read YAML config at path={path!r}: {e}") from e

    try:
        obj = yaml.safe_load(raw)
    except Exception as e:
        # PyYAML exceptions sometimes expose a "problem_mark" with line/col; include if present.
        mark = getattr(e, "problem_mark", None)
        if mark is not None:
            loc = f"line={getattr(mark, 'line', '?')}, column={getattr(mark, 'column', '?')}"
            raise ValueError(f"YAML parse error in {path!r} ({loc}): {e}") from e
        raise ValueError(f"YAML parse error in {path!r}: {e}") from e

    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML config must parse to a mapping/dict, got {type(obj).__name__}")
    return dict(obj)


def _cfg_from_dict(d: Dict[str, Any]) -> SimConfig:
    """
    Build SimConfig from a YAML-like dict.

    Supports two schemas:

    (A) Legacy single-run schema:
      marginals.{w0,w1}.{pA,pB}
      rule, n, n_reps, seed
      lambdas OR lambda_grid
      path, path_params
      seed_policy, envelope_tol, hard_fail_on_invalid, prob_tol, allow_tiny_negative, tiny_negative_eps
      include_theory_reference

    (B) Pipeline schema (enterprise; we map to a SINGLE "primary" run here):
      marginals.{w0,w1}.{pA,pB}
      composition.primary_rule
      dependence_paths.primary.type
      dependence_paths.primary.lambda_grid_coarse (or lambda_grid)
      sampling.n_per_world, sampling.n_reps, sampling.seed, sampling.seed_policy
      simulate.* (optional overrides: prob_tol, allow_tiny_negative, tiny_negative_eps, hard_fail_on_invalid, include_theory_reference)
      sanity.* (optional, but simulate.py only consumes prob/simplex controls)

    NOTE: run_all.py is the matrix orchestrator. simulate.py’s _cfg_from_dict returns
    one SimConfig (primary path + primary rule) for CLI and simple harness use.
    """

    # ----------------------------
    # Small internal utilities
    # ----------------------------
    def _dget(obj: Any, dotted: str, default: Any = None) -> Any:
        cur = obj
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _require_dict(x: Any, name: str) -> Dict[str, Any]:
        if not isinstance(x, dict):
            raise ValueError(f"Expected mapping for '{name}', got {type(x).__name__}")
        return x

    def _f(x: Any, name: str, *, finite: bool = True) -> float:
        try:
            v = float(x)
        except Exception as e:
            raise ValueError(f"{name} must be a number, got {x!r}") from e
        if finite and not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
        return v

    def _i(x: Any, name: str) -> int:
        try:
            v = int(x)
        except Exception as e:
            raise ValueError(f"{name} must be an int, got {x!r}") from e
        return v

    def _b(x: Any, name: str) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)) and x in (0, 1):
            return bool(x)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "yes", "y", "1"):
                return True
            if s in ("false", "no", "n", "0"):
                return False
        raise ValueError(f"{name} must be a bool, got {x!r}")

    def _p(x: Any, name: str) -> float:
        v = _f(x, name, finite=True)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")
        return v

    def _parse_path_params(base: Any, *, extra: Dict[str, Any]) -> Dict[str, Any]:
        pp: Dict[str, Any] = {}
        if isinstance(base, dict):
            pp.update(dict(base))
        # Accept convenience keys even if caller didn’t nest them under path_params
        for k, v in extra.items():
            if v is not None and k not in pp:
                pp[k] = v
        return pp

    def _parse_lambdas_from_grid(grid_obj: Any, *, where: str) -> Sequence[float]:
        g = _require_dict(grid_obj, where)
        start = _f(g.get("start", 0.0), f"{where}.start")
        stop = _f(g.get("stop", 1.0), f"{where}.stop")
        num = _i(g.get("num", 21), f"{where}.num")
        # Keep closed='both' for theory + envelope endpoints
        arr = build_linear_lambda_grid(num=num, start=start, stop=stop, closed="both")
        return [float(x) for x in arr.tolist()]

    def _ensure_monotone_increasing(lams: Sequence[float]) -> Sequence[float]:
        if len(lams) == 0:
            raise ValueError("lambdas must be non-empty.")
        xs = [float(x) for x in lams]
        for i, x in enumerate(xs):
            if not (math.isfinite(x) and 0.0 <= x <= 1.0):
                raise ValueError(
                    f"lambda values must be finite and in [0,1]; bad lambdas[{i}]={x!r}"
                )
        # Require non-decreasing; strict increasing is nicer but allow duplicates if user insists.
        for i in range(len(xs) - 1):
            if xs[i + 1] < xs[i]:
                raise ValueError(
                    "lambdas must be non-decreasing (sorted). "
                    f"Found lambdas[{i}]={xs[i]} > lambdas[{i + 1}]={xs[i + 1]}."
                )
        return xs

    # ----------------------------
    # Marginals (shared)
    # ----------------------------
    md = _require_dict(d.get("marginals", {}), "marginals")
    w0 = _require_dict(md.get("w0", {}), "marginals.w0")
    w1 = _require_dict(md.get("w1", {}), "marginals.w1")

    if "pA" not in w0 or "pB" not in w0 or "pA" not in w1 or "pB" not in w1:
        raise ValueError("marginals.w0 and marginals.w1 must each define pA and pB.")

    marg = TwoWorldMarginals(
        w0=WorldMarginals(pA=_p(w0["pA"], "marginals.w0.pA"), pB=_p(w0["pB"], "marginals.w0.pB")),
        w1=WorldMarginals(pA=_p(w1["pA"], "marginals.w1.pA"), pB=_p(w1["pB"], "marginals.w1.pB")),
    )

    # ----------------------------
    # Detect schema
    # ----------------------------
    has_pipeline = (
        isinstance(d.get("composition"), dict)
        or isinstance(d.get("dependence_paths"), dict)
        or isinstance(d.get("sampling"), dict)
    )

    if has_pipeline:
        # Rule (primary only)
        primary_rule = str(_dget(d, "composition.primary_rule", "OR")).upper()
        if primary_rule not in ("OR", "AND"):
            raise ValueError(f"composition.primary_rule must be OR/AND, got {primary_rule!r}")
        rule = primary_rule

        # Path (primary only)
        path = str(_dget(d, "dependence_paths.primary.type", "fh_linear"))
        if path not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
            raise ValueError(f"dependence_paths.primary.type invalid: {path!r}")

        # Path params: accept nested path_params, plus convenience keys
        primary_dep = _dget(d, "dependence_paths.primary", {})
        if not isinstance(primary_dep, dict):
            raise ValueError("dependence_paths.primary must be a mapping if provided.")
        extra_params = {
            "gamma": primary_dep.get("gamma", None),
            "k": primary_dep.get("k", None),
            "ppf_clip_eps": primary_dep.get("ppf_clip_eps", None),
        }
        path_params = _parse_path_params(primary_dep.get("path_params", {}), extra=extra_params)

        # Lambdas: coarse grid in pipeline config (simulate.py is single-run)
        if "lambdas" in d:
            lambdas = [float(x) for x in (d.get("lambdas") or [])]
        else:
            grid = _dget(d, "dependence_paths.primary.lambda_grid_coarse", None)
            if grid is None:
                grid = _dget(d, "dependence_paths.primary.lambda_grid", None)
            if grid is None:
                grid = d.get("lambda_grid", {"num": 21})
            lambdas = list(
                _parse_lambdas_from_grid(grid, where="dependence_paths.primary.lambda_grid_coarse")
            )

        lambdas = list(_ensure_monotone_increasing(lambdas))

        # Sampling
        n = _i(_dget(d, "sampling.n_per_world", d.get("n", None)), "sampling.n_per_world")
        n_reps = _i(_dget(d, "sampling.n_reps", d.get("n_reps", 1)), "sampling.n_reps")
        seed = _i(_dget(d, "sampling.seed", d.get("seed", 0)), "sampling.seed")
        seed_policy = str(_dget(d, "sampling.seed_policy", d.get("seed_policy", "stable_per_cell")))
        if seed_policy not in ("stable_per_cell", "sequential"):
            raise ValueError(f"sampling.seed_policy invalid: {seed_policy!r}")

        # Numeric / validation policies: allow overrides in simulate.*
        envelope_tol = _f(
            _dget(d, "simulate.envelope_tol", d.get("envelope_tol", 5e-3)), "simulate.envelope_tol"
        )
        hard_fail_on_invalid = _b(
            _dget(d, "simulate.hard_fail_on_invalid", d.get("hard_fail_on_invalid", True)),
            "simulate.hard_fail_on_invalid",
        )
        include_theory_reference = _b(
            _dget(d, "simulate.include_theory_reference", d.get("include_theory_reference", True)),
            "simulate.include_theory_reference",
        )

        prob_tol = _f(_dget(d, "simulate.prob_tol", d.get("prob_tol", 1e-12)), "simulate.prob_tol")
        allow_tiny_negative = _b(
            _dget(d, "simulate.allow_tiny_negative", d.get("allow_tiny_negative", True)),
            "simulate.allow_tiny_negative",
        )
        tiny_negative_eps = _f(
            _dget(d, "simulate.tiny_negative_eps", d.get("tiny_negative_eps", 1e-15)),
            "simulate.tiny_negative_eps",
        )

    else:
        # ----------------------------
        # Legacy schema
        # ----------------------------
        rule = str(d.get("rule", "OR")).upper()
        if rule not in ("OR", "AND"):
            raise ValueError(f"Invalid rule: {rule!r}")

        path = str(d.get("path", "fh_linear"))
        if path not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
            raise ValueError(f"Invalid path: {path!r}")

        base_pp = d.get("path_params", {})
        extra_pp = {
            "gamma": d.get("gamma", None),
            "k": d.get("k", None),
            "ppf_clip_eps": d.get("ppf_clip_eps", None),
        }
        path_params = _parse_path_params(base_pp, extra=extra_pp)

        if "lambdas" in d:
            lambdas = [float(x) for x in (d.get("lambdas") or [])]
        else:
            lg = d.get("lambda_grid", {"num": 21})
            lambdas = list(_parse_lambdas_from_grid(lg, where="lambda_grid"))

        lambdas = list(_ensure_monotone_increasing(lambdas))

        if "n" not in d:
            raise ValueError("Legacy schema requires top-level 'n' (sample size per world).")
        n = _i(d["n"], "n")
        n_reps = _i(d.get("n_reps", 1), "n_reps")
        seed = _i(d.get("seed", 0), "seed")

        seed_policy = str(d.get("seed_policy", "stable_per_cell"))
        if seed_policy not in ("stable_per_cell", "sequential"):
            raise ValueError(f"Invalid seed_policy: {seed_policy!r}")

        envelope_tol = _f(d.get("envelope_tol", 5e-3), "envelope_tol")
        hard_fail_on_invalid = bool(d.get("hard_fail_on_invalid", True))
        include_theory_reference = bool(d.get("include_theory_reference", True))

        prob_tol = _f(d.get("prob_tol", 1e-12), "prob_tol")
        allow_tiny_negative = bool(d.get("allow_tiny_negative", True))
        tiny_negative_eps = _f(d.get("tiny_negative_eps", 1e-15), "tiny_negative_eps")

    # ----------------------------
    # Final validation for core numeric knobs
    # ----------------------------
    if n <= 0:
        raise ValueError("n (sample size per world) must be positive.")
    if n_reps <= 0:
        raise ValueError("n_reps must be positive.")
    if not (0.0 <= prob_tol <= 1e-6):
        raise ValueError(f"prob_tol seems unreasonable: {prob_tol} (expected [0,1e-6])")
    if allow_tiny_negative and not (0.0 < tiny_negative_eps <= 1e-6):
        raise ValueError(
            f"tiny_negative_eps seems unreasonable: {tiny_negative_eps} (expected (0,1e-6])"
        )
    if envelope_tol < 0.0 or not math.isfinite(envelope_tol):
        raise ValueError(f"envelope_tol must be finite and >=0, got {envelope_tol!r}")

    return SimConfig(
        marginals=marg,
        rule=rule,  # type: ignore
        lambdas=list(lambdas),
        n=int(n),
        n_reps=int(n_reps),
        seed=int(seed),
        path=path,  # type: ignore
        path_params=dict(path_params),
        seed_policy=seed_policy,  # type: ignore
        envelope_tol=float(envelope_tol),
        hard_fail_on_invalid=bool(hard_fail_on_invalid),
        prob_tol=float(prob_tol),
        allow_tiny_negative=bool(allow_tiny_negative),
        tiny_negative_eps=float(tiny_negative_eps),
        include_theory_reference=bool(include_theory_reference),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entrypoint for simulate.py.

    Enterprise-grade behavior:
      - Strong config error messages and non-zero exit codes
      - Deterministic outputs (cfg-driven), with atomic writes to avoid partial files
      - Optional "out_dir" that writes a complete artifact bundle:
          sim_long.csv, sim_summary.csv, diagnostics.json, config_resolved.json, manifest.json
      - Still supports explicit --out_csv / --out_summary_csv for backwards compatibility
      - Prints a stable JSON diagnostics blob to stdout (useful for CI)
    """
    import argparse
    import hashlib
    import json
    import logging
    import platform
    import subprocess
    import sys
    import time
    from datetime import datetime
    from pathlib import Path

    LOG = logging.getLogger("correlation_cliff.simulate")

    # ----------------------------
    # local helpers (CLI-only)
    # ----------------------------
    def _now_stamp() -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _atomic_write_text(path: Path, text: str) -> None:
        _ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(path)

    def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
        _ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)

    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _maybe_git_commit(repo_root: Path) -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or None
        except Exception:
            return None

    def _repo_root_guess() -> Path:
        # experiments/correlation_cliff/simulate.py -> repo root is usually 2 levels up
        here = Path(__file__).resolve()
        parents = list(here.parents)
        if len(parents) >= 3:
            return parents[2]
        return here.parent

    # ----------------------------
    # CLI
    # ----------------------------
    ap = argparse.ArgumentParser(description="Run correlation cliff simulation grid.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Write a full artifact bundle to this directory "
            "(sim_long.csv, sim_summary.csv, diagnostics.json, manifest.json). "
            "If not provided, you may use --out_csv/--out_summary_csv, "
            "or the CLI will create ./artifacts/<UTCstamp>/ by default."
        ),
    )
    ap.add_argument(
        "--out_csv", type=str, default=None, help="Write replicate-level rows to CSV (legacy)."
    )
    ap.add_argument(
        "--out_summary_csv",
        type=str,
        default=None,
        help="Write per-lambda summary to CSV (legacy).",
    )
    ap.add_argument("--print_head", type=int, default=5, help="Print first N rows of summary.")
    ap.add_argument(
        "--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)."
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If out_dir exists and is non-empty, overwrite files (default: create a new timestamped subdir).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.config is None:
        print(
            "ERROR: Please provide --config path/to/config.yaml (or call simulate_grid() from Python).",
            file=sys.stderr,
        )
        return 2

    run_started_utc = datetime.utcnow().isoformat() + "Z"
    t0 = time.time()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        print(f"ERROR: config path does not exist: {str(config_path)!r}", file=sys.stderr)
        return 2

    # Resolve out_dir policy
    out_dir: Optional[Path] = Path(args.out_dir).expanduser() if args.out_dir else None
    legacy_out_csv: Optional[Path] = Path(args.out_csv).expanduser() if args.out_csv else None
    legacy_out_sum: Optional[Path] = (
        Path(args.out_summary_csv).expanduser() if args.out_summary_csv else None
    )

    if out_dir is None and legacy_out_csv is None and legacy_out_sum is None:
        out_dir = Path.cwd() / "artifacts" / _now_stamp()

    if out_dir is not None:
        # overwrite policy: if dir exists + non-empty and overwrite not requested, nest a timestamped run
        if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
            out_dir = out_dir / f"run_{_now_stamp()}"
        _ensure_dir(out_dir)

    # ----------------------------
    # Load + build cfg
    # ----------------------------
    try:
        cfg_dict = _load_yaml(str(config_path))
        cfg = _cfg_from_dict(cfg_dict)
        _validate_cfg(cfg)
    except Exception as e:
        LOG.exception("Config loading/validation failed.")
        print(f"ERROR: config loading/validation failed: {e}", file=sys.stderr)
        return 2

    # ----------------------------
    # Run simulation
    # ----------------------------
    try:
        LOG.info(
            "Running simulate_grid: n=%s n_reps=%s lambdas=%s path=%s rule=%s seed_policy=%s",
            cfg.n,
            cfg.n_reps,
            len(cfg.lambdas),
            cfg.path,
            cfg.rule,
            cfg.seed_policy,
        )
        df_long = simulate_grid(cfg)
        df_sum = summarize_simulation(df_long)
    except Exception as e:
        LOG.exception("Simulation failed.")
        print(f"ERROR: simulation failed: {e}", file=sys.stderr)
        return 1

    # ----------------------------
    # Write outputs (atomic)
    # ----------------------------
    file_hashes: Dict[str, str] = {}
    outputs: Dict[str, str] = {}

    try:
        if out_dir is not None:
            long_path = out_dir / "sim_long.csv"
            sum_path = out_dir / "sim_summary.csv"
            _atomic_write_csv(df_long, long_path)
            _atomic_write_csv(df_sum, sum_path)
            outputs["sim_long.csv"] = str(long_path)
            outputs["sim_summary.csv"] = str(sum_path)
            file_hashes["sim_long.csv"] = _sha256_file(long_path)
            file_hashes["sim_summary.csv"] = _sha256_file(sum_path)

            # Save config snapshot as actually used by this run (no lies)
            cfg_snapshot = {
                "config_path": str(config_path),
                "marginals": {
                    "w0": {"pA": float(cfg.marginals.w0.pA), "pB": float(cfg.marginals.w0.pB)},
                    "w1": {"pA": float(cfg.marginals.w1.pA), "pB": float(cfg.marginals.w1.pB)},
                },
                "rule": str(cfg.rule),
                "path": str(cfg.path),
                "path_params": dict(cfg.path_params),
                "seed_policy": str(cfg.seed_policy),
                "seed": int(cfg.seed),
                "n": int(cfg.n),
                "n_reps": int(cfg.n_reps),
                "lambda_points": int(len(cfg.lambdas)),
                "lambdas": [float(x) for x in cfg.lambdas],
                "envelope_tol": float(cfg.envelope_tol),
                "hard_fail_on_invalid": bool(cfg.hard_fail_on_invalid),
                "prob_tol": float(cfg.prob_tol),
                "allow_tiny_negative": bool(cfg.allow_tiny_negative),
                "tiny_negative_eps": float(cfg.tiny_negative_eps),
                "include_theory_reference": bool(cfg.include_theory_reference),
            }
            _atomic_write_text(
                out_dir / "config_resolved.json",
                json.dumps(cfg_snapshot, indent=2, sort_keys=True) + "\n",
            )
            outputs["config_resolved.json"] = str(out_dir / "config_resolved.json")
            file_hashes["config_resolved.json"] = _sha256_file(out_dir / "config_resolved.json")

        # Legacy explicit targets still supported
        if legacy_out_csv is not None:
            _ensure_dir(legacy_out_csv.parent if legacy_out_csv.parent.as_posix() else Path("."))
            _atomic_write_csv(df_long, legacy_out_csv)
            outputs["out_csv"] = str(legacy_out_csv)
        if legacy_out_sum is not None:
            _ensure_dir(legacy_out_sum.parent if legacy_out_sum.parent.as_posix() else Path("."))
            _atomic_write_csv(df_sum, legacy_out_sum)
            outputs["out_summary_csv"] = str(legacy_out_sum)

    except Exception as e:
        LOG.exception("Writing outputs failed.")
        print(f"ERROR: writing outputs failed: {e}", file=sys.stderr)
        return 1

    # ----------------------------
    # Diagnostics (stable JSON)
    # ----------------------------
    vio_rate = (
        float(df_long["JC_env_violation"].mean())
        if "JC_env_violation" in df_long.columns and not df_long.empty
        else float("nan")
    )
    invalid_cols = [c for c in df_long.columns if c.startswith("invalid_joint_w")]
    invalid_rates = {c: float(df_long[c].fillna(False).astype(bool).mean()) for c in invalid_cols}

    diag = {
        "run_started_utc": run_started_utc,
        "run_finished_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": float(time.time() - t0),
        "rows": int(len(df_long)),
        "lambda_points": int(len(set(df_long["lambda"]))) if "lambda" in df_long.columns else 0,
        "env_violation_rate": vio_rate,
        "seed_policy": str(cfg.seed_policy),
        "seed": int(cfg.seed),
        "path": str(cfg.path),
        "rule": str(cfg.rule),
        **{f"{k}_rate": v for k, v in invalid_rates.items()},
    }

    # Print diagnostics to stdout (machine-readable)
    print(json.dumps(diag, sort_keys=True))

    # Persist diagnostics + manifest if out_dir is set
    if out_dir is not None:
        try:
            _atomic_write_text(
                out_dir / "diagnostics.json", json.dumps(diag, indent=2, sort_keys=True) + "\n"
            )
            outputs["diagnostics.json"] = str(out_dir / "diagnostics.json")
            file_hashes["diagnostics.json"] = _sha256_file(out_dir / "diagnostics.json")

            repo_root = _repo_root_guess()
            manifest = {
                "run_started_utc": run_started_utc,
                "run_finished_utc": datetime.utcnow().isoformat() + "Z",
                "elapsed_seconds": float(time.time() - t0),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": getattr(np, "__version__", None),
                "pandas": getattr(pd, "__version__", None),
                "git_commit": _maybe_git_commit(repo_root),
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "outputs": outputs,
                "file_sha256": file_hashes,
                "diagnostics": diag,
            }
            _atomic_write_text(
                out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            outputs["manifest.json"] = str(out_dir / "manifest.json")
            file_hashes["manifest.json"] = _sha256_file(out_dir / "manifest.json")
        except Exception:
            # Manifest should never crash a successful simulation; log but keep exit code 0.
            LOG.exception("Failed to write diagnostics/manifest bundle.")

    # ----------------------------
    # Optional pretty summary head
    # ----------------------------
    if args.print_head and int(args.print_head) > 0:
        with pd.option_context("display.width", 160, "display.max_columns", 200):
            print(df_sum.head(int(args.print_head)))

    # Helpful user-facing line (non-JSON, so send to stderr)
    if out_dir is not None:
        print(f"[correlation_cliff.simulate] outputs written to: {out_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
