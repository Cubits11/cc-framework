# experiments/correlation_cliff/simulate/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol

import math


# ---------------------------------------------------------------------
# Public types expected across simulate/*
# ---------------------------------------------------------------------
Rule = Literal["OR", "AND"]
Path = Literal["fh_linear", "fh_power", "fh_scurve", "gaussian_tau"]
SeedPolicy = Literal["stable_per_cell", "sequential"]


class ConfigError(ValueError):
    """User-fixable configuration error."""


# ---------------------------------------------------------------------
# Protocols to avoid circular imports (cli builds these from utils.py)
# ---------------------------------------------------------------------
class _WorldMarginals(Protocol):
    pA: float
    pB: float


class _TwoWorldMarginals(Protocol):
    w0: _WorldMarginals
    w1: _WorldMarginals


# ---------------------------------------------------------------------
# Canonical config used by core.py / cli.py / tests
# ---------------------------------------------------------------------
@dataclass
class SimConfig:
    marginals: _TwoWorldMarginals
    rule: Rule
    lambdas: List[float]

    n: int
    n_reps: int
    seed: int

    path: Path
    path_params: Dict[str, Any] = field(default_factory=dict)

    seed_policy: SeedPolicy = "stable_per_cell"

    envelope_tol: float = 5e-3
    hard_fail_on_invalid: bool = True

    prob_tol: float = 1e-12
    allow_tiny_negative: bool = True
    tiny_negative_eps: float = 1e-15

    include_theory_reference: bool = True

    # internal cache for stable_per_cell mapping
    _lambda_index_map: Dict[float, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.rule, str):
            self.rule = self.rule.upper()  # type: ignore[assignment]
        if isinstance(self.path, str):
            self.path = self.path.lower()  # type: ignore[assignment]
        validate_cfg(self)
        # Build an order-invariant lambda index mapping (for stable_per_cell seeding).
        # Use rounded keys to make float parsing stable across YAML/JSON representations.
        keys = [round(float(x), 12) for x in self.lambdas]
        uniq = sorted(set(keys))
        if len(uniq) != len(keys):
            # duplicates after rounding -> ambiguous seeding
            raise ConfigError(
                "cfg.lambdas contains duplicates (after rounding to 12 decimals). "
                f"Got: {self.lambdas!r}"
            )
        self._lambda_index_map = {k: i for i, k in enumerate(uniq)}

    def lambda_index_for_seed(self, lam: float) -> int:
        k = round(float(lam), 12)
        try:
            return int(self._lambda_index_map[k])
        except KeyError as e:
            raise ConfigError(
                f"lambda_index_for_seed: lambda {lam!r} not present in cfg.lambdas. "
                f"Known keys={sorted(self._lambda_index_map.keys())[:10]}..."
            ) from e


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def validate_cfg(cfg: SimConfig) -> None:
    # --- rule/path/seed_policy ---
    if cfg.rule not in ("OR", "AND"):
        raise ConfigError(f"rule must be OR/AND, got {cfg.rule!r}")
    if cfg.path not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
        raise ConfigError(f"path invalid: {cfg.path!r}")
    if cfg.seed_policy not in ("stable_per_cell", "sequential"):
        raise ConfigError(f"seed_policy invalid: {cfg.seed_policy!r}")

    # --- sizes ---
    if not isinstance(cfg.n, int) or cfg.n <= 0:
        raise ConfigError(f"n must be positive int, got {cfg.n!r}")
    if not isinstance(cfg.n_reps, int) or cfg.n_reps <= 0:
        raise ConfigError(f"n_reps must be positive int, got {cfg.n_reps!r}")
    if not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ConfigError(f"seed must be int >= 0, got {cfg.seed!r}")

    # --- tolerances ---
    if not _finite(cfg.envelope_tol) or float(cfg.envelope_tol) < 0.0 or float(cfg.envelope_tol) > 1e-1:
        raise ConfigError(f"envelope_tol must be finite and reasonably small, got {cfg.envelope_tol!r}")

    if not _finite(cfg.prob_tol) or float(cfg.prob_tol) < 0.0 or float(cfg.prob_tol) > 1e-3:
        raise ConfigError(f"prob_tol must be finite and reasonably small, got {cfg.prob_tol!r}")

    if not isinstance(cfg.allow_tiny_negative, bool):
        raise ConfigError(f"allow_tiny_negative must be bool, got {cfg.allow_tiny_negative!r}")
    if cfg.allow_tiny_negative:
        if not _finite(cfg.tiny_negative_eps) or not (0.0 < float(cfg.tiny_negative_eps) <= 1e-6):
            raise ConfigError(f"tiny_negative_eps must be finite in (0,1e-6], got {cfg.tiny_negative_eps!r}")

    if not isinstance(cfg.hard_fail_on_invalid, bool):
        raise ConfigError(f"hard_fail_on_invalid must be bool, got {cfg.hard_fail_on_invalid!r}")
    if not isinstance(cfg.include_theory_reference, bool):
        raise ConfigError(f"include_theory_reference must be bool, got {cfg.include_theory_reference!r}")

    # --- lambdas ---
    if not isinstance(cfg.lambdas, list) or len(cfg.lambdas) == 0:
        raise ConfigError("lambdas must be a non-empty list")
    last = None
    for i, lam in enumerate(cfg.lambdas):
        lf = float(lam)
        if not math.isfinite(lf) or not (0.0 <= lf <= 1.0):
            raise ConfigError(f"lambdas[{i}] must be finite and in [0,1], got {lam!r}")
        if last is not None and lf < last:
            raise ConfigError(f"lambdas must be non-decreasing; lambdas[{i-1}]={last} > lambdas[{i}]={lf}")
        last = lf

    # --- marginals (duck-typed) ---
    try:
        w0 = cfg.marginals.w0
        w1 = cfg.marginals.w1
        for tag, w in (("w0", w0), ("w1", w1)):
            pA = float(w.pA)
            pB = float(w.pB)
            if not math.isfinite(pA) or not (0.0 <= pA <= 1.0):
                raise ConfigError(f"marginals.{tag}.pA must be finite in [0,1], got {w.pA!r}")
            if not math.isfinite(pB) or not (0.0 <= pB <= 1.0):
                raise ConfigError(f"marginals.{tag}.pB must be finite in [0,1], got {w.pB!r}")
    except AttributeError as e:
        raise ConfigError(
            "marginals must have .w0/.w1 each with .pA/.pB fields "
            "(cli builds this as utils.TwoWorldMarginals)."
        ) from e

    # path_params
    if cfg.path_params is None:
        cfg.path_params = {}
    if not isinstance(cfg.path_params, Mapping):
        raise ConfigError(f"path_params must be a dict-like mapping, got {type(cfg.path_params).__name__}")


__all__ = [
    "ConfigError",
    "Rule",
    "Path",
    "SeedPolicy",
    "SimConfig",
    "validate_cfg",
]
