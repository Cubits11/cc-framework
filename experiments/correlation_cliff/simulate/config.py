# experiments/correlation_cliff/simulate/config.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Protocol, Sequence, Tuple

import numpy as np

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
# Internal coercion helpers (avoid importing utils.py)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class _CoercedWorldMarginals:
    pA: float
    pB: float

    def validate(self) -> None:
        if not math.isfinite(self.pA) or not (0.0 <= self.pA <= 1.0):
            raise ConfigError(f"marginals.pA must be finite in [0,1], got {self.pA!r}")
        if not math.isfinite(self.pB) or not (0.0 <= self.pB <= 1.0):
            raise ConfigError(f"marginals.pB must be finite in [0,1], got {self.pB!r}")


@dataclass(frozen=True)
class _CoercedTwoWorldMarginals:
    w0: _CoercedWorldMarginals
    w1: _CoercedWorldMarginals

    def validate(self) -> None:
        self.w0.validate()
        self.w1.validate()


def _is_finite_real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _strict_int(x: Any, name: str, *, min_value: int | None = None) -> int:
    # Reject bool explicitly (bool is subclass of int).
    if isinstance(x, bool):
        raise ConfigError(f"{name} must be an int, got bool {x!r}")
    if isinstance(x, (float, np.floating)):
        raise ConfigError(f"{name} must be an int (no silent coercion), got {x!r}")
    if not isinstance(x, (int, np.integer)):
        raise ConfigError(f"{name} must be an int, got {type(x).__name__}={x!r}")
    if min_value is not None and int(x) < min_value:
        raise ConfigError(f"{name} must be >= {min_value}, got {x}")
    return int(x)


def _canonical_lambda_keys(
    lambdas: Sequence[Any], *, decimals: int = 12
) -> Tuple[List[float], List[float]]:
    """
    Return (vals, keys) where:
      - vals are float-cast lambdas in original order
      - keys are rounded float keys (stable across YAML/JSON parsing)
    Raises ConfigError for non-finite, out-of-range, or duplicates-after-rounding.
    """
    if len(lambdas) == 0:
        raise ConfigError("lambdas must be a non-empty sequence")

    vals: List[float] = []
    keys: List[float] = []
    for i, lam in enumerate(lambdas):
        if isinstance(lam, bool):
            raise ConfigError(f"lambdas[{i}] must be a real number in [0,1], got bool {lam!r}")
        try:
            lf = float(lam)
        except Exception as e:
            raise ConfigError(f"lambdas[{i}] must be a real number in [0,1], got {lam!r}") from e
        if not math.isfinite(lf) or not (0.0 <= lf <= 1.0):
            raise ConfigError(f"lambdas[{i}] must be finite and in [0,1], got {lam!r}")
        vals.append(lf)
        keys.append(round(lf, decimals))

    if len(set(keys)) != len(keys):
        raise ConfigError(
            f"cfg.lambdas contains duplicates (after rounding to {decimals} decimals). Got: {list(lambdas)!r}"
        )
    return vals, keys


def _deep_no_nan_inf(x: Any, path: str) -> None:
    """
    Reject NaN/Inf nested anywhere in path_params.
    Allows scalars, lists/tuples, dict-like mappings.
    """
    if x is None:
        return
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        if not math.isfinite(float(x)):
            raise ConfigError(f"path_params contains non-finite number at {path}: {x!r}")
        return
    if isinstance(x, str):
        return
    if isinstance(x, Mapping):
        for k, v in x.items():
            if not isinstance(k, str):
                raise ConfigError(
                    f"path_params keys must be str; found key={k!r} ({type(k).__name__}) at {path}"
                )
            _deep_no_nan_inf(v, f"{path}.{k}")
        return
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _deep_no_nan_inf(v, f"{path}[{i}]")
        return
    # If you want to allow arbitrary objects, remove this.
    raise ConfigError(f"path_params contains unsupported type at {path}: {type(x).__name__}")


def _coerce_marginals(m: Any) -> _TwoWorldMarginals:
    """
    Accept either:
      - a duck-typed object with .w0/.w1 each having .pA/.pB, OR
      - a mapping like {"w0": {"pA": .., "pB": ..}, "w1": {...}}
    Returns an object satisfying the _TwoWorldMarginals protocol.
    """
    # Mapping case
    if isinstance(m, Mapping):
        try:
            w0 = m["w0"]
            w1 = m["w1"]
        except Exception as e:
            raise ConfigError("marginals mapping must contain keys 'w0' and 'w1'") from e

        def _get(w: Any, key: str, tag: str) -> float:
            if isinstance(w, Mapping):
                if key not in w:
                    raise ConfigError(f"marginals.{tag} missing key {key!r}")
                val = w[key]
            else:
                if not hasattr(w, key):
                    raise ConfigError(f"marginals.{tag} missing attribute {key!r}")
                val = getattr(w, key)
            if isinstance(val, bool):
                raise ConfigError(
                    f"marginals.{tag}.{key} must be finite in [0,1], got bool {val!r}"
                )
            vf = float(val)
            if not math.isfinite(vf) or not (0.0 <= vf <= 1.0):
                raise ConfigError(f"marginals.{tag}.{key} must be finite in [0,1], got {val!r}")
            return vf

        w0c = _CoercedWorldMarginals(pA=_get(w0, "pA", "w0"), pB=_get(w0, "pB", "w0"))
        w1c = _CoercedWorldMarginals(pA=_get(w1, "pA", "w1"), pB=_get(w1, "pB", "w1"))
        return _CoercedTwoWorldMarginals(w0=w0c, w1=w1c)

    # Duck-typed object case
    if not (hasattr(m, "w0") and hasattr(m, "w1")):
        raise ConfigError(
            "marginals must have .w0/.w1 each with .pA/.pB fields "
            "(cli builds this as utils.TwoWorldMarginals), or be a mapping with w0/w1."
        )

    # Validate without coercing (keep the caller's object)
    for tag in ("w0", "w1"):
        w = getattr(m, tag)
        for fld in ("pA", "pB"):
            if not hasattr(w, fld):
                raise ConfigError(f"marginals.{tag} missing field {fld!r}")
            v = getattr(w, fld)
            if isinstance(v, bool):
                raise ConfigError(f"marginals.{tag}.{fld} must be finite in [0,1], got bool {v!r}")
            vf = float(v)
            if not math.isfinite(vf) or not (0.0 <= vf <= 1.0):
                raise ConfigError(f"marginals.{tag}.{fld} must be finite in [0,1], got {v!r}")

    return m


# ---------------------------------------------------------------------
# Canonical config used by core.py / cli.py / tests
# ---------------------------------------------------------------------
@dataclass
class SimConfig:
    """
    Simulation configuration for Correlation Cliff experiments.

    batch_sampling:
      - If True and seed_policy == "sequential", draws all reps at once per lambda/world.
        This is faster but produces a different RNG stream than sequential per-rep sampling.
      - Ignored for seed_policy == "stable_per_cell".
      - Set to False to preserve legacy sequential stream behavior.
    """
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
    batch_sampling: bool = True

    # internal cache for stable_per_cell mapping
    _lambda_index_map: Dict[float, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Normalize string enums
        if isinstance(self.rule, str):
            self.rule = self.rule.upper()  # type: ignore[assignment]
        if isinstance(self.path, str):
            self.path = self.path.lower()  # type: ignore[assignment]
        if isinstance(self.seed_policy, str):
            self.seed_policy = self.seed_policy.lower()  # type: ignore[assignment]

        # Normalize containers (no validator side-effects)
        if self.path_params is None:  # type: ignore[redundant-expr]
            self.path_params = {}
        if not isinstance(self.lambdas, list):
            self.lambdas = list(self.lambdas)  # type: ignore[arg-type]

        # Coerce/validate marginals early so the object satisfies the protocol downstream.
        self.marginals = _coerce_marginals(self.marginals)  # type: ignore[assignment]

        validate_cfg(self)

        # Build an order-invariant lambda index mapping (for stable_per_cell seeding).
        _, keys = _canonical_lambda_keys(self.lambdas, decimals=12)
        uniq = sorted(keys)
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


def validate_cfg(cfg: SimConfig) -> None:
    # --- rule/path/seed_policy ---
    if cfg.rule not in ("OR", "AND"):
        raise ConfigError(f"rule must be OR/AND, got {cfg.rule!r}")
    if cfg.path not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
        raise ConfigError(f"path invalid: {cfg.path!r}")
    if cfg.seed_policy not in ("stable_per_cell", "sequential"):
        raise ConfigError(f"seed_policy invalid: {cfg.seed_policy!r}")

    # --- sizes (strict: reject bool) ---
    _strict_int(cfg.n, "n", min_value=1)
    _strict_int(cfg.n_reps, "n_reps", min_value=1)
    _strict_int(cfg.seed, "seed", min_value=0)

    # --- tolerances ---
    if (
        not _is_finite_real(cfg.envelope_tol)
        or float(cfg.envelope_tol) < 0.0
        or float(cfg.envelope_tol) > 1e-1
    ):
        raise ConfigError(
            f"envelope_tol must be finite and reasonably small, got {cfg.envelope_tol!r}"
        )

    if not _is_finite_real(cfg.prob_tol) or float(cfg.prob_tol) < 0.0 or float(cfg.prob_tol) > 1e-3:
        raise ConfigError(f"prob_tol must be finite and reasonably small, got {cfg.prob_tol!r}")

    if not isinstance(cfg.allow_tiny_negative, bool):
        raise ConfigError(f"allow_tiny_negative must be bool, got {cfg.allow_tiny_negative!r}")
    if cfg.allow_tiny_negative:
        if not _is_finite_real(cfg.tiny_negative_eps) or not (
            0.0 < float(cfg.tiny_negative_eps) <= 1e-6
        ):
            raise ConfigError(
                f"tiny_negative_eps must be finite in (0,1e-6], got {cfg.tiny_negative_eps!r}"
            )

    if not isinstance(cfg.hard_fail_on_invalid, bool):
        raise ConfigError(f"hard_fail_on_invalid must be bool, got {cfg.hard_fail_on_invalid!r}")
    if not isinstance(cfg.include_theory_reference, bool):
        raise ConfigError(
            f"include_theory_reference must be bool, got {cfg.include_theory_reference!r}"
        )
    if not isinstance(cfg.batch_sampling, bool):
        raise ConfigError(f"batch_sampling must be bool, got {cfg.batch_sampling!r}")

    # --- lambdas ---
    if not isinstance(cfg.lambdas, list) or len(cfg.lambdas) == 0:
        raise ConfigError("lambdas must be a non-empty list")
    vals, _ = _canonical_lambda_keys(cfg.lambdas, decimals=12)

    # Sequential policy: enforce monotone non-decreasing (repeatable sweep semantics).
    if cfg.seed_policy == "sequential":
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                raise ConfigError(
                    f"lambdas must be non-decreasing for seed_policy='sequential'; "
                    f"lambdas[{i - 1}]={vals[i - 1]} > lambdas[{i}]={vals[i]}"
                )

    # --- marginals (already coerced/validated in __post_init__) ---
    # (Keep a cheap sanity check here)
    for tag in ("w0", "w1"):
        w = getattr(cfg.marginals, tag)
        pA = float(w.pA)
        pB = float(w.pB)
        if not math.isfinite(pA) or not (0.0 <= pA <= 1.0):
            raise ConfigError(f"marginals.{tag}.pA must be finite in [0,1], got {w.pA!r}")
        if not math.isfinite(pB) or not (0.0 <= pB <= 1.0):
            raise ConfigError(f"marginals.{tag}.pB must be finite in [0,1], got {w.pB!r}")

    # --- path_params ---
    if not isinstance(cfg.path_params, Mapping):
        raise ConfigError(
            f"path_params must be a dict-like mapping, got {type(cfg.path_params).__name__}"
        )
    _deep_no_nan_inf(cfg.path_params, "path_params")


__all__ = [
    "ConfigError",
    "Rule",
    "Path",
    "SeedPolicy",
    "SimConfig",
    "validate_cfg",
]
