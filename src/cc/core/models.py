# src/cc/core/models.py
"""
Module: core.models (data layer for CC framework)
Purpose: Strongly-typed, versioned, and serializable data models used across the
         two-world protocol, calibration, analysis, and reporting.

Author: Pranav Bhave
Dates:
  - 2025-08-31: original
  - 2025-09-28: ultimate upgrade (validation, slots, versioned schemas, robust
                (de)serialization, hashing helpers, immutability patterns, numpy
                safety, convenience constructors)

Design notes
------------
- **Backwards compatible** field names for previously persisted payloads.
- Dataclasses use `slots=True` to reduce memory overhead and catch typos.
- All models provide: `.to_dict()`, `.to_json()`, `from_dict()`, `clone(**kw)`.
- Robust JSON-safety: numpy scalars/arrays converted to native/`list`.
- Lightweight validation in `__post_init__` (no heavy deps or pydantic).
- Optional content hashing utilities to ensure referential integrity.

Dependencies: standard library + numpy (for arrays in results).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np


# =============================================================================
# Helpers
# =============================================================================

T = TypeVar("T")

_SCHEMA_VERSION = "2.0"  # bump when changing wire schema semantics


def _now_unix() -> float:
    return float(time.time())


def _iso_from_unix(ts: float) -> str:
    tm = time.gmtime(ts)
    ms = int((ts - int(ts)) * 1000)
    return f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}T{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{ms:03d}Z"


def _to_native(x: Any) -> Any:
    """Convert common non-JSON-safe objects to JSON-safe types."""
    # numpy scalars
    if isinstance(x, (np.generic,)):
        return x.item()
    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    # pathlib paths
    if isinstance(x, Path):
        return str(x)
    # sets/tuples
    if isinstance(x, (set, tuple)):
        return list(x)
    # bytes/bytearray -> hex
    if isinstance(x, (bytes, bytearray)):
        return x.hex()
    return x


def _json_dumpable(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-ish conversion to JSON-serializable dict without heavy recursion."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _json_dumpable(v)
        elif isinstance(v, list):
            out[k] = [_to_native(i) if not isinstance(i, dict) else _json_dumpable(i) for i in v]
        else:
            out[k] = _to_native(v)
    return out


def _stable_hash(*parts: Any) -> str:
    """Stable sha256 hexdigest over provided parts after JSON canonicalization."""
    payload = json.dumps([_json_dumpable({"v": p}) for p in parts], sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _validate_prob(p: float, name: str) -> None:
    if not (0.0 <= p <= 1.0) or math.isnan(p):
        raise ValueError(f"{name} must be in [0,1], got {p!r}")


# Base mixin to unify (de)serialization & cloning
class _ModelIO:
    __slots__ = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict (including schema_version)."""
        raw = {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]
        raw["schema_version"] = _SCHEMA_VERSION
        return _json_dumpable(raw)

    def to_json(self, *, compact: bool = True) -> str:
        """Serialize to JSON string."""
        d = self.to_dict()
        if compact:
            return json.dumps(d, sort_keys=True, separators=(",", ":"))
        return json.dumps(d, sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls: Type[T], d: Mapping[str, Any]) -> T:
        """Construct from a (possibly JSON-loaded) mapping. Ignores unknown keys."""
        # Accept legacy payloads that may not carry schema_version
        fields = getattr(cls, "__dataclass_fields__")  # type: ignore[attr-defined]
        kwargs = {}
        for name in fields:
            if name in d:
                kwargs[name] = d[name]
        return cls(**kwargs)  # type: ignore[misc]

    def clone(self: T, **overrides: Any) -> T:
        """Return a copy with fields overridden."""
        return replace(self, **overrides)


# =============================================================================
# Models
# =============================================================================

@dataclass(slots=True)
class AttackResult(_ModelIO):
    """
    Result of a single attack session in two-world protocol.

    Backward-compatible fields with additional optional metadata.
    """
    world_bit: int  # 0 (baseline) or 1 (guardrail-enabled)
    success: bool
    attack_id: str
    transcript_hash: str
    guardrails_applied: str
    rng_seed: int
    timestamp: float
    # --- optional / metadata
    session_id: str = ""
    attack_strategy: str = ""
    utility_score: Optional[float] = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    iso_time: Optional[str] = None
    schema_version: str = field(init=False, default=_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        if self.world_bit not in (0, 1):
            raise ValueError(f"world_bit must be 0 or 1, got {self.world_bit}")
        if self.utility_score is not None and math.isnan(self.utility_score):
            self.utility_score = None
        if self.timestamp <= 0:
            # if caller forgot, stamp now
            self.timestamp = _now_unix()
        if not self.iso_time:
            self.iso_time = _iso_from_unix(self.timestamp)
        # normalize common string fields
        self.attack_id = str(self.attack_id)
        self.guardrails_applied = str(self.guardrails_applied)
        self.session_id = str(self.session_id)
        self.attack_strategy = str(self.attack_strategy)

    # Convenience factory
    @classmethod
    def make(
        cls,
        *,
        world_bit: int,
        success: bool,
        attack_id: str,
        transcript: Union[str, bytes],
        guardrails_applied: str,
        rng_seed: int,
        timestamp: Optional[float] = None,
        session_id: str = "",
        attack_strategy: str = "",
        utility_score: Optional[float] = None,
    ) -> "AttackResult":
        ts = _now_unix() if timestamp is None else float(timestamp)
        thash = _stable_hash(transcript)
        return cls(
            world_bit=world_bit,
            success=success,
            attack_id=str(attack_id),
            transcript_hash=thash,
            guardrails_applied=str(guardrails_applied),
            rng_seed=int(rng_seed),
            timestamp=ts,
            session_id=str(session_id),
            attack_strategy=str(attack_strategy),
            utility_score=float(utility_score) if utility_score is not None else None,
        )


@dataclass(slots=True)
class GuardrailSpec(_ModelIO):
    """Specification for a guardrail configuration."""
    name: str
    params: Dict[str, Any]
    calibration_fpr_target: float = 0.05
    calibration_data_hash: str = ""
    version: str = "1.0"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    config_hash: str = ""

    def __post_init__(self) -> None:
        # Ensure dict for params
        if self.params is None:
            self.params = {}
        if not isinstance(self.params, dict):
            self.params = dict(self.params)  # type: ignore[arg-type]
        _validate_prob(self.calibration_fpr_target, "calibration_fpr_target")
        # Compute config hash if missing (stable over name+version+params)
        if not self.config_hash:
            self.config_hash = _stable_hash(self.name, self.version, self.params)

    @classmethod
    def from_name_params(
        cls, name: str, params: Optional[Mapping[str, Any]] = None, *, target_fpr: float = 0.05, version: str = "1.0"
    ) -> "GuardrailSpec":
        return cls(name=str(name), params=dict(params or {}), calibration_fpr_target=float(target_fpr), version=str(version))


@dataclass(slots=True)
class WorldConfig(_ModelIO):
    """Configuration for a world in two-world protocol."""
    world_id: int  # 0 or 1
    guardrail_stack: List[GuardrailSpec]
    utility_profile: Dict[str, Any]
    env_hash: str = ""
    baseline_success_rate: float = 0.6

    def __post_init__(self) -> None:
        if self.world_id not in (0, 1):
            raise ValueError(f"world_id must be 0 or 1, got {self.world_id}")
        if self.guardrail_stack is None:
            self.guardrail_stack = []
        # defensive copy & type
        self.guardrail_stack = [gr if isinstance(gr, GuardrailSpec) else GuardrailSpec.from_dict(gr)  # type: ignore[arg-type]
                                for gr in self.guardrail_stack]
        if self.utility_profile is None:
            self.utility_profile = {}
        _validate_prob(max(0.0, min(1.0, float(self.baseline_success_rate))), "baseline_success_rate")

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        # Ensure nested specs serialize via their to_dict()
        d["guardrail_stack"] = [gr.to_dict() for gr in self.guardrail_stack]
        return d


@dataclass(slots=True)
class ExperimentConfig(_ModelIO):
    """Complete experiment configuration."""
    experiment_id: str
    n_sessions: int
    attack_strategies: List[str]
    guardrail_configs: Dict[str, List[GuardrailSpec]]
    utility_target: float = 0.9
    utility_tolerance: float = 0.02
    random_seed: int = 42
    created_at: float = field(default_factory=_now_unix)
    iso_time: str = field(init=False)

    def __post_init__(self) -> None:
        if self.n_sessions <= 0:
            raise ValueError("n_sessions must be > 0")
        if not self.attack_strategies:
            raise ValueError("attack_strategies cannot be empty")
        if not isinstance(self.guardrail_configs, dict) or not self.guardrail_configs:
            raise ValueError("guardrail_configs must be a non-empty dict")
        self.iso_time = _iso_from_unix(self.created_at)
        _validate_prob(self.utility_target, "utility_target")
        if not (0.0 <= self.utility_tolerance <= 0.5):
            raise ValueError("utility_tolerance must be in [0, 0.5]")
        # normalize nested specs
        normalized: Dict[str, List[GuardrailSpec]] = {}
        for k, v in self.guardrail_configs.items():
            normalized[k] = [gr if isinstance(gr, GuardrailSpec) else GuardrailSpec.from_dict(gr)  # type: ignore[arg-type]
                             for gr in (v or [])]
        self.guardrail_configs = normalized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "n_sessions": int(self.n_sessions),
            "attack_strategies": list(self.attack_strategies),
            "guardrail_configs": {k: [gr.to_dict() for gr in v] for k, v in self.guardrail_configs.items()},
            "utility_target": float(self.utility_target),
            "utility_tolerance": float(self.utility_tolerance),
            "random_seed": int(self.random_seed),
            "created_at": float(self.created_at),
            "iso_time": self.iso_time,
        }


@dataclass(slots=True)
class CCResult(_ModelIO):
    """Results of CC analysis."""
    j_empirical: float
    cc_max: float
    delta_add: float
    cc_multiplicative: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    bootstrap_samples: Optional[np.ndarray] = None
    n_sessions: int = 0
    ci_method: str = "bootstrap"  # 'bootstrap' | 'wilson' | 'bayes' (label only)
    ci_level: float = 0.95

    def __post_init__(self) -> None:
        # Sanity checks
        if self.confidence_interval is not None:
            lo, hi = float(self.confidence_interval[0]), float(self.confidence_interval[1])
            if lo > hi:
                raise ValueError("confidence_interval must be (lo <= hi)")
        if self.bootstrap_samples is not None and not isinstance(self.bootstrap_samples, np.ndarray):
            self.bootstrap_samples = np.asarray(self.bootstrap_samples, dtype=float)
        if self.ci_level is not None:
            if not (0.5 < float(self.ci_level) < 1.0):
                raise ValueError("ci_level must be in (0.5, 1.0)")

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "j_empirical": float(self.j_empirical),
            "cc_max": float(self.cc_max),
            "delta_add": float(self.delta_add),
            "cc_multiplicative": (None if self.cc_multiplicative is None else float(self.cc_multiplicative)),
            "confidence_interval": (None if self.confidence_interval is None else [float(self.confidence_interval[0]), float(self.confidence_interval[1])]),
            "bootstrap_samples": (None if self.bootstrap_samples is None else self.bootstrap_samples.tolist()),
            "n_sessions": int(self.n_sessions),
            "ci_method": str(self.ci_method),
            "ci_level": float(self.ci_level),
        }
        return data


@dataclass(slots=True)
class AttackStrategy(_ModelIO):
    """Configuration for an attack strategy (declarative)."""
    name: str
    params: Dict[str, Any]
    vocabulary: List[str] = field(default_factory=list)
    success_threshold: float = 0.5
    description: str = ""

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}
        if not isinstance(self.params, dict):
            self.params = dict(self.params)  # type: ignore[arg-type]
        if self.vocabulary is None:
            self.vocabulary = []
        _validate_prob(max(0.0, min(1.0, float(self.success_threshold))), "success_threshold")


# =============================================================================
# Backwards-compat re-exports (module originally named 'data models' inline)
# =============================================================================
# The original file name was 'src/cc/core/<something>.py' with the same classes.
# If your code imports from 'cc.core.models' you are already set. If it imports
# from the old module path, add a thin shim that re-exports these classes.

__all__ = [
    "AttackResult",
    "GuardrailSpec",
    "WorldConfig",
    "ExperimentConfig",
    "CCResult",
    "AttackStrategy",
]