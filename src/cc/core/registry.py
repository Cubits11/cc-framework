# src/cc/core/registry.py
"""
Registry & builders for attackers and guardrails.

This module upgrades the simple factory helpers into a robust, typed,
and extensible registry with:

- Friendly aliases (e.g., "regex", "regex_filter", "keyword")
- Backward-compat with older names (RegexFilters → RegexFilter)
- Validation of required guardrail interface (score / blocks OR evaluate)
- Support for dict configs *or* GuardrailSpec objects
- Helpful error messages with close-name suggestions
- Optional plugin handoff via a late-bound hook

Author: Pranav Bhave
Updated: 2025-09-28
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union

# Core types
from cc.core.attackers import AttackStrategy, RandomInjectionAttacker
from cc.core.models import GuardrailSpec
from cc.guardrails.base import Guardrail

# Built-in guardrails
try:
    from cc.guardrails.keyword_blocker import KeywordBlocker
except Exception:  # pragma: no cover
    KeywordBlocker = None  # type: ignore[assignment]

try:
    # Canonical class name in repo: RegexFilter
    from cc.guardrails.regex_filters import RegexFilter  # modern name
except Exception:  # pragma: no cover
    RegexFilter = None  # type: ignore[assignment]

try:
    from cc.guardrails.semantic_filter import SemanticFilter
except Exception:  # pragma: no cover
    SemanticFilter = None  # type: ignore[assignment]

try:
    from cc.guardrails.toy_threshold import ToyThresholdGuardrail
except Exception:  # pragma: no cover
    ToyThresholdGuardrail = None  # type: ignore[assignment]


# ------------------------------------------------------------------------------
# Attacker registry
# ------------------------------------------------------------------------------

_Attackers: Dict[str, Type[AttackStrategy]] = {
    # canonical
    "random_injection": RandomInjectionAttacker,
    # aliases
    "random": RandomInjectionAttacker,
    "rand_inject": RandomInjectionAttacker,
}

# ------------------------------------------------------------------------------
# Guardrail registry (class references) + aliases
# ------------------------------------------------------------------------------

_Guardrails: Dict[str, Optional[Type[Guardrail]]] = {
    # canonical names
    "keyword_blocker": KeywordBlocker,
    "regex_filter": RegexFilter,   # preferred modern name
    "semantic_filter": SemanticFilter,
    "toy_threshold": ToyThresholdGuardrail,

    # friendly aliases
    "keyword": KeywordBlocker,
    "regex": RegexFilter,
    "regex_filters": RegexFilter,  # backward compat (plural)
    "semantic": SemanticFilter,
}

_KNOWN_GUARDRAIL_NAMES = tuple(_Guardrails.keys())


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _suggest_name(bad: str, known: Iterable[str]) -> str:
    """Return a short suggestion string for unknown names."""
    cand = difflib.get_close_matches(str(bad), list(known), n=3, cutoff=0.6)
    return f" Did you mean: {', '.join(cand)}?" if cand else ""


def _ensure_guardrail_interface(obj: Any) -> None:
    """
    Best-effort interface check. We accept:
      - evaluate(text) -> (blocked: bool, score: float), OR
      - score(text) -> float  AND  blocks(text) -> bool
    """
    has_eval = hasattr(obj, "evaluate")
    has_score = hasattr(obj, "score")
    has_blocks = hasattr(obj, "blocks")
    if not (has_eval or (has_score and has_blocks)):
        raise TypeError(
            "Guardrail instance lacks required interface. "
            "Expected either `.evaluate(text) -> (bool, float)` or both "
            "`.score(text) -> float` and `.blocks(text) -> bool`."
        )


def _instantiate_guardrail(
    cls: Type[Guardrail],
    params: Optional[Mapping[str, Any]],
) -> Guardrail:
    """Create and validate a guardrail instance."""
    instance = cls(**(params or {}))  # type: ignore[call-arg]
    _ensure_guardrail_interface(instance)
    return instance


def _resolve_guardrail_class(name: str) -> Type[Guardrail]:
    """
    Resolve a guardrail class by (case-insensitive) name with aliasing,
    raising a clear error if missing or not importable.
    """
    key = str(name).lower().strip()
    cls = _Guardrails.get(key)
    if cls is None:
        # Either not registered or import failed
        suffix = _suggest_name(key, _KNOWN_GUARDRAIL_NAMES)
        raise ValueError(f"Unknown or unavailable guardrail: '{name}'.{suffix}")
    return cls


# ------------------------------------------------------------------------------
# Public builders
# ------------------------------------------------------------------------------

def build_attacker(cfg: Optional[Dict[str, Any]]) -> AttackStrategy:
    """
    Build an attacker from a dict config:
      cfg = {"type": <name>, "params": {...}}

    Defaults to {"type": "random_injection"}.

    Raises:
        ValueError if type unknown.
    """
    cfg = cfg or {}
    t = str(cfg.get("type", "random_injection")).lower().strip()
    cls = _Attackers.get(t)
    if cls is None:
        raise ValueError(f"Unknown attacker type: '{t}'.{_suggest_name(t, _Attackers.keys())}")
    params = cfg.get("params") or {}
    return cls(**params)  # type: ignore[call-arg]


def build_guardrails(
    cfg_list: Optional[List[Dict[str, Any]]]
) -> List[Guardrail]:
    """
    Build a list of guardrails from a list of dict configs:

      cfg_list = [
        {"name": "regex_filter", "params": {...}},
        {"name": "keyword_blocker", "params": {...}},
      ]

    You may also pass legacy alias names (e.g., "regex", "regex_filters").

    Raises:
        ValueError if any name is unknown or unavailable.
        TypeError if constructed instance fails interface validation.
    """
    out: List[Guardrail] = []
    for cfg in (cfg_list or []):
        name = cfg.get("name")
        if not name:
            raise ValueError("Guardrail config missing 'name' field.")
        cls = _resolve_guardrail_class(name)
        params = cfg.get("params") or {}
        out.append(_instantiate_guardrail(cls, params))
    return out


def build_guardrail_stack_from_specs(specs: Optional[List[GuardrailSpec]]) -> List[Guardrail]:
    """
    Build a guardrail stack from typed GuardrailSpec objects.

    Each GuardrailSpec carries `name`, `params`, and optional calibration hints.
    """
    out: List[Guardrail] = []
    for spec in (specs or []):
        cls = _resolve_guardrail_class(spec.name)
        out.append(_instantiate_guardrail(cls, spec.params))
    return out


# ------------------------------------------------------------------------------
# Introspection & registration (optional extension points)
# ------------------------------------------------------------------------------

def list_attackers() -> List[str]:
    """Return canonical attacker keys available in this registry."""
    return sorted(_Attackers.keys())


def list_guardrails() -> List[str]:
    """Return canonical guardrail keys and known aliases."""
    return sorted(set(_Guardrails.keys()))


def register_attacker(name: str, cls: Type[AttackStrategy]) -> None:
    """
    Register a custom attacker at runtime.

    Notes:
        This is a simple registry; name collisions overwrite previous entries.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Attacker name must be a non-empty string.")
    _Attackers[name.lower().strip()] = cls


def register_guardrail(name: str, cls: Type[Guardrail]) -> None:
    """
    Register a custom guardrail at runtime.

    Notes:
        - Performs an interface sanity-check by instantiating with no args.
          If your guardrail requires parameters, it will be re-validated
          at build time with actual params.
        - Name collisions overwrite previous entries.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Guardrail name must be a non-empty string.")
    key = name.lower().strip()
    _Guardrails[key] = cls
    # lightweight interface check (best-effort)
    try:
        _ensure_guardrail_interface(cls(**{}))  # type: ignore[misc]
    except Exception:
        # Defer strict checks until actual instantiation with params.
        pass


__all__ = [
    "build_attacker",
    "build_guardrails",
    "build_guardrail_stack_from_specs",
    "list_attackers",
    "list_guardrails",
    "register_attacker",
    "register_guardrail",
]