from __future__ import annotations
from typing import Any, Dict, List
from cc.core.attackers import RandomInjectionAttacker
from cc.guardrails.keyword_blocker import KeywordBlocker
from cc.guardrails.regex_filters import RegexFilters
from cc.guardrails.semantic_filter import SemanticFilter
from cc.guardrails.base import BaseGuardrail

_ATTACKERS = {"random_injection": RandomInjectionAttacker}
_GUARDRAILS = {
    "keyword_blocker": KeywordBlocker,
    "regex_filters": RegexFilters,
    "semantic_filter": SemanticFilter,
}

def build_attacker(cfg: Dict[str, Any]):
    t = (cfg or {}).get("type", "random_injection")
    cls = _ATTACKERS.get(t)
    if cls is None:
        raise ValueError(f"Unknown attacker type: {t}")
    return cls(**(cfg.get("params", {}) or {}))

def build_guardrails(cfg_list: List[Dict[str, Any]]) -> List[BaseGuardrail]:
    out: List[BaseGuardrail] = []
    for c in cfg_list or []:
        t = c.get("name")
        cls = _GUARDRAILS.get(t)
        if cls is None:
            raise ValueError(f"Unknown guardrail: {t}")
        out.append(cls(**(c.get("params", {}) or {})))
    return out
