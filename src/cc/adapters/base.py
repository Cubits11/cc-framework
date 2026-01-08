# src/cc/adapters/base.py
"""Base adapter interfaces for third-party guardrail systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TypedDict

Verdict = Literal["allow", "block", "review"]
AUDIT_SCHEMA_VERSION = "cc.guardrail_adapter.audit.v1"


class AuditPayload(TypedDict):
    schema: str
    prompt_hash: str
    response_hash: Optional[str]
    adapter_name: str
    adapter_version: str
    parameters: Dict[str, Any]
    decision: Verdict
    category: Optional[str]
    rationale: Optional[str]
    started_at: float
    completed_at: float
    duration_ms: float
    vendor_request_id: Optional[str]
    config_fingerprint: Optional[str]


def hash_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_payload(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_audit_payload(
    *,
    prompt: str,
    response: Optional[str],
    adapter_name: str,
    adapter_version: str,
    parameters: Dict[str, Any],
    decision: Verdict,
    category: Optional[str],
    rationale: Optional[str],
    started_at: float,
    completed_at: float,
    vendor_request_id: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
) -> AuditPayload:
    duration_ms = max(0.0, (completed_at - started_at) * 1000.0)
    return {
        "schema": AUDIT_SCHEMA_VERSION,
        "prompt_hash": hash_text(prompt) or "",
        "response_hash": hash_text(response),
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "parameters": parameters,
        "decision": decision,
        "category": category,
        "rationale": rationale,
        "started_at": float(started_at),
        "completed_at": float(completed_at),
        "duration_ms": float(duration_ms),
        "vendor_request_id": vendor_request_id,
        "config_fingerprint": config_fingerprint,
    }


@dataclass(frozen=True)
class Decision:
    """Normalized guardrail decision returned by adapter checks.

    Attributes
    ----------
    verdict:
        One of {"allow", "block", "review"}.
    category:
        Optional policy category or violation label when available.
    score:
        Optional confidence or risk score (adapter-specific scale).
    rationale:
        Optional short explanation for the decision.
    raw:
        Raw adapter payload (dict or string) for reproducibility/debugging.
    adapter_name:
        Stable adapter name (e.g., "llama_guard").
    adapter_version:
        Adapter or model version (e.g., model id or library version).
    audit:
        Optional audit payload for tamper-evident logging.
    """

    verdict: Verdict
    category: Optional[str]
    score: Optional[float]
    rationale: Optional[str]
    raw: Dict[str, Any] | str
    adapter_name: str
    adapter_version: str
    audit: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.verdict not in {"allow", "block", "review"}:
            raise ValueError(f"Invalid verdict: {self.verdict}")
        if not isinstance(self.raw, (dict, str)):
            raise TypeError("Decision.raw must be a dict or string")
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise TypeError("Decision.score must be float-like when provided")
        if self.audit is not None and not isinstance(self.audit, dict):
            raise TypeError("Decision.audit must be a dict when provided")


class GuardrailAdapter(ABC):
    """Abstract base class for guardrail adapters."""

    name: str
    version: str
    supports_input_check: bool
    supports_output_check: bool

    @abstractmethod
    def check(self, prompt: str, response: Optional[str], metadata: Dict[str, Any]) -> Decision:
        """Check a prompt/response pair and return a Decision."""
        raise NotImplementedError
