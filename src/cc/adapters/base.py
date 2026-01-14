# src/cc/adapters/base.py
"""Base adapter interfaces for third-party guardrail systems.

Enterprise goals:
- Deterministic, tamper-evident audit payloads (stable canonicalization + hash)
- Secret-safe logging (deep redaction of sensitive keys)
- Normalized adapter contract across vendors (Decision + error taxonomy)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass, asdict
import hashlib
import json
import time
import uuid
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, TypedDict, Union


# ----------------------------
# Public contract
# ----------------------------

Verdict = Literal["allow", "block", "review"]

# v2: adds event_id/event_hash + canonicalization + redaction + config fingerprint behavior.
AUDIT_SCHEMA_VERSION = "cc.guardrail_adapter.audit.v2"

# Common keys that should NEVER be written verbatim into audit logs.
# We match by substring (case-insensitive) on key names, including nested structures.
_SENSITIVE_KEY_FRAGMENTS: Sequence[str] = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "token",
    "secret",
    "password",
    "passwd",
    "private_key",
    "client_secret",
    "access_key",
    "session",
)


JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], list["JsonValue"]]


class AuditPayloadV1(TypedDict):
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


class AuditPayload(AuditPayloadV1, total=False):
    """Audit payload v2 extends v1 with optional fields (safe for older callers)."""

    event_id: str
    event_hash: str
    created_at: float

    # Optional diagnostics (no raw prompt/response, only sizes)
    prompt_chars: int
    response_chars: int

    # Sanitized view for reproducibility without secrets
    metadata: Dict[str, Any]
    parameters_fingerprint: str
    metadata_fingerprint: str

    # Optional link hooks for external ledger / chain layers
    chain_prev_hash: Optional[str]
    chain_seq: Optional[int]


# ----------------------------
# Canonicalization + hashing
# ----------------------------

def hash_text(text: Optional[str]) -> Optional[str]:
    """SHA256 over UTF-8 bytes. Never returns empty string unless text is empty."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _looks_sensitive_key(key: str) -> bool:
    k = key.lower()
    return any(fragment in k for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _redact(obj: Any) -> Any:
    """Deep-redact sensitive keys in mappings, recursively."""
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if _looks_sensitive_key(ks):
                out[ks] = "<redacted>"
            else:
                out[ks] = _redact(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_redact(x) for x in obj]
    return obj


def _to_jsonable(obj: Any, *, strict: bool) -> JsonValue:
    """Convert python objects to a deterministic JSON-serializable structure.

    strict=True  -> raises on unknown types
    strict=False -> degrades unknown types to stable placeholders
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        # Note: floats with NaN/Inf should be avoided; json.dumps(allow_nan=False) will enforce.
        return obj

    # bytes -> hex for determinism
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_hex__": bytes(obj).hex()}

    # dataclasses -> asdict recursively
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj), strict=strict)

    # mappings -> dict with string keys
    if isinstance(obj, Mapping):
        out: Dict[str, JsonValue] = {}
        for k, v in obj.items():
            out[str(k)] = _to_jsonable(v, strict=strict)
        return out

    # sequences -> list
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x, strict=strict) for x in obj]

    # fallback
    if strict:
        raise TypeError(f"Object not JSON-serializable: {type(obj).__name__}")
    return {"__nonserializable__": type(obj).__name__}


def canonical_json(obj: Any, *, strict: bool = False) -> str:
    """Deterministic JSON encoding for hashing/fingerprinting."""
    jsonable = _to_jsonable(obj, strict=strict)
    return json.dumps(
        jsonable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,  # prevents NaN/Inf nondeterminism
    )


def fingerprint_payload(payload: Dict[str, Any], *, strict: bool = False) -> str:
    """Stable fingerprint of a payload for config/audit integrity."""
    canonical = canonical_json(payload, strict=strict)
    return hashlib.sha256(canonical.encode("utf-8", errors="surrogatepass")).hexdigest()


# ----------------------------
# Audit payload builder
# ----------------------------

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
    metadata: Optional[Dict[str, Any]] = None,
    vendor_request_id: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
    # Ledger hooks (optional)
    chain_prev_hash: Optional[str] = None,
    chain_seq: Optional[int] = None,
) -> AuditPayload:
    """Build a deterministic, secret-safe audit payload.

    Never includes raw prompt/response text. Only hashes + sizes.
    Redacts sensitive keys inside parameters/metadata.
    Adds event_id + event_hash for tamper-evident logging.
    """
    meta = metadata or {}

    # Clamp/validate time fields defensively.
    try:
        started = float(started_at)
        completed = float(completed_at)
    except Exception:
        started = float(time.time())
        completed = started

    if completed < started:
        # Keep invariants for downstream consumers.
        completed = started

    duration_ms = max(0.0, (completed - started) * 1000.0)

    # Redact secrets *before* fingerprinting.
    safe_parameters = _redact(parameters)
    safe_metadata = _redact(meta)

    # Deterministic fingerprints (strict=False so logging cannot crash a run).
    parameters_fp = fingerprint_payload({"parameters": safe_parameters}, strict=False)
    metadata_fp = fingerprint_payload({"metadata": safe_metadata}, strict=False)

    # Derive a config fingerprint if caller didn't provide one:
    # binds adapter identity + version + parameters fingerprint (and optional metadata fingerprint).
    derived_cfg_fp = fingerprint_payload(
        {
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "parameters_fingerprint": parameters_fp,
            "metadata_fingerprint": metadata_fp,
        },
        strict=False,
    )
    cfg_fp = config_fingerprint or derived_cfg_fp

    # Stable event identity + tamper-evident hash for ledger chaining.
    event_id = uuid.uuid4().hex
    created_at = started

    payload: AuditPayload = {
        "schema": AUDIT_SCHEMA_VERSION,
        "prompt_hash": hash_text(prompt) or "",
        "response_hash": hash_text(response),
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "parameters": safe_parameters,  # safe view only
        "decision": decision,
        "category": category,
        "rationale": (rationale[:2000] if isinstance(rationale, str) else rationale),
        "started_at": started,
        "completed_at": completed,
        "duration_ms": float(duration_ms),
        "vendor_request_id": vendor_request_id,
        "config_fingerprint": cfg_fp,
        # v2 extras
        "event_id": event_id,
        "created_at": float(created_at),
        "prompt_chars": len(prompt),
        "response_chars": (len(response) if response is not None else 0),
        "metadata": safe_metadata,
        "parameters_fingerprint": parameters_fp,
        "metadata_fingerprint": metadata_fp,
        "chain_prev_hash": chain_prev_hash,
        "chain_seq": chain_seq,
    }

    # event_hash excludes itself; binds the entire event deterministically.
    event_hash = fingerprint_payload(
        {k: v for k, v in payload.items() if k != "event_hash"},
        strict=False,
    )
    payload["event_hash"] = event_hash
    return payload


# ----------------------------
# Decision + errors
# ----------------------------

@dataclass(frozen=True)
class Decision:
    """Normalized guardrail decision returned by adapter checks."""

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

    def with_audit(self, audit_payload: Dict[str, Any]) -> "Decision":
        """Return a copy of this Decision with audit attached."""
        return Decision(
            verdict=self.verdict,
            category=self.category,
            score=self.score,
            rationale=self.rationale,
            raw=self.raw,
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            audit=audit_payload,
        )


class AdapterError(RuntimeError):
    """Base adapter error."""


class AdapterMisconfigured(AdapterError):
    """Raised when adapter is misconfigured (missing key, invalid endpoint, etc.)."""


class AdapterRateLimited(AdapterError):
    """Raised when vendor rate limiting is encountered."""


class AdapterTransientError(AdapterError):
    """Raised for retryable failures (timeouts, temporary network errors)."""


class AdapterPermanentError(AdapterError):
    """Raised for non-retryable vendor failures (bad request, policy violation, etc.)."""


# ----------------------------
# Adapter base class
# ----------------------------

class GuardrailAdapter(ABC):
    """Abstract base class for guardrail adapters.

    Enterprise contract:
    - deterministic adapter name/version
    - stable capability flags
    - check() returns a normalized Decision
    """

    name: str
    version: str
    supports_input_check: bool = True
    supports_output_check: bool = True

    def get_config(self) -> Mapping[str, Any]:
        """Return adapter configuration (safe, non-secret) for fingerprinting."""
        return {}

    def fingerprint_config(self) -> str:
        """Stable fingerprint for adapter identity + config."""
        return fingerprint_payload(
            {
                "adapter_name": self.name,
                "adapter_version": self.version,
                "config": _redact(dict(self.get_config())),
            },
            strict=False,
        )

    def check_input(self, prompt: str, metadata: Dict[str, Any]) -> Decision:
        """Default input-only check via check(prompt, None, ...)."""
        if not self.supports_input_check:
            raise AdapterMisconfigured(f"{self.name} does not support input checks")
        return self.check(prompt=prompt, response=None, metadata=metadata)

    def check_output(self, prompt: str, response: str, metadata: Dict[str, Any]) -> Decision:
        """Default output check via check(prompt, response, ...)."""
        if not self.supports_output_check:
            raise AdapterMisconfigured(f"{self.name} does not support output checks")
        return self.check(prompt=prompt, response=response, metadata=metadata)

    @abstractmethod
    def check(self, prompt: str, response: Optional[str], metadata: Dict[str, Any]) -> Decision:
        """Check a prompt/response pair and return a Decision."""
        raise NotImplementedError
