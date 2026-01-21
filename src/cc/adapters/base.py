# src/cc/adapters/base.py
"""Base adapter interfaces for third-party guardrail systems.

Enterprise goals:
- Deterministic, tamper-evident audit payloads (stable canonicalization + hash)
- Secret-safe logging (deep redaction of sensitive keys)
- Normalized adapter contract across vendors (Decision + error taxonomy)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
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

_PROMPT_KEY_FRAGMENTS: Sequence[str] = (
    "prompt",
    "content",
    "input",
    "output",
    "response",
    "instruction",
    "assistant",
    "user",
)

_SAFE_PREVIEW_RE = re.compile(r"^[a-zA-Z0-9 _.,:/-]{0,32}$")
_PII_PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # credit card-ish
    re.compile(r"\b\d{10,}\b"),  # long digit runs
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
    metadata_summary: Dict[str, Any]
    parameters_fingerprint: str
    metadata_fingerprint: str
    error_summary: Dict[str, Any]

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


def _looks_prompt_like_key(key: str) -> bool:
    k = key.lower()
    return any(fragment in k for fragment in _PROMPT_KEY_FRAGMENTS)


def _looks_like_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PII_PATTERNS)


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


@dataclass(frozen=True)
class SanitizationPolicy:
    """Policy for deterministic, leak-safe sanitization."""

    max_depth: int = 6
    max_list_items: int = 50
    max_string_length: int = 128
    allow_preview: bool = False


def _safe_preview(text: str, policy: SanitizationPolicy) -> Optional[str]:
    if not policy.allow_preview:
        return None
    if len(text) > 32:
        return None
    if _looks_like_pii(text):
        return None
    if not _SAFE_PREVIEW_RE.match(text):
        return None
    return text


def _hash_summary(text: str, policy: SanitizationPolicy) -> Dict[str, Any]:
    return {
        "sha256": hash_text(text) or "",
        "len": len(text),
        "type": "str",
        "preview": _safe_preview(text, policy),
    }


def sanitize_value(
    value: Any,
    policy: Optional[SanitizationPolicy] = None,
    *,
    key: Optional[str] = None,
    depth: int = 0,
) -> JsonValue:
    """Sanitize a value deterministically to prevent leaks.

    Strings are hashed when they are long, match PII, or are under prompt-like keys.
    Nested containers are traversed with a depth limit.
    """
    policy = policy or SanitizationPolicy()
    if depth > policy.max_depth:
        return {"__truncated__": True, "type": type(value).__name__}

    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        if _looks_like_pii(value) or len(value) > policy.max_string_length:
            return _hash_summary(value, policy)
        if key and (_looks_sensitive_key(key) or _looks_prompt_like_key(key)):
            return _hash_summary(value, policy)
        if _looks_prompt_like_key(value):
            return _hash_summary(value, policy)
        return value

    if isinstance(value, (bytes, bytearray)):
        return {"__bytes_hex__": bytes(value).hex()}

    if is_dataclass(value):
        return sanitize_value(asdict(value), policy, key=key, depth=depth + 1)

    if isinstance(value, Mapping):
        out: Dict[str, JsonValue] = {}
        for k in sorted(value.keys(), key=lambda x: str(x)):
            ks = str(k)
            out[ks] = sanitize_value(value[k], policy, key=ks, depth=depth + 1)
        return out

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if isinstance(value, set):
            items = sorted(items, key=lambda x: str(x))
        sanitized = [
            sanitize_value(item, policy, key=key, depth=depth + 1)
            for item in items[: policy.max_list_items]
        ]
        if len(items) > policy.max_list_items:
            sanitized.append({"__truncated__": len(items) - policy.max_list_items})
        return sanitized

    return {"__nonserializable__": type(value).__name__}


def summarize_value(value: Any, policy: Optional[SanitizationPolicy] = None) -> Dict[str, Any]:
    """Summarize a value with hash/length/type only (no raw content)."""
    policy = policy or SanitizationPolicy()
    if isinstance(value, str):
        return _hash_summary(value, policy)
    if value is None:
        return {"sha256": hash_text("null") or "", "len": 0, "type": "null", "preview": None}
    sanitized = sanitize_value(value, policy)
    canonical = canonical_json(sanitized, strict=False)
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8", errors="surrogatepass")).hexdigest(),
        "len": len(canonical),
        "type": type(value).__name__,
        "preview": None,
    }


def sanitize_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize metadata with hash/length/type only."""
    if not metadata:
        return {}
    policy = SanitizationPolicy()
    return {str(k): summarize_value(v, policy) for k, v in sorted(metadata.items())}


def sanitize_vendor_payload(payload: Any) -> JsonValue:
    """Sanitize vendor payloads defensively, including arbitrary objects."""
    policy = SanitizationPolicy(max_string_length=0)
    if payload is None:
        return {}
    if is_dataclass(payload):
        return sanitize_value(asdict(payload), policy)
    if isinstance(payload, Mapping):
        return sanitize_value(payload, policy)
    if hasattr(payload, "__dict__"):
        return sanitize_value(vars(payload), policy)
    return {"__type__": type(payload).__name__, "sha256": hash_text(repr(payload)) or ""}


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
    error_summary: Optional[Dict[str, Any]] = None,
    event_id: Optional[str] = None,
    created_at: Optional[float] = None,
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
    safe_parameters = sanitize_value(_redact(parameters))
    metadata_summary = sanitize_metadata(meta)

    # Deterministic fingerprints (strict=False so logging cannot crash a run).
    parameters_fp = fingerprint_payload({"parameters": safe_parameters}, strict=False)
    metadata_fp = fingerprint_payload({"metadata_summary": metadata_summary}, strict=False)

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
    created_at_value = float(created_at) if created_at is not None else started
    event_id_value = event_id or fingerprint_payload(
        {
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "prompt_hash": hash_text(prompt) or "",
            "response_hash": hash_text(response),
            "started_at": started,
        },
        strict=False,
    )

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
        "event_id": event_id_value,
        "created_at": created_at_value,
        "prompt_chars": len(prompt),
        "response_chars": (len(response) if response is not None else 0),
        "metadata_summary": metadata_summary,
        "parameters_fingerprint": parameters_fp,
        "metadata_fingerprint": metadata_fp,
        "chain_prev_hash": chain_prev_hash,
        "chain_seq": chain_seq,
        "error_summary": error_summary,
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


def error_summary_from_exception(exc: Exception, *, where: str) -> Dict[str, Any]:
    """Build a safe error summary for audit payloads."""
    retryable = isinstance(exc, (AdapterTransientError, TimeoutError))
    message_hash = hash_text(str(exc)) or ""
    return {
        "type": type(exc).__name__,
        "message_hash": message_hash,
        "where": where,
        "retryable": retryable,
    }


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
                "config": sanitize_value(_redact(dict(self.get_config()))),
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
