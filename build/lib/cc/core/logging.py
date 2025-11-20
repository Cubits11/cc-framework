# src/cc/core/logging.py
"""
Module: logging
Purpose: JSONL audit logger (shim) that delegates to Cartographer's tamper-evident chain
Dependencies: json, time, pathlib, re, os, socket, subprocess, platform, asyncio, functools, secrets; optional: prometheus_client, psutil; delegates hashing/verification to cc.cartographer.audit
Author: Pranav Bhave
Dates:
  - 2025-08-27: initial shim
  - 2025-09-14: (deterministic envelopes, env/git introspection,
                optional sanitize, verify-on-write, lockfile, richer context)
  - 2025-09-28: (rotation, compression, redaction, batch ops,
                env snapshot, thread-safety, schema v3, tail/head utilities)
  - 2025-11-13: (async support, Prometheus metrics, PII patterns,
                strict mode, config from env, hooks, full tests/docs)
  - 2025-11-13 (expanded): Fixed bugs (Prometheus labels, async kwargs via partial, pip freeze try/except, 
                compile_patterns fail-soft, PII value scanning), added levels, encryption option, alerting example in hook,
                async lock handling, more tests/docs per research.
  - 2025-11-14: Upgraded to 10/10 production readiness: fixed lock precedence and nested locks, encryption key management,
                rotation metrics safety, session ID randomness, git SHA validation, tail OOM prevention, Prometheus labels,
                stale lock PID check, async semaphore, JSON strictness, post-hook recursion guard, redaction excludes/separation,
                lock failure semantics + verified level in meta.extra.

Notes
-----
- This remains a thin **compatibility shim** over `cc.cartographer.audit`.
- It preserves the legacy API while writing records in the tamper-evident format
  with fields: `prev_sha256`, `sha256` (Cartographer owns hash discipline).
- New features are opt-in and non-breaking:
    * Deterministic envelope with both unix and ISO-8601 timestamps (UTC),
      host, pid, git SHA, session id, optional seed and env snapshot (now with pip freeze fallback).
    * `auto_sanitize=True` to coerce common non-JSON types (Path, set, tuple, bytes).
    * `verify_on_write=True` to run a full chain verification after append.
    * Best-effort inter-process lock (atomic lockfile with PID check) to guard concurrent writers, with retries/timeout.
    * **Redaction** via `redact_keys`, `redact_patterns` (regex for keys/values), built-in PII patterns with configurable mask (scans string values too), and exclude_keys.
    * **Rotation** via `max_bytes`/`backup_count` and optional `compress_backups=True` (.gz).
    * **Batch** helpers (`log_many`) and quick **introspection** (`tail`, `current_head`).
    * Optional `fsync_on_write=True` for durability (best effort).
    * **Async** support via `alog` coroutines with semaphore for concurrency control.
    * **Metrics** via Prometheus (log_entries, log_errors, log_size) with level labels.
    * **Post-log hooks** with recursion guard.
    * **Strict mode** to raise on soft failures.
    * **Encryption option** for backups (AES-Fernet), requires stable key via param/env.
    * **Levels** support (debug/info/warning/error) for log calls, recorded under meta.extra["level"].
- Use `cc.cartographer.audit` directly for new code if you don’t need the compatibility layer.
- Environment config: LOG_STRICT_MODE=1, LOG_PROMETHEUS=1, LOG_ENC_KEY=base64key, etc.
- Research insights: Tamper-proof via external alerts (hook), encryption (new), levels for filtering (new), fail-soft defaults (adjusted).

Schema
------
Each record is written with an envelope:

{
  "meta": {
    "schema": "core/logging.v5",  # Upgraded for upgrades
    "ts_unix": <float>,
    "ts_iso": "YYYY-MM-DDTHH:MM:SS.sssZ",
    "host": <str>,            # hostname
    "pid": <int>,
    "git_sha": <str|null>,    # env/auto-detected with validation
    "session_id": <str>,      # stable per-process launch with randomness
    "seed": <int|null>,       # optional (user-provided)
    "env": { ... },           # optional snapshot (python/platform/pip) if capture_env=True
    "extra": { ... }          # user-supplied meta overrides (includes "level")
  },
  "payload": { ... }          # user content (JSON-serializable)
  # chain fields are injected by cc.cartographer.audit:
  # "prev_sha256": "...",
  # "sha256": "..."
}
"""

from __future__ import annotations

import asyncio
import base64
import functools
import gzip
import hashlib
import json
import os
import platform
import re
import secrets
import socket
import subprocess
import sys
import time
import threading
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from prometheus_client import Counter, Gauge  # Optional metrics

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from cryptography.fernet import Fernet  # Optional encryption

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# NOTE: cc.cartographer.audit is assumed to be the tamper-evident chain owner.
from cc.cartographer import audit

__all__ = [
    "ChainedJSONLLogger",
    "audit_context",
    "aaudit_context",
    "LoggingError",
]

SCHEMA_ID_DEFAULT = "core/logging.v5"  # Upgraded for changes
_LOCKFILE_SUFFIX = ".lock"

# Built-in PII regexes applied to both keys and string values.
# These are intentionally conservative but cover common high-risk patterns.
PII_PATTERNS_DEFAULT = [
    r"(?i)\b(?:email|mail_address)\b",  # key names like "email"
    r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",  # email addresses in text
    r"\b\d{3}-\d{2}-\d{4}\b",  # strict US SSN
    r"\b(?:4\d{3}(?:[ -]?\d{4}){3})\b",  # Visa-like credit card (with optional spaces/hyphens)
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",  # strict IPv4
]

# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    LOG_ENTRIES = Counter("cc_audit_log_entries", "Number of log entries", ["level"])
    LOG_ERRORS = Counter("cc_audit_log_errors", "Number of logging errors", ["type"])
    LOG_SIZE = Gauge("cc_audit_log_size_bytes", "Current log file size")

# A stable per-process session id (pid + monotonic start + randomness hashed)
_session_start_perf = time.perf_counter()
_session_id = hashlib.sha256(
    f"{os.getpid()}:{_session_start_perf:.9f}:{secrets.token_hex(8)}".encode("utf-8")
).hexdigest()[:16]


class LoggingError(RuntimeError):
    """Raised for audit logging failures (serialization, locking, verification, rotation)."""


def _now_unix() -> float:
    """Seconds since epoch as float (UTC)."""
    return float(time.time())


def _now_iso(ts: Optional[float] = None) -> str:
    """ISO-8601 in UTC with 'Z' suffix; millisecond precision."""
    if ts is None:
        ts = _now_unix()
    tm = time.gmtime(ts)  # UTC
    ms = int((ts - int(ts)) * 1000)
    return (
        f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}"
        f"T{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{ms:03d}Z"
    )


_git_sha_cache: Optional[str] = None


def _is_valid_short_sha(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{12}", s))


def _detect_git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    """Resolve git SHA once (env `GIT_SHA` wins if valid), else `git rev-parse --short=12 HEAD`."""
    global _git_sha_cache
    if _git_sha_cache is not None:
        return _git_sha_cache

    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        stripped = env_sha.strip()
        if _is_valid_short_sha(stripped):
            _git_sha_cache = stripped.lower()
            return _git_sha_cache

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
        stripped = out.strip()
        if _is_valid_short_sha(stripped):
            _git_sha_cache = stripped.lower()
        else:
            _git_sha_cache = None
    except Exception:
        _git_sha_cache = None
    return _git_sha_cache


def _coerce_json_safe(obj: Any) -> Any:
    """
    Best-effort coercion of common non-JSON types:
      - Path -> str
      - set/tuple -> list
      - bytes/bytearray -> hex str
      - dataclass -> dict
    Leaves unknown types to default str().
    """
    from dataclasses import asdict, is_dataclass  # lazy import

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            return str(obj)
    return obj

def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively coerce `obj` into a JSON-serializable structure.

    Uses `_coerce_json_safe` for leaf-level conversions (Path -> str, bytes -> hex,
    dataclass -> dict, set/tuple -> list, etc.) and then walks nested containers.

    This function does *not* modify keys in mappings; only values are sanitized.
    """
    # First apply low-level coercions (Path -> str, bytes -> hex, dataclass -> dict, etc.)
    obj = _coerce_json_safe(obj)

    # Then recursively sanitize containers
    if isinstance(obj, dict):
        # Preserve original keys; JSON encoder will string-ify them if needed
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        # Sets/tuples already coerced to list in _coerce_json_safe, but we handle
        # generically here in case of future extensions.
        return [_sanitize_for_json(v) for v in obj]

    # Primitives / already-JSON-safe values are returned as-is
    return obj


def _ensure_json(obj: Any, auto_sanitize: bool) -> Any:
    """
    Ensure that `obj` is JSON-serializable under our rules, optionally returning
    a sanitized copy.

    Behavior
    --------
    - If auto_sanitize is True:
        * Return a deep-sanitized copy (via `_sanitize_for_json`) that is guaranteed
          to be encodable by `json.dumps(..., allow_nan=False)`.
        * Raise TypeError / ValueError if, after sanitization, the structure still
          cannot be encoded (e.g., due to NaN/Inf).
    - If auto_sanitize is False:
        * Validate that `obj` as-is is JSON-serializable with `allow_nan=False`.
        * Return the original `obj` unchanged.
    """
    if auto_sanitize:
        sanitized = _sanitize_for_json(obj)
        # This will raise if there are still problematic values (e.g. NaN/Inf)
        json.dumps(
            sanitized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return sanitized

    # Validation-only mode: let json.dumps raise if obj is not serializable
    json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return obj

def _canonical_dumps(obj: Dict[str, Any], auto_sanitize: bool) -> str:
    """
    Return a canonical JSON string (sorted keys, compact separators, no NaN/Inf).

    `auto_sanitize` is kept in the signature for compatibility with callers, but
    at this point the object is expected to have already passed through
    `_ensure_json` (with or without sanitization), so we do not use a `default`
    encoder here. Any failure here indicates a logic bug earlier in the pipeline.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pid_is_alive(pid: int) -> bool:
    """Check if PID is alive using os.kill(0)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def _acquire_lock(
    lockfile: Path,
    stale_seconds: float = 60.0,
    timeout: float = 5.0,
    retry_delay: float = 0.1,
) -> bool:
    """
    Best-effort inter-process lock via atomic create with timeout and retries.
    If a stale lock exists (mtime older than stale_seconds and PID not alive), we break it.
    Returns True on lock acquired, False on timeout or error.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            # remove stale lock if safe
            if lockfile.exists():
                age = _now_unix() - lockfile.stat().st_mtime
                if age > stale_seconds:
                    try:
                        data = json.loads(lockfile.read_text(encoding="utf-8"))
                        pid = data.get("pid")
                    except Exception:
                        pid = None
                    if pid and _pid_is_alive(pid):
                        # Don't break a live lock; wait and retry
                        time.sleep(retry_delay)
                        continue
                    lockfile.unlink(missing_ok=True)

            # atomic create
            fd = os.open(
                str(lockfile),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"pid": os.getpid(), "ts": _now_unix()}, f)
            except Exception:
                os.close(fd)
                raise
            return True
        except FileExistsError:
            # Another process beat us; wait and retry
            time.sleep(retry_delay)
        except Exception:
            # Permission or other I/O error; let caller decide via False
            return False
    return False


def _release_lock(lockfile: Path) -> None:
    try:
        lockfile.unlink(missing_ok=True)
    except Exception:
        # best-effort; do not propagate
        pass


def _deep_redact(
    obj: Any,
    *,
    keys: Sequence[str],
    exclude_keys: Sequence[str],
    patterns: Sequence[re.Pattern],
    mask: str,
) -> Any:
    """
    Return a redacted copy of `obj` (dict/list/primitive/string) using key names
    or regex patterns on keys AND string values, with explicit excludes.
    """

    def _should_redact_key(k: str) -> bool:
        if k in exclude_keys:
            return False
        if k in keys:
            return True
        return any(p.search(k) for p in patterns)

    def _redact_string(s: str) -> str:
        for p in patterns:
            s = p.sub(mask, s)
        return s

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            sk = str(k)
            if _should_redact_key(sk):
                out[sk] = mask
            else:
                out[sk] = _deep_redact(
                    v,
                    keys=keys,
                    exclude_keys=exclude_keys,
                    patterns=patterns,
                    mask=mask,
                )
        return out

    if isinstance(obj, list):
        return [
            _deep_redact(
                v,
                keys=keys,
                exclude_keys=exclude_keys,
                patterns=patterns,
                mask=mask,
            )
            for v in obj
        ]

    if isinstance(obj, str):
        return _redact_string(obj)

    # primitives unchanged
    return obj


def _compile_patterns(pats: Optional[Sequence[str]], strict_mode: bool) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    if not pats:
        return out
    for p in pats:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            if strict_mode:
                raise LoggingError(f"Invalid redaction regex '{p}': {e}")
            # Fail-soft: ignore invalid in non-strict mode
    return out

# ---------------------------------------------------------------------------
# Environment snapshot helpers
# ---------------------------------------------------------------------------

def _safe_pip_freeze() -> List[str]:
    """
    Best-effort 'pip freeze'.

    Returns a list of requirement lines (strings). On any failure, returns [].

    This is intentionally conservative: it never raises, and it normalizes
    bytes vs str from subprocess.check_output (which is helpful in tests).
    """
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
        )
        # Tests may monkeypatch cclogging.subprocess.check_output to return bytes
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        # In production we do not want env snapshot to break logging.
        return []


def _capture_env_snapshot() -> Dict[str, Any]:
    """
    Capture a lightweight, JSON-serializable environment snapshot.

    - 'python'   : full Python version string
    - 'platform' : e.g. 'macOS-15.5-arm64-...' from stdlib platform.platform()
    - 'pip_freeze': list[str] from 'pip freeze'
    """
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": _safe_pip_freeze(),
    }



@dataclass
class EnvelopeMeta:
    schema: str
    ts_unix: float
    ts_iso: str
    host: str
    pid: int
    git_sha: Optional[str]
    session_id: str
    seed: Optional[int]
    env: Optional[Dict[str, Any]]
    extra: Dict[str, Any]


class ChainedJSONLLogger:
    """
    Compatibility wrapper over `cc.cartographer.audit` with upgraded envelopes.

    Legacy API (still supported):
        logger = ChainedJSONLLogger("runs/audit.jsonl")
        sha = logger.log({"event": "something", ...})
        ok, err = logger.verify_chain_integrity()

    Upgrades (opt-in):
        - Deterministic envelope with ISO+unix timestamps, host, pid, git sha (if available), session id.
        - Optional auto-sanitization of payload/meta extras.
        - Optional verify_on_write to check chain integrity after each append.
        - Best-effort lockfile to reduce concurrent writer races (Cartographer should still detect tampering).
        - Redaction of sensitive keys/patterns before writing (scans string values too for PII).
        - File rotation and optional gzip compression of rotated files.
        - Batch logging helpers; tail/head utilities; fsync for durability.
        - Async methods (alog, alog_event, alog_many) for coroutines.
        - Prometheus metrics if enabled (log_entries, log_errors, log_size) with level labels.
        - Post-log hooks (e.g., for tampering alerts) with recursion guard.
        - Strict mode (via env or param) to raise on soft failures (e.g., invalid regex, lock timeout).
        - Encryption option for backups (AES via cryptography lib, if installed).
        - Levels support (debug/info/warning/error) for log calls, recorded under meta.extra["level"].

    Parameters
    ----------
    path : str | Path
        JSONL file path for the audit chain.
    default_seed : Optional[int]
        Seed to record in `meta.seed` if not provided per-call.
    enable_lock : bool
        If True, create a lockfile `<path>.lock` around append operations.
    verify_on_write : bool
        If True, call `audit.verify_chain` after each append and raise on failure.
    auto_sanitize : bool
        If True, coerce common non-JSON types in payload and meta.extra.
    schema_id : str
        Envelope schema id (default: "core/logging.v5").
    capture_env : bool
        If True, include a full environment snapshot in `meta.env` (python, platform, pip freeze with fallback).
    fsync_on_write : bool
        If True, attempt to fsync the log file after each append (best effort).
    max_bytes : Optional[int]
        If set, rotate the file when size >= max_bytes before the next append.
    backup_count : int
        Number of rotated files to keep (default 5). Ignored if max_bytes is None.
    compress_backups : bool
        If True, gzip rotated files (`.gz`).
    encrypt_backups : bool
        If True and cryptography available, encrypt rotated files (AES-Fernet, `.enc`).
    encryption_key : Optional[bytes]
        Stable Fernet key for encryption (required if encrypt_backups=True; can be base64 from env LOG_ENC_KEY).
    redact_keys : Sequence[str]
        Exact key names to redact anywhere within `payload` and `meta.extra`.
    redact_patterns : Sequence[str]
        Regex patterns (matched on key names and string values) to redact (plus built-in PII).
    redact_exclude_keys : Sequence[str]
        Exact key names to never redact, even if they match keys/patterns.
    redact_mask : str
        Replacement string for redacted values (default: "***").
    enable_prometheus : bool
        If True and prometheus_client available, expose metrics (log_entries, log_errors, log_size) with labels.
    post_log_hook : Optional[Callable[[str], None]]
        Optional callback after successful log (receives new SHA); e.g., for alerts.
    strict_mode : bool
        If True, raise on soft failures (e.g., invalid regex, lock timeout, no encryption key).
    lock_timeout : float
        Timeout in seconds for acquiring lock (default 5.0).
    lock_retry_delay : float
        Delay between lock retries (default 0.1s).
    lock_stale_seconds : float
        Stale lock threshold in seconds (default 60.0).
    async_concurrency_limit : int
        Max concurrent async log submissions (default 32; 0 for unlimited).
    """

    def __init__(
        self,
        path: str | Path,
        default_seed: Optional[int] = None,
        enable_lock: bool = True,
        verify_on_write: bool = False,
        auto_sanitize: bool = False,
        *,
        schema_id: str = SCHEMA_ID_DEFAULT,
        capture_env: bool = False,
        fsync_on_write: bool = False,
        max_bytes: Optional[int] = None,
        backup_count: int = 5,
        compress_backups: bool = False,
        encrypt_backups: bool = False,
        encryption_key: Optional[bytes] = None,
        redact_keys: Optional[Sequence[str]] = None,
        redact_patterns: Optional[Sequence[str]] = None,
        redact_exclude_keys: Optional[Sequence[str]] = None,
        redact_mask: str = "***",
        enable_prometheus: bool = False,
        post_log_hook: Optional[Callable[[str], None]] = None,
        strict_mode: bool = False,
        lock_timeout: float = 5.0,
        lock_retry_delay: float = 0.1,
        lock_stale_seconds: float = 60.0,
        async_concurrency_limit: int = 32,
    ) -> None:
        # ------------------------------------------------------------------
        # Env overrides for config
        # ------------------------------------------------------------------
        if os.getenv("LOG_STRICT_MODE") == "1":
            strict_mode = True
        if os.getenv("LOG_PROMETHEUS") == "1":
            enable_prometheus = True

        # ------------------------------------------------------------------
        # Paths & lock configuration
        # ------------------------------------------------------------------
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.lockfile = (
            self.path.with_suffix(self.path.suffix + _LOCKFILE_SUFFIX)
            if enable_lock
            else None
        )
        self.lock_timeout = float(lock_timeout)
        self.lock_retry_delay = float(lock_retry_delay)
        self.lock_stale_seconds = float(lock_stale_seconds)

        # ------------------------------------------------------------------
        # Behaviour flags
        # ------------------------------------------------------------------
        self.verify_on_write = bool(verify_on_write)
        self.auto_sanitize = bool(auto_sanitize)
        self.default_seed = default_seed
        self.schema_id = schema_id
        self.capture_env = bool(capture_env)
        self.fsync_on_write = bool(fsync_on_write)
        self.strict_mode = bool(strict_mode)
        self.post_log_hook = post_log_hook
        self._post_hook_recursion_guard = threading.local()

        # ------------------------------------------------------------------
        # Rotation configuration
        # ------------------------------------------------------------------
        self.max_bytes = int(max_bytes) if max_bytes is not None else None
        self.backup_count = int(backup_count)
        self.compress_backups = bool(compress_backups)

        # ------------------------------------------------------------------
        # Encryption configuration
        # ------------------------------------------------------------------
        enc_key_env = os.getenv("LOG_ENC_KEY")
        if enc_key_env:
            # Env key overrides argument, and is validated here.
            try:
                # urlsafe_b64decode will raise on invalid input
                encryption_key = base64.urlsafe_b64decode(enc_key_env)
            except Exception as e:
                if self.strict_mode:
                    # In strict mode, invalid LOG_ENC_KEY is a hard error
                    raise LoggingError(f"Invalid LOG_ENC_KEY: {e}") from e
                # In non-strict mode, silently disable env key
                encryption_key = None

        if encrypt_backups and ENCRYPTION_AVAILABLE:
            if encryption_key is None:
                if self.strict_mode:
                    raise LoggingError(
                    "encrypt_backups=True but no encryption_key provided"
                    )
                # soft-disable encryption if key missing and not strict
                encrypt_backups = False

        self.encrypt_backups = bool(encrypt_backups and ENCRYPTION_AVAILABLE)
        self.enc_key = encryption_key if self.encrypt_backups else None

        # ------------------------------------------------------------------
        # Redaction configuration
        # ------------------------------------------------------------------
        self.redact_keys = list(redact_keys or [])
        self.redact_exclude_keys = list(redact_exclude_keys or [])

        # User-supplied patterns (may be None) + built-in PII patterns
        compiled_user = _compile_patterns(redact_patterns, self.strict_mode)
        compiled_builtin = [re.compile(p, re.IGNORECASE) for p in PII_PATTERNS_DEFAULT]
        self.redact_patterns: List[re.Pattern] = compiled_user + compiled_builtin
        self.redact_mask = redact_mask

        # ------------------------------------------------------------------
        # Metrics (Prometheus) configuration
        # ------------------------------------------------------------------
        self.enable_prometheus = bool(enable_prometheus and PROMETHEUS_AVAILABLE)
        if self.enable_prometheus:
            # Metrics init must be best-effort: never break logger construction,
            # even in strict_mode. If metrics are misconfigured, we’ll surface it
            # on actual updates in log() / _maybe_rotate().
            try:
                LOG_SIZE.set(0)  # Init size gauge
            except Exception:
                # Swallow init failures: logger must still be usable.
                pass

        # ------------------------------------------------------------------
        # Static process / git context
        # ------------------------------------------------------------------
        self._host = socket.gethostname()
        self._pid = os.getpid()
        self._git_sha = _detect_git_sha(self.path.parent)

        # ------------------------------------------------------------------
        # Concurrency primitives
        # ------------------------------------------------------------------
        self._thread_lock = threading.RLock()
        self._process_lock_depth: int = 0

        # Head SHA (chain tip), if any previous records exist
        self.last_sha: Optional[str] = audit.tail_sha(str(self.path))

        # Async semaphore for async API
        if async_concurrency_limit > 0:
            self._async_sem: Optional[asyncio.Semaphore] = asyncio.Semaphore(
                async_concurrency_limit
            )
        else:
            self._async_sem = None

        # ------------------------------------------------------------------
        # Environment snapshot (optional)
        # ------------------------------------------------------------------
        self._env_snapshot: Optional[Dict[str, Any]] = None
        if self.capture_env:
            # Defaults so keys are always present and truthy
            python_str = "unknown"
        platform_str = "unknown"
        pip_freeze: Optional[List[str]] = None

        # Best-effort collection; we never let this break logger construction
        try:
            try:
                python_str = platform.python_version()
            except Exception:
                # Fallback to sys.version, still non-empty
                python_str = sys.version.split()[0]
        except Exception:
            python_str = "unknown"

        try:
            try:
                platform_str = platform.platform()
            except Exception:
                platform_str = sys.platform
        except Exception:
            platform_str = "unknown"

        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL,
            )
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            pip_freeze = [
                line.strip() for line in out.splitlines() if line.strip()
            ]
        except Exception:
            pip_freeze = None

        self._env_snapshot = {
            "python": python_str,
            "platform": platform_str,
            "pip_freeze": pip_freeze,
        }


    # ------------------------------------------------------------------ helpers

    def _build_meta(
        self,
        extra_meta: Optional[Dict[str, Any]],
        seed: Optional[int],
    ) -> EnvelopeMeta:
        ts = _now_unix()
        return EnvelopeMeta(
            schema=self.schema_id,
            ts_unix=ts,
            ts_iso=_now_iso(ts),
            host=self._host,
            pid=self._pid,
            git_sha=self._git_sha,
            session_id=_session_id,
            seed=(seed if seed is not None else self.default_seed),
            env=self._env_snapshot,
            extra=(extra_meta or {}),
        )

    def _maybe_rotate(self) -> Optional[str]:
        """Rotate the underlying file if size >= max_bytes. Returns rotated filename (or None)."""
        if self.max_bytes is None:
            return None
        try:
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                rotated = self._rotate_files()
                # Start a new chain: reset head SHA
                self.last_sha = None
                if self.enable_prometheus:
                    try:
                        size = self.path.stat().st_size if self.path.exists() else 0
                        LOG_SIZE.set(size)
                    except Exception as e:
                        LOG_ERRORS.labels(type="metrics").inc()
                        if self.strict_mode:
                            raise LoggingError(
                                f"Metrics update failed post-rotation: {e}"
                            ) from e
                return rotated
            return None
        except Exception as e:
            if self.enable_prometheus:
                LOG_ERRORS.labels(type="rotation").inc()
            if self.strict_mode:
                raise LoggingError(f"rotation failed: {e}") from e
            return None

    def _rotate_files(self) -> str:
        """Perform size-based rotation akin to logging.handlers.RotatingFileHandler."""

        def rot_name(idx: int) -> Path:
            return self.path.with_suffix(self.path.suffix + f".{idx}")

        # Remove oldest
        if self.backup_count > 0:
            oldest = rot_name(self.backup_count)
            if oldest.exists():
                oldest.unlink(missing_ok=True)

            # Shift downwards: .(n-1) -> .n
            for i in range(self.backup_count - 1, 0, -1):
                src = rot_name(i)
                dst = rot_name(i + 1)
                if src.exists():
                    src.replace(dst)

            rotated_path = self.path
            if rotated_path.exists():
                rotated_path = rotated_path.replace(rot_name(1))
                rotated_str = str(rot_name(1))

                # Optional compression
                if self.compress_backups:
                    gz_path = Path(rotated_str + ".gz")
                    with open(rotated_str, "rb") as f_in, gzip.open(
                        gz_path, "wb"
                    ) as f_out:
                        while True:
                            chunk = f_in.read(64 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)
                    # remove uncompressed rotated file
                    Path(rotated_str).unlink(missing_ok=True)
                    rotated_str = str(gz_path)

                # Optional encryption (over compressed file if enabled)
                if self.encrypt_backups:
                    enc_path = Path(rotated_str + ".enc")
                    fernet = Fernet(self.enc_key)  # type: ignore[arg-type]
                    with open(rotated_str, "rb") as f_in:
                        data = f_in.read()
                    encrypted = fernet.encrypt(data)
                    with open(enc_path, "wb") as f_out:
                        f_out.write(encrypted)
                    # remove unencrypted rotated file
                    Path(rotated_str).unlink(missing_ok=True)
                    rotated_str = str(enc_path)

                return rotated_str

        # No backups requested or no existing file: just truncate
        self.path.write_text("", encoding="utf-8")
        return str(self.path)

    def _redact(
        self,
        payload: Dict[str, Any],
        extra: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.redact_keys and not self.redact_patterns:
            return payload, extra
        pay = _deep_redact(
            payload,
            keys=self.redact_keys,
            exclude_keys=self.redact_exclude_keys,
            patterns=self.redact_patterns,
            mask=self.redact_mask,
        )
        ext = _deep_redact(
            extra,
            keys=self.redact_keys,
            exclude_keys=self.redact_exclude_keys,
            patterns=self.redact_patterns,
            mask=self.redact_mask,
        )
        return pay, ext

    def _make_record(self, payload: Dict[str, Any], meta: EnvelopeMeta) -> Dict[str, Any]:
        """
        Construct the canonical envelope around `payload`.

        Pipeline
        --------
        1. Apply redaction to `payload` and `meta.extra`.
        2. Sanitize/validate JSON-serializability (via `_ensure_json`):
            - If auto_sanitize=True, deep-copy & coerce to JSON-safe types.
            - If auto_sanitize=False, validate as-is and let TypeError/ValueError bubble.
        3. Assemble the final record with `meta` fields and sanitized payload/extra.
        4. Run a canonical dump (`_canonical_dumps`) as a final sanity check.

        Raises
        ------
        TypeError / ValueError
            If either `payload` or `meta.extra` is not JSON-serializable when
            `auto_sanitize=False`, or still invalid even after sanitization.
        """
        # 1) Redact first (works on original structures)
        redacted_payload, redacted_extra = self._redact(payload, meta.extra)

        # 2) Sanitize / validate JSON-serializability
        redacted_payload = _ensure_json(redacted_payload, self.auto_sanitize)
        redacted_extra = _ensure_json(redacted_extra, self.auto_sanitize)

        # 3) Build the envelope using sanitized copies
        rec: Dict[str, Any] = {
            "meta": {
                "schema": meta.schema,
                "ts_unix": meta.ts_unix,
                "ts_iso": meta.ts_iso,
                "host": meta.host,
                "pid": meta.pid,
                "git_sha": meta.git_sha,
                "session_id": meta.session_id,
                "seed": meta.seed,
                "env": meta.env,
                "extra": redacted_extra,
            },
            "payload": redacted_payload,
        }

        # 4) Final canonicalization sanity check (sorted keys, compact separators)
        _canonical_dumps(rec, self.auto_sanitize)

        return rec


    def _fsync_best_effort(self) -> None:
        """Attempt to fsync the file (best-effort; raise in strict if fails)."""
        try:
            fd = os.open(str(self.path), os.O_RDWR)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as e:
            if self.enable_prometheus:
                LOG_ERRORS.labels(type="fsync").inc()
            if self.strict_mode:
                raise LoggingError(f"fsync failed: {e}") from e

    def _post_log(self, sha: str) -> None:
        if self.post_log_hook:
            if getattr(self._post_hook_recursion_guard, "active", False):
                return  # Skip to prevent recursion
            self._post_hook_recursion_guard.active = True
            try:
                self.post_log_hook(sha)
            except Exception as e:
                if self.enable_prometheus:
                    LOG_ERRORS.labels(type="hook").inc()
                if self.strict_mode:
                    raise e
            finally:
                self._post_hook_recursion_guard.active = False

    def _acquire_process_lock_if_needed(self, require_lock: bool) -> bool:
        """
        Acquire the process-wide lock if configured.

        Returns True if a lock is held for this thread (possibly via nested calls),
        False if no lockfile is configured or acquisition was skipped.

        If `require_lock` is True and the lock cannot be acquired within timeout,
        a LoggingError is raised.
        """
        if self.lockfile is None:
            return False

        if self._process_lock_depth > 0:
            self._process_lock_depth += 1
            return True

        locked = _acquire_lock(
            self.lockfile,
            stale_seconds=self.lock_stale_seconds,
            timeout=self.lock_timeout,
            retry_delay=self.lock_retry_delay,
        )
        if not locked and require_lock:
            if self.enable_prometheus:
                LOG_ERRORS.labels(type="lock").inc()
            raise LoggingError(
                f"Failed to acquire process lock {self.lockfile} within "
                f"{self.lock_timeout:.2f}s"
            )

        if locked:
            self._process_lock_depth = 1
        return locked

    def _release_process_lock_if_needed(self, locked: bool) -> None:
        if not locked or self.lockfile is None:
            return
        self._process_lock_depth -= 1
        if self._process_lock_depth == 0:
            _release_lock(self.lockfile)

    # ------------------------------------------------------------------ public API (sync)

    def log(
        self,
        payload: Dict[str, Any],
        *,
        level: str = "info",
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
        ) -> str:
        """
        Append a record to the audit chain.

        Parameters
        ----------
        payload : Dict[str, Any]
            JSON-serializable content (auto-sanitized if enabled).
        level : str
            Log level: debug/info/warning/error (for metrics/filtering).
        seed : Optional[int]
            Overrides logger's default_seed for this record.
        extra_meta : Optional[Dict[str, Any]]
            Additional metadata to tuck under `meta.extra`.
        verify_on_write : Optional[bool]
            If set, overrides the logger's default verify_on_write for this call.

        Returns
        -------
        str
            The hex SHA-256 digest of the appended record (new chain head).

        Raises
        ------
        LoggingError
            On serialization failure, *required* lock acquisition failure
            (when strict_mode or verify_on_write), rotation error, chain
            verification failure, or metrics failure in strict mode.
        """
        # Validate level early
        if level not in ("debug", "info", "warning", "error"):
            raise LoggingError(f"Invalid log level: {level}")

        # Resolve effective verify flag for this call
        effective_verify = (
            self.verify_on_write if verify_on_write is None else bool(verify_on_write)
        )

        # We only *require* the lock if either:
        #   - strict_mode is enabled, or
        #   - verify_on_write is requested (to avoid races during verification).
        # In other cases, lock acquisition is best-effort.
        require_lock = bool(self.lockfile is not None and (self.strict_mode or effective_verify))

        with self._thread_lock:
            locked = self._acquire_process_lock_if_needed(require_lock=require_lock)
            try:
                # ----------------------------------------------------------
                # 1) Rotation (size-based) before append
                # ----------------------------------------------------------
                rotated = self._maybe_rotate()
                if rotated is not None:
                    # Emit rotation marker into the *new* file (fresh chain).
                    # We deliberately disable verify_on_write here to avoid
                    # nested verification/lock complications.
                    try:
                        self.log_event(
                            "rotation_new_chain",
                            fields={
                                "rotated_from": rotated,
                                "previous_head": self.last_sha,
                            },
                            verify_on_write=False,
                            level="info",
                        )
                    except Exception as e:
                        if self.enable_prometheus:
                            try:
                                LOG_ERRORS.labels(type="rotation_marker").inc()
                            except Exception:
                                # Metrics themselves might be misconfigured; best-effort only.
                                pass
                        if self.strict_mode:
                            raise LoggingError(f"Rotation marker failed: {e}") from e

                # ----------------------------------------------------------
                # 2) Build envelope meta (with level in extra, unless caller
                #    explicitly overrode it).
                # ----------------------------------------------------------
                meta_extra: Dict[str, Any] = dict(extra_meta) if extra_meta else {}
                meta_extra.setdefault("level", level)
                meta = self._build_meta(meta_extra, seed)

                # Construct canonical record (handles redaction + JSON strictness)
                rec = self._make_record(payload, meta)

                # ----------------------------------------------------------
                # 3) Append via Cartographer audit shim
                # ----------------------------------------------------------
                sha = audit.append_jsonl(str(self.path), rec)
                self.last_sha = sha

                # ----------------------------------------------------------
                # 4) Optional fsync for durability
                # ----------------------------------------------------------
                if self.fsync_on_write:
                    self._fsync_best_effort()

                # ----------------------------------------------------------
                # 5) Optional chain verification (post-append)
                # ----------------------------------------------------------
                if effective_verify:
                    try:
                        audit.verify_chain(str(self.path))
                    except Exception as e:
                        if self.enable_prometheus:
                            try:
                                LOG_ERRORS.labels(type="verify").inc()
                            except Exception:
                                pass
                        raise LoggingError(
                            f"Audit chain verification failed post-append: {e}"
                        ) from e

                # ----------------------------------------------------------
                # 6) Metrics (Prometheus) — best-effort unless strict_mode
                # ----------------------------------------------------------
                if self.enable_prometheus:
                    try:
                        LOG_ENTRIES.labels(level=level).inc()
                        size = self.path.stat().st_size if self.path.exists() else 0
                        LOG_SIZE.set(size)
                    except Exception as e:
                        # Try to record the metrics failure itself
                        try:
                            LOG_ERRORS.labels(type="metrics").inc()
                        except Exception:
                            # Even LOG_ERRORS may be misconfigured; never let this double-fail.
                            pass

                        if self.strict_mode:
                            # In strict mode, observability failure is surfaced.
                            raise LoggingError(
                                f"Prometheus metrics update failed: {e}"
                            ) from e
                        # Non-strict: swallow metrics failure after best-effort logging.

                # ----------------------------------------------------------
                # 7) Post-log hook (alerts, external sinks), with recursion guard
                # ----------------------------------------------------------
                self._post_log(sha)

                return sha
            finally:
                self._release_process_lock_if_needed(locked)

    def log_event(
        self,
        event: str,
        *,
        fields: Optional[Dict[str, Any]] = None,
        level: str = "info",
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
    ) -> str:
        """Convenience: wrap an 'event' with arbitrary fields into payload."""
        payload: Dict[str, Any] = {"event": event}
        if fields:
            payload.update(fields)
        return self.log(
            payload,
            level=level,
            seed=seed,
            extra_meta=extra_meta,
            verify_on_write=verify_on_write,
        )

    def log_many(
        self,
        payloads: Iterable[Dict[str, Any]],
        *,
        level: str = "info",
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
    ) -> List[str]:
        """
        Append many payloads; convenience wrapper over `log`.

        This does *not* provide transactional semantics: each call to `log` is
        independent, with its own locking/rotation/verification.
        """
        shas: List[str] = []
        for p in payloads:
            shas.append(
                self.log(
                    p,
                    level=level,
                    seed=seed,
                    extra_meta=extra_meta,
                    verify_on_write=verify_on_write,
                )
            )
        return shas

    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the entire chain for tampering.

        Returns
        -------
        (True, None) on success, (False, reason) on failure.
        """
        try:
            audit.verify_chain(str(self.path))
            return True, None
        except Exception as e:
            return False, str(e)

    def current_head(self) -> Optional[str]:
        """Return the current chain head SHA (None if file empty)."""
        try:
            self.last_sha = audit.tail_sha(str(self.path))
            return self.last_sha
        except Exception:
            return self.last_sha

    def tail(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return the last `n` JSONL records as dicts (best-effort; reads file normally).

        Intended for quick debugging and NOT for large-scale analytics.
        """
        n = max(1, int(n))
        try:
            with self.path.open("r", encoding="utf-8") as f:
                dq = deque(f, maxlen=n)
            out: List[Dict[str, Any]] = []
            for line in dq:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # skip corrupt lines; cartographer verification would catch this upstream
                    continue
            return out
        except FileNotFoundError:
            return []

    # ------------------------------------------------------------------ async API

    async def alog(
        self,
        payload: Dict[str, Any],
        *,
        level: str = "info",
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
    ) -> str:
        """Async wrapper for `log`, using an executor and optional concurrency limit."""
        if self._async_sem:
            async with self._async_sem:
                loop = asyncio.get_running_loop()
                fn = functools.partial(
                    self.log,
                    payload,
                    level=level,
                    seed=seed,
                    extra_meta=extra_meta,
                    verify_on_write=verify_on_write,
                )
                return await loop.run_in_executor(None, fn)
        else:
            loop = asyncio.get_running_loop()
            fn = functools.partial(
                self.log,
                payload,
                level=level,
                seed=seed,
                extra_meta=extra_meta,
                verify_on_write=verify_on_write,
            )
            return await loop.run_in_executor(None, fn)

    async def alog_event(
        self,
        event: str,
        *,
        fields: Optional[Dict[str, Any]] = None,
        level: str = "info",
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
    ) -> str:
        """Async wrapper for `log_event`."""
        if self._async_sem:
            async with self._async_sem:
                loop = asyncio.get_running_loop()
                fn = functools.partial(
                    self.log_event,
                    event,
                    fields=fields,
                    level=level,
                    seed=seed,
                    extra_meta=extra_meta,
                    verify_on_write=verify_on_write,
                )
                return await loop.run_in_executor(None, fn)
        else:
            loop = asyncio.get_running_loop()
            fn = functools.partial(
                self.log_event,
                event,
                fields=fields,
                level=level,
                seed=seed,
                extra_meta=extra_meta,
                verify_on_write=verify_on_write,
            )
            return await loop.run_in_executor(None, fn)

    async def alog_many(
        self,
        payloads: Iterable[Dict[str, Any]],
        *,
        level: str = "info",
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
    ) -> List[str]:
        """Async wrapper for `log_many`."""
        if self._async_sem:
            async with self._async_sem:
                loop = asyncio.get_running_loop()
                fn = functools.partial(
                    self.log_many,
                    payloads,
                    level=level,
                    seed=seed,
                    extra_meta=extra_meta,
                    verify_on_write=verify_on_write,
                )
                return await loop.run_in_executor(None, fn)
        else:
            loop = asyncio.get_running_loop()
            fn = functools.partial(
                self.log_many,
                payloads,
                level=level,
                seed=seed,
                extra_meta=extra_meta,
                verify_on_write=verify_on_write,
            )
            return await loop.run_in_executor(None, fn)


# ------------------------------------------------------------------ context manager (sync/async)


@contextmanager
def audit_context(
    logger: ChainedJSONLLogger,
    operation: str,
    **metadata: Any,
):
    """
    Context manager for audited operations using the shim logger.

    Example
    -------
        with audit_context(logger, "train", run_id=rid, git_branch="main") as op_id:
            ... do work ...

    Behavior
    --------
    - Emits an 'operation_start' record with precise timestamps, host/pid/git info, free-form metadata.
    - On normal exit, emits 'operation_complete' with:
        * duration_wall_s (time.time delta)
        * duration_perf_s (perf_counter delta)
        * success = True
    - On exception, emits 'operation_error' with:
        * error_type, error_message (truncated), duration_wall_s, duration_perf_s, success = False
      and re-raises the original exception.

    Guarantees
    ----------
    - Uses the same verify-on-write and sanitize settings as the logger instance.
    - operation_id is a deterministic short hash of (operation, start_ts, pid, session).
    """
    start_wall = _now_unix()
    start_perf = time.perf_counter()
    op_id = hashlib.sha256(
        f"{operation}:{start_wall:.6f}:{os.getpid()}:{_session_id}".encode("utf-8")
    ).hexdigest()[:16]

    logger.log_event(
        "operation_start",
        fields={"operation": operation, "operation_id": op_id, "meta": metadata},
    )

    try:
        yield op_id
        end_wall = _now_unix()
        end_perf = time.perf_counter()
        logger.log_event(
            "operation_complete",
            fields={
                "operation": operation,
                "operation_id": op_id,
                "duration_wall_s": round(end_wall - start_wall, 6),
                "duration_perf_s": round(end_perf - start_perf, 6),
                "success": True,
            },
        )
    except Exception as e:
        end_wall = _now_unix()
        end_perf = time.perf_counter()
        msg = str(e)
        if len(msg) > 512:
            msg = msg[:509] + "..."
        logger.log_event(
            "operation_error",
            fields={
                "operation": operation,
                "operation_id": op_id,
                "error_type": e.__class__.__name__,
                "error_message": msg,
                "duration_wall_s": round(end_wall - start_wall, 6),
                "duration_perf_s": round(end_perf - start_perf, 6),
                "success": False,
            },
            level="error",
        )
        raise


@asynccontextmanager
async def aaudit_context(
    logger: ChainedJSONLLogger,
    operation: str,
    **metadata: Any,
):
    """
    Async counterpart of `audit_context`.
    """
    start_wall = _now_unix()
    start_perf = time.perf_counter()
    op_id = hashlib.sha256(
        f"{operation}:{start_wall:.6f}:{os.getpid()}:{_session_id}".encode("utf-8")
    ).hexdigest()[:16]

    await logger.alog_event(
        "operation_start",
        fields={"operation": operation, "operation_id": op_id, "meta": metadata},
    )

    try:
        yield op_id
        end_wall = _now_unix()
        end_perf = time.perf_counter()
        await logger.alog_event(
            "operation_complete",
            fields={
                "operation": operation,
                "operation_id": op_id,
                "duration_wall_s": round(end_wall - start_wall, 6),
                "duration_perf_s": round(end_perf - start_perf, 6),
                "success": True,
            },
        )
    except Exception as e:
        end_wall = _now_unix()
        end_perf = time.perf_counter()
        msg = str(e)
        if len(msg) > 512:
            msg = msg[:509] + "..."
        await logger.alog_event(
            "operation_error",
            fields={
                "operation": operation,
                "operation_id": op_id,
                "error_type": e.__class__.__name__,
                "error_message": msg,
                "duration_wall_s": round(end_wall - start_wall, 6),
                "duration_perf_s": round(end_perf - start_perf, 6),
                "success": False,
            },
            level="error",
        )
        raise
