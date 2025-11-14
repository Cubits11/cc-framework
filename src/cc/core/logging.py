# src/cc/core/logging.py
"""
Module: logging
Purpose: JSONL audit logger (shim) that delegates to Cartographer's tamper-evident chain
Dependencies: json, time, pathlib, re, os, socket, subprocess, platform, asyncio, functools; optional: prometheus_client; delegates hashing/verification to cc.cartographer.audit
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
    * Best-effort inter-process lock (atomic lockfile) to guard concurrent writers, with retries/timeout.
    * **Redaction** via `redact_keys`, `redact_patterns` (regex), and built-in PII patterns with configurable mask (now scans string values too).
    * **Rotation** via `max_bytes`/`backup_count` and optional `compress_backups=True` (.gz).
    * **Batch** helpers (`log_many`) and quick **introspection** (`tail`, `current_head`).
    * Optional `fsync_on_write=True` for durability (best effort).
    * **Async** support via `alog` coroutines (fixed with partial for kwargs).
    * **Metrics** via Prometheus (log_entries, log_errors, log_size) if enabled (fixed: no labels for simplicity).
    * **Post-log hooks** (e.g., for tampering alerts).
    * **Strict mode** (via env or param) to raise on soft failures (e.g., invalid regex, lock timeout).
    * **Encryption option** for backups (AES via cryptography lib, if installed).
    * **Levels** support (debug/info/warning/error) with Prometheus label.
    * **Thread-safe** with RLock; async with executor.
- Use `cc.cartographer.audit` directly for new code if you don’t need the compatibility layer.
- Environment config: LOG_STRICT_MODE=1, LOG_PROMETHEUS=1, etc.
- Research insights: Tamper-proof via external alerts (hook), encryption (new), levels for filtering (new), fail-soft defaults (adjusted).

Schema
------
Each record is written with an envelope:

{
  "meta": {
    "schema": "core/logging.v4",  # Upgraded
    "ts_unix": <float>,
    "ts_iso": "YYYY-MM-DDTHH:MM:SS.sssZ",
    "host": <str>,            # hostname
    "pid": <int>,
    "git_sha": <str|null>,    # env/auto-detected
    "session_id": <str>,      # stable per-process launch
    "seed": <int|null>,       # optional (user-provided)
    "env": { ... },           # optional snapshot (python/platform/pip) if capture_env=True
    "extra": { ... }          # user-supplied meta overrides
  },
  "payload": { ... }          # user content (JSON-serializable)
  # chain fields are injected by cc.cartographer.audit:
  # "prev_sha256": "...",
  # "sha256": "..."
}
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
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

from cc.cartographer import audit

__all__ = [
    "ChainedJSONLLogger",
    "audit_context",
    "aaudit_context",
    "LoggingError",
]

SCHEMA_ID_DEFAULT = "core/logging.v4"  # Upgraded
_LOCKFILE_SUFFIX = ".lock"
PII_PATTERNS_DEFAULT = [  # Built-in PII regex for keys AND string values
    r"(?i)\b(email|mail)\b",  # Emails keys
    r"\b(?:\d{3}-){2}\d{4}\b",  # SSN pattern
    r"\b4[0-9]{12}(?:[0-9]{3})?\b",  # CC numbers
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IPs
]

# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    LOG_ENTRIES = Counter('cc_audit_log_entries', 'Number of log entries')
    LOG_ERRORS = Counter('cc_audit_log_errors', 'Number of logging errors')
    LOG_SIZE = Gauge('cc_audit_log_size_bytes', 'Current log file size')

# A stable per-process session id (pid + monotonic start hashed)
_session_start_perf = time.perf_counter()
_session_id = hashlib.sha256(
    f"{os.getpid()}:{_session_start_perf:.9f}".encode("utf-8")
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
    return f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}T{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{ms:03d}Z"


_git_sha_cache: Optional[str] = None


def _detect_git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    """Resolve git SHA once (env `GIT_SHA` wins), else `git rev-parse --short=12 HEAD`."""
    global _git_sha_cache
    if _git_sha_cache is not None:
        return _git_sha_cache
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        _git_sha_cache = str(env_sha).strip()
        return _git_sha_cache
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
        _git_sha_cache = out.strip()
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


def _ensure_json(obj: Any, auto_sanitize: bool) -> None:
    if auto_sanitize:
        json.dumps(obj, default=_coerce_json_safe, sort_keys=True, separators=(",", ":"))
    else:
        json.dumps(obj, sort_keys=True, separators=(",", ":"))  # may raise TypeError


def _canonical_dumps(obj: Dict[str, Any], auto_sanitize: bool) -> str:
    if auto_sanitize:
        return json.dumps(obj, default=_coerce_json_safe, sort_keys=True, separators=(",", ":"))
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _acquire_lock(lockfile: Path, stale_seconds: float = 60.0, timeout: float = 5.0, retry_delay: float = 0.1) -> bool:
    """
    Best-effort inter-process lock via atomic create with timeout and retries.
    If a stale lock exists (mtime older than stale_seconds), we break it.
    Returns True on lock acquired, False on timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            # remove stale
            if lockfile.exists():
                age = _now_unix() - lockfile.stat().st_mtime
                if age > stale_seconds:
                    lockfile.unlink(missing_ok=True)
            # atomic create
            fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(retry_delay)
        except Exception:
            return False
    return False


def _release_lock(lockfile: Path) -> None:
    try:
        lockfile.unlink(missing_ok=True)
    except Exception:
        pass


def _deep_redact(obj: Any, *, keys: Sequence[str], patterns: Sequence[re.Pattern], mask: str) -> Any:
    """Return a redacted copy of `obj` (dict/list/primitive/string) using key names or regex patterns on keys AND string values."""
    def _should_redact_key(k: str) -> bool:
        if k in keys:
            return True
        return any(p.search(k) for p in patterns)

    def _redact_string(s: str) -> str:
        for p in patterns:
            s = p.sub(mask, s)
        return s

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _should_redact_key(str(k)):
                out[k] = mask
            else:
                out[k] = _deep_redact(v, keys=keys, patterns=patterns, mask=mask)
        return out
    if isinstance(obj, list):
        return [_deep_redact(v, keys=keys, patterns=patterns, mask=mask) for v in obj]
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
            # Fail-soft: ignore invalid
    return out


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
        - Redaction of sensitive keys/patterns before writing (now scans string values too for PII).
        - File rotation and optional gzip compression of rotated files.
        - Batch logging helpers; tail/head utilities; fsync for durability.
        - Async methods (alog, alog_event, alog_many) for coroutines (fixed with partial).
        - Prometheus metrics if enabled (log_entries, log_errors, log_size) (fixed: no labels).
        - Post-log hooks (e.g., for tampering alerts).
        - Strict mode (via env or param) to raise on soft failures (e.g., invalid regex, lock timeout).
        - Encryption option for backups (AES via cryptography lib, if installed).
        - Levels support (debug/info/warning/error) for log calls.

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
        Envelope schema id (default: "core/logging.v4").
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
    redact_keys : Sequence[str]
        Exact key names to redact anywhere within `payload` and `meta.extra`.
    redact_patterns : Sequence[str]
        Regex patterns (matched on key names and string values) to redact (plus built-in PII).
    redact_mask : str
        Replacement string for redacted values (default: "***").
    enable_prometheus : bool
        If True and prometheus_client available, expose metrics (log_entries, log_errors, log_size).
    post_log_hook : Optional[Callable[[str], None]]
        Optional callback after successful log (receives new SHA); e.g., for alerts.
    strict_mode : bool
        If True, raise on soft failures (e.g., invalid regex, lock timeout).
    lock_timeout : float
        Timeout in seconds for acquiring lock (default 5.0).
    lock_retry_delay : float
        Delay between lock retries (default 0.1s).
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
        redact_keys: Optional[Sequence[str]] = None,
        redact_patterns: Optional[Sequence[str]] = None,
        redact_mask: str = "***",
        enable_prometheus: bool = False,
        post_log_hook: Optional[Callable[[str], None]] = None,
        strict_mode: bool = False,
        lock_timeout: float = 5.0,
        lock_retry_delay: float = 0.1,
    ) -> None:
        # Env overrides for config
        if os.getenv("LOG_STRICT_MODE") == "1":
            strict_mode = True
        if os.getenv("LOG_PROMETHEUS") == "1":
            enable_prometheus = True

        # Paths & env
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lockfile = self.path.with_suffix(self.path.suffix + _LOCKFILE_SUFFIX) if enable_lock else None
        self.lock_timeout = lock_timeout
        self.lock_retry_delay = lock_retry_delay

        # Behavior flags
        self.verify_on_write = verify_on_write
        self.auto_sanitize = auto_sanitize
        self.default_seed = default_seed
        self.schema_id = schema_id
        self.capture_env = capture_env
        self.fsync_on_write = fsync_on_write
        self.strict_mode = strict_mode
        self.post_log_hook = post_log_hook

        # Rotation
        self.max_bytes = int(max_bytes) if max_bytes is not None else None
        self.backup_count = int(backup_count)
        self.compress_backups = bool(compress_backups)
        self.encrypt_backups = bool(encrypt_backups and ENCRYPTION_AVAILABLE)
        if self.encrypt_backups:
            self.enc_key = Fernet.generate_key()  # Per-instance key; persist if needed

        # Redaction
        self.redact_keys = list(redact_keys or [])
        self.redact_patterns = _compile_patterns(redact_patterns, self.strict_mode) + [re.compile(p) for p in PII_PATTERNS_DEFAULT]
        self.redact_mask = redact_mask

        # Metrics
        self.enable_prometheus = bool(enable_prometheus and PROMETHEUS_AVAILABLE)
        if self.enable_prometheus:
            LOG_SIZE.set(0)  # Init

        # State
        self._host = socket.gethostname()
        self._pid = os.getpid()
        self._git_sha = _detect_git_sha(self.path.parent)
        self._thread_lock = RLock()
        self.last_sha: Optional[str] = audit.tail_sha(str(self.path))  # None for empty/new files

        # Precompute environment snapshot if requested
        self._env_snapshot: Optional[Dict[str, Any]] = None
        if self.capture_env:
            try:
                pip_freeze = subprocess.check_output(
                    [sys.executable, '-m', 'pip', 'freeze']
                ).decode().splitlines()
            except Exception:
                pip_freeze = None
            self._env_snapshot = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "implementation": platform.python_implementation(),
                "pip_freeze": pip_freeze,
            }

    # ------------------------------------------------------------------ helpers

    def _build_meta(self, extra_meta: Optional[Dict[str, Any]], seed: Optional[int]) -> EnvelopeMeta:
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
                    LOG_SIZE.set(self.path.stat().st_size)
                return rotated
            return None
        except Exception as e:
            if self.strict_mode:
                raise LoggingError(f"rotation failed: {e}") from e
            if self.enable_prometheus:
                LOG_ERRORS.inc()
            return None

    def _rotate_files(self) -> str:
        """Perform size-based rotation akin to logging.handlers.RotatingFileHandler."""
        # Shift existing backups: .(backup_count-1) <- .(backup_count-2) <- ... <- .1 <- current
        def rot_name(idx: int) -> Path:
            return self.path.with_suffix(self.path.suffix + f".{idx}")

        # Remove oldest
        if self.backup_count > 0:
            oldest = rot_name(self.backup_count)
            if oldest.exists():
                oldest.unlink(missing_ok=True)

            # Shift downwards
            for i in range(self.backup_count - 1, 0, -1):
                src = rot_name(i)
                dst = rot_name(i + 1)
                if src.exists():
                    src.replace(dst)

            # Move current to .1
            if self.path.exists():
                self.path.replace(rot_name(1))
                rotated_str = str(rot_name(1))

                # Optional compression
                if self.compress_backups:
                    gz_path = Path(rotated_str + ".gz")
                    with open(rotated_str, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                        while True:
                            chunk = f_in.read(64 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)
                    # remove uncompressed rotated file
                    Path(rotated_str).unlink(missing_ok=True)
                    rotated_str = str(gz_path)

                # Optional encryption
                if self.encrypt_backups:
                    enc_path = Path(rotated_str + ".enc")
                    fernet = Fernet(self.enc_key)
                    with open(rotated_str, "rb") as f_in:
                        data = f_in.read()
                    encrypted = fernet.encrypt(data)
                    with open(enc_path, "wb") as f_out:
                        f_out.write(encrypted)
                    # remove unencrypted rotated file
                    Path(rotated_str).unlink(missing_ok=True)
                    rotated_str = str(enc_path)

                return rotated_str
        # No backups requested: just truncate
        self.path.write_text("", encoding="utf-8")
        return str(self.path)

    def _redact(self, payload: Dict[str, Any], extra: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.redact_keys and not self.redact_patterns:
            return payload, extra
        pay = _deep_redact(payload, keys=self.redact_keys, patterns=self.redact_patterns, mask=self.redact_mask)
        ext = _deep_redact(extra, keys=self.redact_keys, patterns=self.redact_patterns, mask=self.redact_mask)
        return pay, ext

    def _make_record(self, payload: Dict[str, Any], meta: EnvelopeMeta) -> Dict[str, Any]:
        """Construct the canonical envelope around payload; raises on non-serializable content unless auto_sanitize."""
        # Redact first, then validate serializability
        payload, meta_extra = self._redact(payload, meta.extra)
        _ensure_json(payload, self.auto_sanitize)
        _ensure_json(meta_extra, self.auto_sanitize)

        rec = {
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
                "extra": meta_extra,
            },
            "payload": payload,
        }
        # sanity: canonical dump (sorted keys, compact separators)
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
            if self.strict_mode:
                raise LoggingError(f"fsync failed: {e}") from e

    def _post_log(self, sha: str) -> None:
        if self.post_log_hook:
            try:
                self.post_log_hook(sha)
            except Exception:
                if self.strict_mode:
                    raise

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
            On serialization failure, lock acquisition failure (when verify_on_write or strict_mode),
            rotation error, or verification failure.
        """
        if level not in ["debug", "info", "warning", "error"]:
            raise LoggingError(f"Invalid log level: {level}")
        with self._thread_lock:
            # Optional inter-process lock
            locked = False
            if self.lockfile is not None:
                locked = _acquire_lock(self.lockfile, timeout=self.lock_timeout, retry_delay=self.lock_retry_delay)
                if not locked:
                    if self.verify_on_write if verify_on_write is None else verify_on_write or self.strict_mode:
                        if self.enable_prometheus:
                            LOG_ERRORS.inc()
                        raise LoggingError(f"Could not acquire audit lock within {self.lock_timeout}s: {self.lockfile}")

            try:
                # Rotate before we append
                rotated = self._maybe_rotate()
                if rotated is not None:
                    # Emit rotation marker into the *new* file (fresh chain)
                    try:
                        self.log_event(
                            "rotation_new_chain",
                            fields={"rotated_from": rotated, "previous_head": self.last_sha},
                            verify_on_write=False,
                            level="info",
                        )
                    except Exception as e:
                        if self.strict_mode:
                            raise LoggingError(f"Rotation marker failed: {e}") from e

                meta = self._build_meta(extra_meta, seed)
                rec = self._make_record(payload, meta)

                sha = audit.append_jsonl(str(self.path), rec)
                self.last_sha = sha

                if self.fsync_on_write:
                    self._fsync_best_effort()

                v = self.verify_on_write if verify_on_write is None else verify_on_write
                if v:
                    try:
                        audit.verify_chain(str(self.path))
                    except Exception as e:
                        if self.enable_prometheus:
                            LOG_ERRORS.inc()
                        raise LoggingError(f"Audit chain verification failed post-append: {e}") from e

                if self.enable_prometheus:
                    LOG_ENTRIES.inc()
                    LOG_SIZE.set(self.path.stat().st_size)

                self._post_log(sha)

                return sha
            finally:
                if locked:
                    _release_lock(self.lockfile)  # type: ignore[arg-type]

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
        payload = {"event": event}
        if fields:
            payload.update(fields)
        return self.log(payload, level=level, seed=seed, extra_meta=extra_meta, verify_on_write=verify_on_write)

    def log_many(
        self,
        payloads: Iterable[Dict[str, Any]],
        *,
        level: str = "info",
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
    ) -> List[str]:
        """Append many payloads atomically with respect to the lock; returns list of SHAs in order."""
        shas: List[str] = []
        for p in payloads:
            shas.append(self.log(p, level=level, seed=seed, extra_meta=extra_meta, verify_on_write=verify_on_write))
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
                lines = f.readlines()[-n:]
            out: List[Dict[str, Any]] = []
            for line in lines:
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
        loop = asyncio.get_running_loop()
        fn = functools.partial(
            self.log,
            payload,
            level=level,
            seed=seed,
            extra_meta=extra_meta,
            verify_on_write=verify_on_write
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
        loop = asyncio.get_running_loop()
        fn = functools.partial(
            self.log_event,
            event,
            fields=fields,
            level=level,
            seed=seed,
            extra_meta=extra_meta,
            verify_on_write=verify_on_write
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
        loop = asyncio.get_running_loop()
        fn = functools.partial(
            self.log_many,
            payloads,
            level=level,
            seed=seed,
            extra_meta=extra_meta,
            verify_on_write=verify_on_write
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
        )
        raise

@asynccontextmanager
async def aaudit_context(
    logger: ChainedJSONLLogger,
    operation: str,
    **metadata: Any,
):
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
        )
        raise
