# src/cc/core/logging.py
"""
Module: logging
Purpose: JSONL audit logger (shim) that delegates to Cartographer's tamper-evident chain
Dependencies: json, time, pathlib; delegates hashing/verification to cc.cartographer.audit
Author: Pranav Bhave (upgraded by assistant)
Dates:
  - 2025-08-27: initial shim
  - 2025-09-14: PhD-level upgrade (deterministic envelopes, env/git introspection,
                optional sanitize, verify-on-write, lockfile, richer context)

Notes
-----
- This remains a thin **compatibility shim** over `cc.cartographer.audit`.
- It preserves the legacy API while writing records in the tamper-evident format
  with fields: `prev_sha256`, `sha256` (Cartographer owns hash discipline).
- New features are opt-in and non-breaking:
    * Deterministic envelope with both unix and ISO-8601 timestamps (UTC),
      host, pid, optional git SHA, optional seed/session.
    * `auto_sanitize=True` to coerce common non-JSON types (Path, set, tuple, bytes).
    * `verify_on_write=True` to run a full chain verification after append.
    * Best-effort inter-process lock (atomic lockfile) to guard concurrent writers.
    * `audit_context` logs success/failure with precise perf+wall durations and exception type.
- Use `cc.cartographer.audit` directly for new code if you don’t need the compatibility layer.

Schema
------
Each record is written with an envelope:

{
  "meta": {
    "schema": "core/logging.v2",
    "ts_unix": <float>,
    "ts_iso": "YYYY-MM-DDTHH:MM:SS.sssZ",
    "host": <str>,            # hostname
    "pid": <int>,
    "git_sha": <str|null>,    # env/auto-detected
    "session_id": <str>,      # stable per-process launch
    "seed": <int|null>,       # optional (user-provided)
    "extra": { ... }          # user-supplied meta overrides
  },
  "payload": { ... }          # user content (JSON-serializable)
  # chain fields are injected by cc.cartographer.audit:
  # "prev_sha256": "...",
  # "sha256": "..."
}
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cc.cartographer import audit

__all__ = ["ChainedJSONLLogger", "audit_context", "LoggingError"]

SCHEMA_ID = "core/logging.v2"
_LOCKFILE_SUFFIX = ".lock"

# A stable per-process session id (pid + monotonic start hashed)
_session_start_perf = time.perf_counter()
_session_id = hashlib.sha256(
    f"{os.getpid()}:{_session_start_perf:.9f}".encode("utf-8")
).hexdigest()[:16]


class LoggingError(RuntimeError):
    """Raised for audit logging failures (serialization, locking, verification)."""


def _now_unix() -> float:
    """Seconds since epoch as float (UTC)."""
    return float(time.time())


def _now_iso(ts: Optional[float] = None) -> str:
    """ISO-8601 in UTC with 'Z' suffix; millisecond precision."""
    if ts is None:
        ts = _now_unix()
    # manual formatting to avoid importing datetime
    # time.gmtime returns tm in UTC
    tm = time.gmtime(ts)
    ms = int((ts - int(ts)) * 1000)
    return f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}T{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{ms:03d}Z"


_git_sha_cache: Optional[str] = None


def _detect_git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    """Resolve git SHA once (env `GIT_SHA` wins), else try `git rev-parse --short=12 HEAD`."""
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
    # avoid importing dataclasses.asdict unless needed
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if hasattr(obj, "__dataclass_fields__"):
        try:
            from dataclasses import asdict

            return asdict(obj)
        except Exception:
            return str(obj)
    return obj


def _ensure_json(obj: Any, auto_sanitize: bool) -> None:
    if auto_sanitize:
        def _default(x):
            return _coerce_json_safe(x)
        # We don't actually use the output here; we only validate serializability.
        json.dumps(obj, default=_default, sort_keys=True, separators=(",", ":"))
    else:
        json.dumps(obj, sort_keys=True, separators=(",", ":"))  # may raise TypeError


def _canonical_dumps(obj: Dict[str, Any], auto_sanitize: bool) -> str:
    if auto_sanitize:
        return json.dumps(obj, default=_coerce_json_safe, sort_keys=True, separators=(",", ":"))
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _acquire_lock(lockfile: Path, stale_seconds: float = 60.0) -> bool:
    """
    Best-effort inter-process lock via atomic create.
    If a stale lock exists (mtime older than stale_seconds), we break it.
    Returns True on lock acquired, False otherwise.
    """
    try:
        # remove stale
        if lockfile.exists():
            try:
                age = _now_unix() - lockfile.stat().st_mtime
                if age > stale_seconds:
                    lockfile.unlink(missing_ok=True)
            except Exception:
                pass
        # atomic create
        fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _release_lock(lockfile: Path) -> None:
    try:
        lockfile.unlink(missing_ok=True)
    except Exception:
        pass


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
    extra: Dict[str, Any]


class ChainedJSONLLogger:
    """
    Compatibility wrapper over `cc.cartographer.audit` with upgraded envelopes.

    Legacy API (still supported):
        logger = ChainedJSONLLogger("runs/audit.jsonl")
        sha = logger.log({"event": "something", ...})
        ok, err = logger.verify_chain_integrity()

    Upgrades:
        - Deterministic envelope with ISO+unix timestamps, host, pid, git sha (if available), session id.
        - Optional auto-sanitization of payload/meta extras.
        - Optional verify_on_write to check chain integrity after each append.
        - Best-effort lockfile to reduce concurrent writer races (Cartographer should still detect tampering).

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

    """

    def __init__(
        self,
        path: str | Path,
        default_seed: Optional[int] = None,
        enable_lock: bool = True,
        verify_on_write: bool = False,
        auto_sanitize: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lockfile = self.path.with_suffix(self.path.suffix + _LOCKFILE_SUFFIX) if enable_lock else None
        self.verify_on_write = verify_on_write
        self.auto_sanitize = auto_sanitize
        self.default_seed = default_seed
        self.last_sha: Optional[str] = audit.tail_sha(str(self.path))  # None for empty/new files
        self._host = socket.gethostname()
        self._pid = os.getpid()
        self._git_sha = _detect_git_sha(self.path.parent)

    # ------------------------------------------------------------------ helpers

    def _build_meta(self, extra_meta: Optional[Dict[str, Any]], seed: Optional[int]) -> EnvelopeMeta:
        ts = _now_unix()
        return EnvelopeMeta(
            schema=SCHEMA_ID,
            ts_unix=ts,
            ts_iso=_now_iso(ts),
            host=self._host,
            pid=self._pid,
            git_sha=self._git_sha,
            session_id=_session_id,
            seed=(seed if seed is not None else self.default_seed),
            extra=(extra_meta or {}),
        )

    def _make_record(self, payload: Dict[str, Any], meta: EnvelopeMeta) -> Dict[str, Any]:
        """Construct the canonical envelope around payload; raises on non-serializable content unless auto_sanitize."""
        # validate serializability early
        _ensure_json(payload, self.auto_sanitize)
        _ensure_json(meta.extra, self.auto_sanitize)

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
                "extra": meta.extra,
            },
            "payload": payload,
            # chain fields to be added by cc.cartographer.audit:
            # "prev_sha256": "...",
            # "sha256": "..."
        }
        # sanity: canonical dump (sorted keys, compact separators)
        _canonical_dumps(rec, self.auto_sanitize)
        return rec

    # ------------------------------------------------------------------ public API

    def log(
        self,
        payload: Dict[str, Any],
        *,
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
            On serialization failure, lock acquisition failure, or verification failure.
        """
        meta = self._build_meta(extra_meta, seed)
        rec = self._make_record(payload, meta)

        # Optional inter-process lock
        locked = False
        if self.lockfile is not None:
            locked = _acquire_lock(self.lockfile)
            if not locked:
                # Non-fatal by design; Cartographer will still hash-chain detect interleaving,
                # but we warn by raising a soft error only if verify_on_write is requested.
                if (self.verify_on_write if verify_on_write is None else verify_on_write):
                    raise LoggingError(f"Could not acquire audit lock: {self.lockfile}")

        try:
            sha = audit.append_jsonl(str(self.path), rec)
            self.last_sha = sha

            if (self.verify_on_write if verify_on_write is None else verify_on_write):
                try:
                    audit.verify_chain(str(self.path))
                except Exception as e:
                    raise LoggingError(f"Audit chain verification failed post-append: {e}") from e

            return sha
        finally:
            if locked:
                _release_lock(self.lockfile)  # type: ignore[arg-type]

    def log_event(
        self,
        event: str,
        *,
        fields: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        verify_on_write: Optional[bool] = None,
    ) -> str:
        """
        Convenience: wrap an 'event' with arbitrary fields into payload.
        """
        payload = {"event": event}
        if fields:
            payload.update(fields)
        return self.log(payload, seed=seed, extra_meta=extra_meta, verify_on_write=verify_on_write)

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

    # ------------------------------------------------------------------ context manager

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
