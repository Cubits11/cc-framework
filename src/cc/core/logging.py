# src/cc/core/logging.py
"""
Module: logging
Purpose: JSONL audit logger (shim) that delegates to Cartographer's tamper-evident chain
Dependencies: json, time, pathlib; delegates hashing/verification to cc.cartographer.audit
Author: Pranav Bhave
Date: 2025-08-27 (refined 2025-09-03)

Notes
-----
- This module is a thin **compatibility shim** over `cc.cartographer.audit`.
- It preserves the old `ChainedJSONLLogger` API while writing records in the
  new, tamper-evident format with fields:
      prev_sha256, sha256
  (instead of the legacy `prev_hash`, `record_hash`).
- Use `cc.cartographer.audit` directly for new code.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cc.cartographer import audit

__all__ = ["ChainedJSONLLogger", "audit_context"]


class ChainedJSONLLogger:
    """
    Compatibility wrapper over `cc.cartographer.audit`.

    Old API:
        logger = ChainedJSONLLogger("runs/audit.jsonl")
        sha = logger.log({"event": "something", ...})
        ok, err = logger.verify_chain_integrity()

    Implementation detail:
        - `log()` builds a minimal record with a schema + timestamp and nests
          the user payload under `"payload"`. It then appends via `audit.append_jsonl`.
        - `verify_chain_integrity()` calls `audit.verify_chain` and converts exceptions
          into the boolean/str pair expected by legacy callers.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Track the last chain head (None for empty/new files)
        self.last_sha: Optional[str] = audit.tail_sha(str(self.path))

    def _make_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build a minimal, stable record envelope around the payload."""
        # Ensure payload is JSON-serializable (legacy behavior matched)
        _ = json.dumps(payload)  # will raise if not serializable

        return {
            "meta": {"schema": "core/logging.v1", "ts": _now_unix()},
            "payload": payload,
            # Chain fields will be filled by `audit.append_jsonl`:
            # "prev_sha256": ...,
            # "sha256": ...,
        }

    def log(self, payload: Dict[str, Any]) -> str:
        """
        Append a record to the audit chain.

        Returns:
            The hex SHA-256 digest of the appended record (chain head).
        """
        rec = self._make_record(payload)
        sha = audit.append_jsonl(str(self.path), rec)
        self.last_sha = sha
        return sha

    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the entire chain for tampering.

        Returns:
            (True, None) on success, (False, reason) on failure.
        """
        try:
            audit.verify_chain(str(self.path))
            return True, None
        except Exception as e:
            return False, str(e)


@contextmanager
def audit_context(logger: ChainedJSONLLogger, operation: str, **metadata):
    """
    Context manager for audited operations using the shim logger.

    Example:
        with audit_context(logger, "train", run_id=rid) as op_id:
            ... do work ...
    """
    start = _now_unix()
    op_id = _op_id(operation, start)

    logger.log(
        {
            "event": "operation_start",
            "operation": operation,
            "operation_id": op_id,
            "meta": metadata,
        }
    )

    try:
        yield op_id
        logger.log(
            {
                "event": "operation_complete",
                "operation": operation,
                "operation_id": op_id,
                "duration_s": _now_unix() - start,
            }
        )
    except Exception as e:
        logger.log(
            {
                "event": "operation_error",
                "operation": operation,
                "operation_id": op_id,
                "error": str(e),
                "duration_s": _now_unix() - start,
            }
        )
        raise


# ---------------------------------------------------------------------------

def _now_unix() -> float:
    """Seconds since epoch as float (UTC)."""
    return float(time.time())


def _op_id(operation: str, start_ts: float) -> str:
    """Deterministic short operation id (not cryptographic)."""
    import hashlib

    h = hashlib.sha256(f"{operation}:{start_ts:.6f}".encode("utf-8")).hexdigest()
    return h[:16]