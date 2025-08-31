# src/cc/core/logging.py
"""
Module: logging
Purpose: JSONL audit logger with SHA-256 hash chaining
Dependencies: json, hashlib, time, os
Author: Pranav Bhave
Date: 2025-08-27
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


class ChainedJSONLLogger:
    """Audit logger with cryptographic hash chaining for integrity"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hash = "0" * 64  # Genesis hash
        self._load_chain_state()

    def _load_chain_state(self):
        """Load the last hash from existing log file"""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    lines = f.read().strip().split("\n")
                    if lines and lines[-1]:
                        last_record = json.loads(lines[-1])
                        self.last_hash = last_record["record_hash"]
            except (json.JSONDecodeError, KeyError, IndexError):
                # If log is corrupted, start fresh
                self.last_hash = "0" * 64

    def _hash_record(self, rec: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of record for chaining"""
        # Create canonical representation
        canonical = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256()
        h.update(canonical.encode("utf-8"))
        return h.hexdigest()

    def log(self, payload: Dict[str, Any]) -> str:
        """Log a record with hash chaining"""
        # Create record with metadata
        record = dict(payload)
        record.update(
            {
                "logged_at": time.time(),
                "prev_hash": self.last_hash,
                "sequence": self._get_next_sequence(),
            }
        )

        # Compute hash and add to record
        record_hash = self._hash_record(record)
        record["record_hash"] = record_hash

        # Write atomically
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self.last_hash = record_hash
        return record_hash

    def _get_next_sequence(self) -> int:
        """Get next sequence number"""
        if not self.path.exists():
            return 1

        with open(self.path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return 1

        try:
            last_record = json.loads(lines[-1])
            return last_record.get("sequence", 0) + 1
        except:
            return len(lines) + 1

    def verify_chain_integrity(self) -> tuple[bool, Optional[str]]:
        """Verify integrity of the hash chain"""
        if not self.path.exists():
            return True, None

        with open(self.path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return True, None

        expected_prev = "0" * 64  # Genesis

        for i, line in enumerate(lines):
            try:
                record = json.loads(line)

                # Check previous hash linkage
                if record["prev_hash"] != expected_prev:
                    return (
                        False,
                        f"Hash chain broken at record {i+1}: expected prev_hash {expected_prev}, got {record['prev_hash']}",
                    )

                # Verify record hash
                stored_hash = record.pop("record_hash")
                computed_hash = self._hash_record(record)
                record["record_hash"] = stored_hash  # Restore

                if stored_hash != computed_hash:
                    return (
                        False,
                        f"Hash mismatch at record {i+1}: stored {stored_hash}, computed {computed_hash}",
                    )

                expected_prev = stored_hash

            except json.JSONDecodeError:
                return False, f"Invalid JSON at line {i+1}"
            except KeyError as e:
                return False, f"Missing required field {e} at record {i+1}"

        return True, None


@contextmanager
def audit_context(logger: ChainedJSONLLogger, operation: str, **metadata):
    """Context manager for audited operations"""
    start_time = time.time()
    op_id = hashlib.sha256(f"{operation}_{start_time}".encode()).hexdigest()[:16]

    logger.log(
        {"event": "operation_start", "operation": operation, "operation_id": op_id, **metadata}
    )

    try:
        yield op_id
        logger.log(
            {
                "event": "operation_complete",
                "operation": operation,
                "operation_id": op_id,
                "duration": time.time() - start_time,
            }
        )
    except Exception as e:
        logger.log(
            {
                "event": "operation_error",
                "operation": operation,
                "operation_id": op_id,
                "error": str(e),
                "duration": time.time() - start_time,
            }
        )
        raise
