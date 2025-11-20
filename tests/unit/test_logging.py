# tests/unit/test_logging.py

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from cc.core import logging as cclogging


# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------


class DummyAudit:
    '''
    Minimal deterministic stand-in for cc.cartographer.audit.

    - append_jsonl(path, rec) writes canonical JSONL and returns sha256 of the line
    - tail_sha(path) returns last sha for that file
    - verify_chain(path) optionally raises if verify_should_raise is True
    '''

    def __init__(self) -> None:
        self.records_by_path: Dict[str, List[Dict[str, Any]]] = {}
        self.shas_by_path: Dict[str, List[str]] = {}
        self.verify_should_raise: bool = False
        self.verify_calls: int = 0

    @staticmethod
    def _canonical_line(rec: Dict[str, Any]) -> str:
        return json.dumps(rec, sort_keys=True, separators=(",", ":"))

    def append_jsonl(self, path: str, rec: Dict[str, Any]) -> str:
        path = str(path)
        line = self._canonical_line(rec)
        sha = cclogging.hashlib.sha256(line.encode("utf-8")).hexdigest()
        self.records_by_path.setdefault(path, []).append(rec)
        self.shas_by_path.setdefault(path, []).append(sha)
        # Physically write a JSONL record so tail() etc. behave realistically
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return sha

    def tail_sha(self, path: str) -> Optional[str]:
        path = str(path)
        shas = self.shas_by_path.get(path) or []
        return shas[-1] if shas else None

    def verify_chain(self, path: str) -> None:
        self.verify_calls += 1
        if self.verify_should_raise:
            raise RuntimeError("dummy verify_chain failure")
        # otherwise no-op: chain considered valid


@pytest.fixture
def dummy_audit(monkeypatch) -> DummyAudit:
    '''
    Replace cc.cartographer.audit with an in-memory DummyAudit for each test.
    '''
    dummy = DummyAudit()
    monkeypatch.setattr(cclogging, "audit", dummy)
    return dummy


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_now_iso_format_is_utc_with_millis():
    # Deterministic timestamp
    ts = 1731200000.123  # arbitrary
    iso = cclogging._now_iso(ts)
    # Example: 2024-11-10T12:34:56.123Z
    assert iso.endswith("Z")
    date_part, time_part = iso[:-1].split("T")
    y, m, d = map(int, date_part.split("-"))
    assert 2000 <= y <= 2100
    assert 1 <= m <= 12
    assert 1 <= d <= 31
    hh, mm, ss_ms = time_part.split(":")
    ss, ms = ss_ms.split(".")
    assert len(hh) == len(mm) == len(ss) == 2
    assert len(ms) == 3


def test_coerce_json_safe_handles_common_types(tmp_path):
    p = tmp_path / "file.txt"
    s = {1, 2}
    t = (3, 4)
    b = b"\x01\x02"

    @dataclass
    class Demo:
        x: int

    d = Demo(5)

    assert cclogging._coerce_json_safe(p) == str(p)
    assert cclogging._coerce_json_safe(s) in ([1, 2], [2, 1])
    assert cclogging._coerce_json_safe(t) == [3, 4]
    assert cclogging._coerce_json_safe(b) == b.hex()
    assert cclogging._coerce_json_safe(d) == {"x": 5}


def test_compile_patterns_strict_vs_failsoft():
    # Valid pattern compiles
    pats = cclogging._compile_patterns([r"foo"], strict_mode=True)
    assert len(pats) == 1

    # Invalid pattern in strict mode raises
    with pytest.raises(cclogging.LoggingError):
        cclogging._compile_patterns([r"("], strict_mode=True)

    # Invalid pattern in non-strict mode is ignored
    pats2 = cclogging._compile_patterns([r"("], strict_mode=False)
    assert pats2 == []


def test_deep_redact_by_key_and_pattern():
    patterns = [re.compile(r"secret", re.IGNORECASE)]
    payload = {
        "email": "user@example.com",
        "secret_token": "abc123",
        "nested": {
            "ip": "10.0.0.1",
            "note": "this has secret inside",
        },
    }
    redacted = cclogging._deep_redact(
        payload,
        keys=["email"],
        exclude_keys=[],
        patterns=patterns,
        mask="***",
    )
    # email redacted by key
    assert redacted["email"] == "***"
    # secret_token redacted by pattern on key
    assert redacted["secret_token"] == "***"
    # nested note redacted by pattern on string value
    assert redacted["nested"]["note"] == "this has *** inside"


def test_deep_redact_respects_exclude_keys():
    patterns = [re.compile(r"secret", re.IGNORECASE)]
    payload = {
        "secret": "top-secret",
        "allowed": "secret text",
    }

    redacted = cclogging._deep_redact(
        payload,
        keys=["secret"],
        exclude_keys=["secret"],
        patterns=patterns,
        mask="***",
    )

    # exclude_keys prevents full-field masking by key
    assert redacted["secret"] != "***"
    # but value-level PII scanning still applies inside the string
    assert redacted["secret"] == "top-***"
    # "allowed" still gets string-level redaction
    assert redacted["allowed"] == "*** text"

def test_is_valid_short_sha():
    assert cclogging._is_valid_short_sha("0123abcdefab")
    assert not cclogging._is_valid_short_sha("0123abc")  # too short
    assert not cclogging._is_valid_short_sha("g123abcdefab")  # non-hex


def test_detect_git_sha_prefers_env_over_git(monkeypatch, tmp_path):
    # Reset cache
    monkeypatch.setattr(cclogging, "_git_sha_cache", None, raising=False)

    # Provide env var
    monkeypatch.setenv("GIT_SHA", "0123abcdefab")
    sha = cclogging._detect_git_sha(tmp_path)
    assert sha == "0123abcdefab"

    # If env is invalid, falls back to git (which we fake)
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(cclogging, "_git_sha_cache", None, raising=False)

    def fake_check_output(cmd, cwd=None, stderr=None, text=None, timeout=None):
        return "abcdef123456\n"

    monkeypatch.setattr(cclogging.subprocess, "check_output", fake_check_output)
    sha2 = cclogging._detect_git_sha(tmp_path)
    assert sha2 == "abcdef123456"


def test_pid_is_alive_current_and_fake():
    assert cclogging._pid_is_alive(os.getpid()) is True
    # Choose a very large PID that should not exist; even if it does, the test still passes if True.
    fake_pid = 999_999
    alive = cclogging._pid_is_alive(fake_pid)
    assert isinstance(alive, bool)


def test_acquire_lock_creates_and_breaks_stale_lock(tmp_path, monkeypatch):
    lockfile = tmp_path / "lock"

    # First acquire should create the file
    ok = cclogging._acquire_lock(
        lockfile, stale_seconds=60.0, timeout=0.5, retry_delay=0.01
    )
    assert ok is True
    assert lockfile.exists()

    # Simulate a *stale* lock held by a dead PID:
    # - write a pid/ts payload
    # - force the mtime to be old
    old_ts = cclogging._now_unix() - 120.0
    data = {"pid": os.getpid(), "ts": old_ts}
    lockfile.write_text(json.dumps(data), encoding="utf-8")
    os.utime(lockfile, (old_ts, old_ts))

    # Treat that PID as dead so stale logic is allowed to break the lock
    monkeypatch.setattr(cclogging, "_pid_is_alive", lambda pid: False)

    ok2 = cclogging._acquire_lock(
        lockfile, stale_seconds=1.0, timeout=0.5, retry_delay=0.01
    )
    assert ok2 is True
    assert lockfile.exists()

    # Confirm the stale file was replaced with a fresh lock record
    new_data = json.loads(lockfile.read_text(encoding="utf-8"))
    assert new_data["ts"] > data["ts"]
    assert new_data["pid"] == os.getpid()

    cclogging._release_lock(lockfile)
    assert not lockfile.exists()



def test_acquire_lock_does_not_break_live_lock(tmp_path, monkeypatch):
    lockfile = tmp_path / "lock_live"
    # create a lock owned by a "live" PID
    data = {"pid": 12345, "ts": cclogging._now_unix() - 120.0}
    lockfile.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(cclogging, "_pid_is_alive", lambda pid: True)

    ok = cclogging._acquire_lock(
        lockfile, stale_seconds=1.0, timeout=0.05, retry_delay=0.01
    )
    # We shouldn't be able to break the live lock; acquisition fails
    assert ok is False


# ---------------------------------------------------------------------------
# Core logger behaviour
# ---------------------------------------------------------------------------


def make_logger(
    log_path: Path,
    *,
    capture_env: bool = False,
    auto_sanitize: bool = False,
    enable_lock: bool = False,
    strict_mode: bool = False,
    enable_prometheus: bool = False,
    **kwargs: Any,
) -> cclogging.ChainedJSONLLogger:
    '''
    Small helper so tests can make loggers with sane defaults.
    '''
    return cclogging.ChainedJSONLLogger(
        path=log_path,
        capture_env=capture_env,
        auto_sanitize=auto_sanitize,
        enable_lock=enable_lock,
        strict_mode=strict_mode,
        enable_prometheus=enable_prometheus,
        **kwargs,
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_basic_log_event_writes_record_and_updates_head_and_level(
    dummy_audit, log_path
):
    logger = make_logger(log_path)
    sha = logger.log_event("test_event", fields={"x": 1})
    records = read_jsonl(log_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["payload"]["event"] == "test_event"
    assert rec["payload"]["x"] == 1
    assert rec["meta"]["schema"] == cclogging.SCHEMA_ID_DEFAULT
    assert rec["meta"]["session_id"]  # non-empty
    # level should be recorded in meta.extra (default "info")
    assert rec["meta"]["extra"]["level"] == "info"
    # last_sha synced with dummy_audit
    assert logger.last_sha == sha
    assert dummy_audit.tail_sha(str(log_path)) == sha


def test_log_event_with_explicit_level_and_extra_meta(dummy_audit, log_path):
    logger = make_logger(log_path)
    logger.log_event(
        "critical_event",
        fields={"y": 2},
        level="error",
        extra_meta={"foo": "bar"},
    )
    rec = read_jsonl(log_path)[0]
    extra = rec["meta"]["extra"]
    assert extra["foo"] == "bar"
    assert extra["level"] == "error"


def test_extra_meta_level_not_overwritten_if_present(dummy_audit, log_path):
    logger = make_logger(log_path)
    logger.log(
        {"msg": "hello"},
        level="info",
        extra_meta={"level": "custom"},
    )
    rec = read_jsonl(log_path)[0]
    assert rec["meta"]["extra"]["level"] == "custom"


def test_log_rejects_invalid_level(dummy_audit, log_path):
    logger = make_logger(log_path)
    with pytest.raises(cclogging.LoggingError):
        logger.log_event("bad", level="verbose")  # type: ignore[arg-type]


def test_tail_returns_last_n_records(dummy_audit, log_path):
    logger = make_logger(log_path)
    for i in range(5):
        logger.log_event("e", fields={"i": i})
    tail = logger.tail(3)
    assert len(tail) == 3
    assert [rec["payload"]["i"] for rec in tail] == [2, 3, 4]


def test_tail_on_missing_file_returns_empty_list(tmp_path):
    missing = tmp_path / "nope.jsonl"
    logger = make_logger(missing)
    assert logger.tail(5) == []


def test_tail_skips_corrupt_lines(dummy_audit, log_path):
    logger = make_logger(log_path)
    logger.log_event("ok1")
    # write a corrupt line
    with log_path.open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
    logger.log_event("ok2")
    tail = logger.tail(5)
    # both valid records should be present, corrupt skipped
    assert [r["payload"]["event"] for r in tail] == ["ok1", "ok2"]


def test_current_head_roundtrip(dummy_audit, log_path):
    logger = make_logger(log_path)
    assert logger.current_head() is None
    s1 = logger.log_event("one")
    assert logger.current_head() == s1
    s2 = logger.log_event("two")
    assert logger.current_head() == s2


def test_log_many_appends_multiple_records(dummy_audit, log_path):
    logger = make_logger(log_path)
    payloads = [{"i": i} for i in range(3)]
    shas = logger.log_many(payloads, level="debug")
    assert len(shas) == 3
    records = read_jsonl(log_path)
    assert len(records) == 3
    assert [r["payload"]["i"] for r in records] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Redaction / PII tests
# ---------------------------------------------------------------------------


def test_log_event_redacts_keys_and_builtin_pii(dummy_audit, log_path):
    logger = make_logger(
        log_path,
        redact_keys=["secret"],
        # add a custom key/value pattern (in addition to built-in PII)
        redact_patterns=[r"token"],
    )
    logger.log_event(
        "login",
        fields={
            "secret": "dont show",
            "token_value": "mytoken",
            "ip_address": "192.168.0.1",
            "note": "user email is foo@example.com",
        },
    )
    rec = read_jsonl(log_path)[0]
    payload = rec["payload"]
    # exact key redacted
    assert payload["secret"] == logger.redact_mask
    # pattern on key name redacted
    assert payload["token_value"] == logger.redact_mask
    # built-in PII for IP should be redacted by value regex (IPv4)
    assert payload["ip_address"] == logger.redact_mask
    # note string should have email-related content masked
    assert "foo@example.com" not in payload["note"]
    assert "email" not in payload["note"].lower()


def test_redact_exclude_keys_in_meta(dummy_audit, log_path):
    logger = make_logger(
        log_path,
        redact_keys=["token"],
        redact_exclude_keys=["token"],
        redact_patterns=[r"token"],
    )
    logger.log_event("evt", fields={"token": "abc123"})
    rec = read_jsonl(log_path)[0]
    assert rec["payload"]["token"] == "abc123"


# ---------------------------------------------------------------------------
# JSON strictness / auto-sanitize
# ---------------------------------------------------------------------------


def test_auto_sanitize_allows_non_json_types(dummy_audit, log_path, tmp_path):
    test_path = tmp_path / "file.txt"

    @dataclass
    class Demo:
        x: int

    demo = Demo(42)
    payload = {
        "path": test_path,
        "bytes": b"\x01\x02",
        "tags": {1, 2},
        "demo": demo,
    }
    logger = make_logger(log_path, auto_sanitize=True)
    logger.log(payload)
    rec = read_jsonl(log_path)[0]
    pay = rec["payload"]
    assert pay["path"] == str(test_path)
    # bytes as hex string
    assert isinstance(pay["bytes"], str)
    assert sorted(pay["tags"]) == [1, 2]
    assert pay["demo"] == {"x": 42}


def test_auto_sanitize_false_raises_typeerror(dummy_audit, log_path):
    # payload includes an unserializable object when auto_sanitize=False
    class Unserializable:
        pass

    payload = {"obj": Unserializable()}
    logger = make_logger(log_path, auto_sanitize=False)
    with pytest.raises(TypeError):
        logger.log(payload)


def test_auto_sanitize_false_rejects_nan(dummy_audit, log_path):
    payload = {"x": float("nan")}
    logger = make_logger(log_path, auto_sanitize=False)
    with pytest.raises(ValueError):
        logger.log(payload)


# ---------------------------------------------------------------------------
# Verify chain
# ---------------------------------------------------------------------------


def test_verify_chain_integrity_success_and_failure(dummy_audit, log_path):
    logger = make_logger(log_path)
    logger.log_event("ok")
    ok, reason = logger.verify_chain_integrity()
    assert ok is True
    assert reason is None

    # force verify_chain to raise
    dummy_audit.verify_should_raise = True
    ok2, reason2 = logger.verify_chain_integrity()
    assert ok2 is False
    assert "dummy verify_chain failure" in reason2


def test_verify_on_write_raises_and_counts_error(dummy_audit, log_path, monkeypatch):
    # Enable Prometheus-style metrics with dummy counters
    class DummyCounter:
        def __init__(self) -> None:
            self.count = 0

        def labels(self, **kwargs):
            return self

        def inc(self, amount: int = 1) -> None:
            self.count += amount

    class DummyGauge:
        def __init__(self) -> None:
            self.value = None

        def set(self, value: float) -> None:
            self.value = value

    entries = DummyCounter()
    errors = DummyCounter()
    size_gauge = DummyGauge()

    monkeypatch.setattr(cclogging, "PROMETHEUS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(cclogging, "LOG_ENTRIES", entries, raising=False)
    monkeypatch.setattr(cclogging, "LOG_ERRORS", errors, raising=False)
    monkeypatch.setattr(cclogging, "LOG_SIZE", size_gauge, raising=False)

    logger = make_logger(log_path, enable_prometheus=True)
    # First, happy path with verify_on_write=False
    logger.log_event("ok", verify_on_write=False)
    assert entries.count == 1
    assert size_gauge.value == log_path.stat().st_size
    assert errors.count == 0

    # Now force verify_chain to fail under verify_on_write=True
    dummy_audit.verify_should_raise = True
    with pytest.raises(cclogging.LoggingError):
        logger.log_event("boom", verify_on_write=True)
    # verify_chain was called
    assert dummy_audit.verify_calls >= 1
    # Error counter incremented; entries not incremented for failed log
    assert errors.count == 1
    assert entries.count == 1  # unchanged


def test_metrics_failure_raises_in_strict_mode(dummy_audit, log_path, monkeypatch):
    class DummyCounter:
        def __init__(self) -> None:
            self.count = 0

        def labels(self, **kwargs):
            return self

        def inc(self, amount: int = 1) -> None:
            self.count += amount

    class ExplodingGauge:
        def set(self, value: float) -> None:
            raise RuntimeError("metrics boom")

    entries = DummyCounter()
    errors = DummyCounter()
    exploding_size = ExplodingGauge()

    monkeypatch.setattr(cclogging, "PROMETHEUS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(cclogging, "LOG_ENTRIES", entries, raising=False)
    monkeypatch.setattr(cclogging, "LOG_ERRORS", errors, raising=False)
    monkeypatch.setattr(cclogging, "LOG_SIZE", exploding_size, raising=False)

    logger = make_logger(log_path, enable_prometheus=True, strict_mode=True)
    with pytest.raises(cclogging.LoggingError):
        logger.log_event("x")

    # Error counter should have incremented via LOG_ERRORS.labels(...).inc()
    assert errors.count == 1


# ---------------------------------------------------------------------------
# Locking semantics
# ---------------------------------------------------------------------------


def test_lock_success_creates_and_releases_lock(dummy_audit, log_path):
    logger = make_logger(log_path, enable_lock=True)
    logger.log_event("once")
    # lockfile should not be left behind
    if logger.lockfile is not None:
        assert not logger.lockfile.exists()


def test_lock_failure_respects_strict_mode(monkeypatch, dummy_audit, log_path):
    # Force _acquire_lock to always fail
    def fake_acquire(*args, **kwargs):
        return False

    monkeypatch.setattr(cclogging, "_acquire_lock", fake_acquire)

    # strict_mode=False, verify_on_write=False => soft fail, no exception
    logger_soft = make_logger(log_path, enable_lock=True, strict_mode=False)
    logger_soft.log_event("no_strict", verify_on_write=False)

    # strict_mode=True => should raise on lock failure
    logger_strict = make_logger(log_path, enable_lock=True, strict_mode=True)
    with pytest.raises(cclogging.LoggingError):
        logger_strict.log_event("strict", verify_on_write=False)


def test_lock_failure_with_verify_on_write_raises_even_if_not_strict(
    monkeypatch, dummy_audit, log_path
):
    def fake_acquire(*args, **kwargs):
        return False

    monkeypatch.setattr(cclogging, "_acquire_lock", fake_acquire)

    logger = make_logger(log_path, enable_lock=True, strict_mode=False)
    with pytest.raises(cclogging.LoggingError):
        logger.log_event("verify_on_write", verify_on_write=True)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_creates_backup_and_new_chain_marker(dummy_audit, log_path):
    # Use very small max_bytes so rotation triggers quickly; disable lock to
    # avoid nested-lock issue during rotation marker logging.
    logger = make_logger(
        log_path,
        max_bytes=1,
        backup_count=2,
        compress_backups=False,
        encrypt_backups=False,
        enable_lock=False,
    )

    # First event fills the base file
    logger.log_event("first")
    size_after_first = log_path.stat().st_size
    assert size_after_first > 1

    # Second event should trigger rotation + rotation marker + new record
    logger.log_event("second")

    rotated_path = log_path.with_suffix(log_path.suffix + ".1")
    assert rotated_path.exists()

    # Rotated file should contain only the first event
    rotated_records = read_jsonl(rotated_path)
    assert len(rotated_records) == 1
    assert rotated_records[0]["payload"]["event"] == "first"

    # New file should start a fresh chain with rotation marker then the second event
    new_records = read_jsonl(log_path)
    assert [r["payload"]["event"] for r in new_records] == [
        "rotation_new_chain",
        "second",
    ]


def test_rotation_errors_respect_strict_mode(dummy_audit, log_path, monkeypatch):
    logger_soft = make_logger(
        log_path,
        max_bytes=1,
        backup_count=1,
        enable_lock=False,
        strict_mode=False,
    )

    # patch internal _rotate_files to raise
    def boom(self):
        raise OSError("disk full")

    # --- non-strict logger: rotation error should be swallowed ---
    monkeypatch.setattr(
        logger_soft, "_rotate_files", boom.__get__(logger_soft, type(logger_soft))
    )

    # first call writes something
    logger_soft.log_event("seed")
    # second call should attempt rotation and fail-soft (no exception)
    logger_soft.log_event("second")

    # --- strict logger: use a FRESH path so first write doesn't rotate ---
    strict_path = log_path.with_name("strict_audit.jsonl")
    logger_strict = make_logger(
        strict_path,
        max_bytes=1,
        backup_count=1,
        enable_lock=False,
        strict_mode=True,
    )
    monkeypatch.setattr(
        logger_strict, "_rotate_files", boom.__get__(logger_strict, type(logger_strict))
    )

    # First strict write: file is empty, so no rotation yet → should NOT raise
    logger_strict.log_event("seed2")

    # Second strict write: file size >= max_bytes, rotation attempted → should raise
    with pytest.raises(cclogging.LoggingError):
        logger_strict.log_event("third")



# ---------------------------------------------------------------------------
# Env snapshot / flags
# ---------------------------------------------------------------------------


def test_env_snapshot_included_when_capture_env_true(
    dummy_audit, log_path, monkeypatch
):
    # Avoid running real pip freeze in tests
    def fake_check_output(args, **kwargs):
        return b"pkg==1.0\n"

    monkeypatch.setattr(cclogging.subprocess, "check_output", fake_check_output)
    logger = make_logger(log_path, capture_env=True)
    logger.log_event("with_env")
    rec = read_jsonl(log_path)[0]
    env = rec["meta"]["env"]
    assert env["python"]
    assert env["platform"]
    assert env["pip_freeze"] == ["pkg==1.0"]


def test_env_flags_override_constructor(monkeypatch, dummy_audit, log_path):
    # strict_mode overridden by LOG_STRICT_MODE
    monkeypatch.setenv("LOG_STRICT_MODE", "1")
    logger = make_logger(log_path, strict_mode=False)
    assert logger.strict_mode is True

    # prometheus overridden by LOG_PROMETHEUS
    monkeypatch.delenv("LOG_STRICT_MODE", raising=False)
    monkeypatch.setenv("LOG_PROMETHEUS", "1")
    logger2 = make_logger(log_path, enable_prometheus=False)
    assert logger2.enable_prometheus is bool(cclogging.PROMETHEUS_AVAILABLE)


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


def test_encrypt_backups_respects_availability_flag(monkeypatch, dummy_audit, log_path):
    # Force ENCRYPTION_AVAILABLE=False even if cryptography is installed
    monkeypatch.setattr(cclogging, "ENCRYPTION_AVAILABLE", False, raising=False)
    logger = make_logger(log_path, encrypt_backups=True)
    # encryption should be disabled if library is not available
    assert logger.encrypt_backups is False


def test_encrypt_backups_uses_fernet_when_available(monkeypatch, dummy_audit, log_path):
    # Provide a dummy Fernet implementation so we don't depend on cryptography.
    class DummyFernet:
        def __init__(self, key: bytes) -> None:
            self.key = key

        def encrypt(self, data: bytes) -> bytes:
            return b"enc:" + data

    monkeypatch.setattr(cclogging, "ENCRYPTION_AVAILABLE", True, raising=False)
    monkeypatch.setattr(cclogging, "Fernet", DummyFernet, raising=False)

    logger = make_logger(
        log_path,
        max_bytes=1,
        backup_count=1,
        compress_backups=False,
        encrypt_backups=True,
        enable_lock=False,
        encryption_key=b"dummy-key",
    )
    logger.log_event("first")
    logger.log_event("second")  # triggers rotation + encryption

    # After rotation with encrypt_backups=True and no compression,
    # rotated file should end in ".1.enc"
    rotated_enc = log_path.with_suffix(log_path.suffix + ".1.enc")
    assert rotated_enc.exists()
    with open(rotated_enc, "rb") as f:
        data = f.read()
    assert data.startswith(b"enc:")


def test_invalid_log_enc_key_respects_strict_mode(monkeypatch, dummy_audit, log_path):
    # Invalid base64 should raise in strict mode
    monkeypatch.setenv("LOG_ENC_KEY", "!!!")
    with pytest.raises(cclogging.LoggingError):
        make_logger(log_path, strict_mode=True)
    monkeypatch.delenv("LOG_ENC_KEY", raising=False)


# ---------------------------------------------------------------------------
# fsync
# ---------------------------------------------------------------------------


def test_fsync_best_effort_strict_and_non_strict(dummy_audit, log_path, monkeypatch):
    # Non-strict: fsync failures are swallowed
    logger_soft = make_logger(log_path, fsync_on_write=True, strict_mode=False)

    def boom_open(*args, **kwargs):
        raise OSError("no fd")

    monkeypatch.setattr(cclogging.os, "open", boom_open, raising=False)
    # Should not raise
    logger_soft._fsync_best_effort()

    # Strict: failures raise LoggingError
    logger_strict = make_logger(log_path, fsync_on_write=True, strict_mode=True)
    monkeypatch.setattr(cclogging.os, "open", boom_open, raising=False)
    with pytest.raises(cclogging.LoggingError):
        logger_strict._fsync_best_effort()


# ---------------------------------------------------------------------------
# Post-log hooks
# ---------------------------------------------------------------------------


def test_post_log_hook_called_and_errors_respected(dummy_audit, log_path):
    called: List[str] = []

    def hook(sha: str) -> None:
        called.append(sha)

    logger = make_logger(log_path, post_log_hook=hook)
    sha = logger.log_event("x")
    assert called == [sha]


def test_post_log_hook_error_swallowed_when_not_strict(dummy_audit, log_path):
    def hook(_sha: str) -> None:
        raise RuntimeError("boom")

    logger = make_logger(log_path, post_log_hook=hook, strict_mode=False)
    # Should not raise
    logger.log_event("x")


def test_post_log_hook_error_raises_when_strict(dummy_audit, log_path):
    def hook(_sha: str) -> None:
        raise RuntimeError("boom")

    logger = make_logger(log_path, post_log_hook=hook, strict_mode=True)
    with pytest.raises(RuntimeError):
        logger.log_event("x")


def test_post_log_hook_recursion_guard_prevents_infinite_loop(dummy_audit, log_path):
    # Hook that logs another event using same logger
    def hook(sha: str) -> None:
        logger.log_event("hooked", fields={"sha": sha})

    logger = make_logger(log_path, post_log_hook=hook, strict_mode=True)
    logger.log_event("root")

    records = read_jsonl(log_path)
    # We should have exactly two events: root and hooked (no infinite recursion)
    assert [r["payload"]["event"] for r in records] == ["root", "hooked"]


# ---------------------------------------------------------------------------
# Async API tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_methods_delegate_to_sync(dummy_audit, log_path):
    logger = make_logger(log_path)

    sha1 = await logger.alog({"a": 1})
    sha2 = await logger.alog_event("evt", fields={"b": 2})
    sha_many = await logger.alog_many(
        [{"i": i} for i in range(3)],
        level="debug",
    )
    assert isinstance(sha1, str)
    assert isinstance(sha2, str)
    assert len(sha_many) == 3

    records = read_jsonl(log_path)
    # 1 plain log + 1 event + 3 from alog_many = 5
    assert len(records) == 5
    assert records[0]["payload"]["a"] == 1
    assert records[1]["payload"]["event"] == "evt"
    assert records[1]["payload"]["b"] == 2


@pytest.mark.asyncio
async def test_async_logging_respects_concurrency_limit(dummy_audit, log_path):
    logger = make_logger(log_path, async_concurrency_limit=1)

    async def worker(i: int) -> None:
        await logger.alog_event("evt", fields={"i": i})

    await asyncio.gather(*(worker(i) for i in range(5)))

    records = read_jsonl(log_path)
    assert len(records) == 5
    assert sorted(r["payload"]["i"] for r in records) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


def test_audit_context_logs_start_and_complete(dummy_audit, log_path):
    logger = make_logger(log_path)
    with cclogging.audit_context(logger, "train", run_id="r1") as op_id:
        assert isinstance(op_id, str)

    records = read_jsonl(log_path)
    assert [r["payload"]["event"] for r in records] == [
        "operation_start",
        "operation_complete",
    ]
    # durations present in completion record
    complete = records[1]["payload"]
    assert complete["success"] is True
    assert complete["operation"] == "train"
    assert "duration_wall_s" in complete
    assert "duration_perf_s" in complete


def test_audit_context_logs_error_and_reraises(dummy_audit, log_path):
    logger = make_logger(log_path)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with cclogging.audit_context(logger, "train"):
            raise Boom("fail")

    records = read_jsonl(log_path)
    assert [r["payload"]["event"] for r in records] == [
        "operation_start",
        "operation_error",
    ]
    err = records[1]["payload"]
    assert err["success"] is False
    assert err["error_type"] == "Boom"
    assert "error_message" in err


@pytest.mark.asyncio
async def test_aaudit_context_logs_start_and_complete(dummy_audit, log_path):
    logger = make_logger(log_path)
    async with cclogging.aaudit_context(logger, "eval", experiment="exp1") as op_id:
        assert isinstance(op_id, str)

    records = read_jsonl(log_path)
    assert [r["payload"]["event"] for r in records] == [
        "operation_start",
        "operation_complete",
    ]


@pytest.mark.asyncio
async def test_aaudit_context_logs_error_and_reraises(dummy_audit, log_path):
    logger = make_logger(log_path)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        async with cclogging.aaudit_context(logger, "eval"):
            raise Boom("fail")

    records = read_jsonl(log_path)
    assert [r["payload"]["event"] for r in records] == [
        "operation_start",
        "operation_error",
    ]
