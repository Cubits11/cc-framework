# tests/integration/test_e2e_ledger_attestation.py
"""End-to-end ledger attestation hash chain."""
from __future__ import annotations

import json
from pathlib import Path

from cc.adapters.base import build_audit_payload, canonical_json, fingerprint_payload, hash_text


def _append_ledger(path: Path, payload: dict, prev_hash: str, seq: int) -> str:
    record = {
        "seq": seq,
        "prev_hash": prev_hash,
        "payload": payload,
    }
    record_hash = fingerprint_payload({"prev_hash": prev_hash, "payload": payload}, strict=False)
    record["record_hash"] = record_hash
    path.write_text(
        (path.read_text(encoding="utf-8") if path.exists() else "")
        + json.dumps(record, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return record_hash


def test_ledger_attestation_chain(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    prompt = "deterministic prompt"
    response = "deterministic response"
    payload = build_audit_payload(
        prompt=prompt,
        response=response,
        adapter_name="mock",
        adapter_version="0.1",
        parameters={"threshold": 0.5},
        decision="allow",
        category=None,
        rationale="ok",
        started_at=100.0,
        completed_at=101.0,
        metadata={"user": "abc"},
        vendor_request_id="req-1",
        config_fingerprint=hash_text("cfg"),
    )
    payload2 = build_audit_payload(
        prompt=prompt,
        response=response,
        adapter_name="mock",
        adapter_version="0.1",
        parameters={"threshold": 0.5},
        decision="allow",
        category=None,
        rationale="ok",
        started_at=100.0,
        completed_at=101.0,
        metadata={"user": "abc"},
        vendor_request_id="req-2",
        config_fingerprint=hash_text("cfg"),
    )

    prev_hash = "0" * 64
    h1 = _append_ledger(ledger_path, payload, prev_hash, 1)
    h2 = _append_ledger(ledger_path, payload2, h1, 2)

    content = ledger_path.read_text(encoding="utf-8")
    assert prompt not in content
    assert response not in content

    # Recompute chain deterministically.
    lines = [json.loads(line) for line in content.strip().splitlines()]
    recomputed = "0" * 64
    for line in lines:
        recomputed = fingerprint_payload(
            {"prev_hash": recomputed, "payload": line["payload"]}, strict=False
        )
        assert line["record_hash"] == recomputed
    assert recomputed == h2

    # Canonical serialization is stable.
    assert canonical_json(payload, strict=False) == canonical_json(payload, strict=False)
