import json

import numpy as np
import pytest

from cc.cartographer import audit

# ---------------------------
# JSONL audit chain tests
# ---------------------------


def test_append_strips_reserved_keys_and_links_correctly(tmp_path):
    p = tmp_path / "audit.jsonl"

    # User record *wrongly* includes reserved keys; these must be ignored.
    rec = {
        "payload": {"x": 1},
        "sha256": "malicious",
        "prev_sha256": "evil",
        "record_hash": "legacy",
        "prev_hash": "legacy_prev",
    }
    h1 = audit.append_jsonl(str(p), rec)

    # File must contain the recomputed sha256, not "malicious", and prev must be None.
    line = p.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["sha256"] == h1 and obj["sha256"] != "malicious"
    assert obj["prev_sha256"] is None
    # Legacy fields must not persist
    assert "record_hash" not in obj and "prev_hash" not in obj

    # Chain verifies
    audit.verify_chain(str(p))


def test_chain_two_records_and_verify(tmp_path):
    p = tmp_path / "audit.jsonl"
    h1 = audit.append_jsonl(str(p), {"event": "first"})
    h2 = audit.append_jsonl(str(p), {"event": "second"})
    assert h2 and h2 != h1
    audit.verify_chain(str(p))


def test_verify_chain_detects_tamper(tmp_path):
    p = tmp_path / "audit.jsonl"
    audit.append_jsonl(str(p), {"payload": {"x": 1}})
    audit.append_jsonl(str(p), {"payload": {"y": 2}})
    # Tamper with line 1 payload but keep the original sha256 -> should fail verify
    lines = p.read_text(encoding="utf-8").splitlines()
    obj1 = json.loads(lines[0])
    obj1["payload"]["x"] = 999  # content change, stale sha remains
    lines[0] = json.dumps(obj1, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(audit.AuditError) as ei:
        audit.verify_chain(str(p))
    assert "SHA mismatch" in str(ei.value)


def test_tail_sha_ignores_trailing_junk_and_truncate(tmp_path):
    p = tmp_path / "audit.jsonl"
    h1 = audit.append_jsonl(str(p), {"a": 1})

    # Simulate torn write / trailing junk
    with open(p, "a", encoding="utf-8") as f:
        f.write("{ this is not json")

    # tail should still point to the last *valid* record
    assert audit.tail_sha(str(p)) == h1

    # Truncate to last valid and verify
    count = audit.truncate_to_last_valid(str(p))
    assert count == 1
    audit.verify_chain(str(p))


def test_rehash_file_requires_valid_chain_and_preserves_valid(tmp_path):
    p = tmp_path / "audit.jsonl"
    # First, make an invalid file
    p.write_text('{"not":"valid without sha"}\n', encoding="utf-8")
    with pytest.raises(audit.AuditError):
        audit.rehash_file(str(p))

    # Now create a valid chain
    p.write_text("", encoding="utf-8")
    audit.append_jsonl(str(p), {"k": 1})
    audit.append_jsonl(str(p), {"k": 2})
    # Rehash should complete and chain must still verify
    audit.rehash_file(str(p))
    audit.verify_chain(str(p))


def test_legacy_record_hash_is_normalized(tmp_path):
    """
    Build a file that uses legacy 'record_hash' (but modern 'prev_sha256') and verify.
    """
    p = tmp_path / "audit_legacy.jsonl"

    # First record: content to be hashed (without any sha fields)
    base1 = {"prev_sha256": None, "payload": {"a": 1}}
    sha1 = audit._compute_record_sha(base1)  # use module's canonical hasher
    rec1 = dict(base1)
    rec1["record_hash"] = sha1  # legacy field for sha

    # Second record
    base2 = {"prev_sha256": sha1, "payload": {"b": 2}}
    sha2 = audit._compute_record_sha(base2)
    rec2 = dict(base2)
    rec2["record_hash"] = sha2

    with open(p, "w", encoding="utf-8") as f:
        f.write(audit._stable_dumps(rec1) + "\n")
        f.write(audit._stable_dumps(rec2) + "\n")

    # Should pass thanks to normalization of 'record_hash' -> 'sha256'
    audit.verify_chain(str(p))


def test_verify_chain_raises_on_missing_sha_field(tmp_path):
    p = tmp_path / "audit_broken.jsonl"
    # Append a valid record
    audit.append_jsonl(str(p), {"ok": True})
    # Append a record missing sha256 manually
    with open(p, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"prev_sha256": "abc", "payload": {"oops": 1}},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    with pytest.raises(audit.AuditError):
        audit.verify_chain(str(p))


# ---------------------------
# FH ceiling auditor tests
# ---------------------------


def _tiny_rocs():
    # Simple monotone ROCs: diagonal + one interior point
    A = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.0, 0.0],
            [0.3, 0.7],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    return A, B


def test_audit_fh_index_flags_violation(tmp_path):
    A, B = _tiny_rocs()
    # Pick a valid grid coordinate and force an impossible J_obs = 2.0 (> any FH cap)
    pairs = [(1, 1, 2.0)]
    bad = audit.audit_fh_ceiling_by_index(A, B, pairs, comp="or", add_anchors=True, tol=1e-12)
    assert bad and bad[0][0] == 1 and bad[0][1] == 1
    # No violation if J_obs is small/negative
    pairs2 = [(1, 1, -1.0)]
    bad2 = audit.audit_fh_ceiling_by_index(A, B, pairs2, comp="AND", add_anchors=False, tol=1e-12)
    assert bad2 == []


def test_audit_fh_points_strict_and_tolerance():
    A, B = _tiny_rocs()
    # Exact points present in arrays
    pa = (0.2, 0.8)
    pb = (0.3, 0.7)

    # Gross violation
    triples = [(pa, pb, 1.5)]
    out = audit.audit_fh_ceiling_by_points(
        A, B, triples, comp="And", add_anchors=False, tol=1e-12, strict=True
    )
    assert len(out) == 1 and out[0][0] == pa and out[0][1] == pb

    # Slightly perturbed point: strict=False should skip silently, strict=True should raise
    pa_eps = (0.2 + 1e-9, 0.8)  # smaller than default point_tol? we'll set a tighter tol below

    # With strict=False and tight point_tol, this should be skipped (no exception, no output)
    out2 = audit.audit_fh_ceiling_by_points(
        A,
        B,
        [(pa_eps, pb, 1.5)],
        comp="OR",
        add_anchors=False,
        tol=1e-12,
        strict=False,
        point_tol=1e-12,
    )
    assert out2 == []

    # With strict=True and very tight point_tol, should raise because point isn't found
    with pytest.raises(audit.AuditError):
        audit.audit_fh_ceiling_by_points(
            A,
            B,
            [(pa_eps, pb, 1.5)],
            comp="OR",
            add_anchors=False,
            tol=1e-12,
            strict=True,
            point_tol=1e-12,
        )
