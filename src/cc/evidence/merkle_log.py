"""Append-only Merkle transparency log for audit evidence.

The tree uses the RFC 6962 Certificate Transparency hash shape:

* leaf hash: ``SHA256(0x00 || canonical_record_json)``
* internal node hash: ``SHA256(0x01 || left_child || right_child)``
* empty tree hash: ``SHA256(b"")``

The verifier functions in this module do not read local log files. A third
party can verify an inclusion proof or a consistency proof using only the
record, proof object, and trusted root hashes/checkpoints supplied out of band.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMPTY_ROOT_HASH = hashlib.sha256(b"").hexdigest()
LOG_RECORD_SCHEMA = "cc/merkle-log-record.v1"
INCLUSION_PROOF_SCHEMA = "cc/merkle-inclusion-proof.v1"
CONSISTENCY_PROOF_SCHEMA = "cc/merkle-consistency-proof.v1"
HASH_ALGORITHM = "sha256-rfc6962"


class MerkleLogError(ValueError):
    """Raised when a Merkle log or proof is structurally invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _hash_leaf_bytes(record_bytes: bytes) -> bytes:
    return _sha256(b"\x00" + record_bytes)


def _hash_node_bytes(left: bytes, right: bytes) -> bytes:
    return _sha256(b"\x01" + left + right)


def _hash_node_hex(left: str, right: str) -> str:
    return _hash_node_bytes(bytes.fromhex(left), bytes.fromhex(right)).hex()


def _is_hash_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def leaf_hash(record: Any) -> str:
    """Return the RFC 6962 leaf hash for a JSON-serializable audit record."""

    return _hash_leaf_bytes(_canonical_json_bytes(record)).hex()


def _largest_power_of_two_less_than(n: int) -> int:
    if n <= 1:
        raise ValueError("n must be greater than one")
    return 1 << ((n - 1).bit_length() - 1)


def _root_from_leaf_hashes(leaf_hashes: Sequence[str]) -> str:
    size = len(leaf_hashes)
    if size == 0:
        return EMPTY_ROOT_HASH
    if size == 1:
        h = leaf_hashes[0]
        if not _is_hash_hex(h):
            raise MerkleLogError("invalid leaf hash")
        return h

    split = _largest_power_of_two_less_than(size)
    left = _root_from_leaf_hashes(leaf_hashes[:split])
    right = _root_from_leaf_hashes(leaf_hashes[split:])
    return _hash_node_hex(left, right)


def root_from_records(records: Iterable[Any]) -> str:
    """Return the Merkle root for a sequence of audit records."""

    return _root_from_leaf_hashes([leaf_hash(record) for record in records])


@dataclass(frozen=True)
class InclusionProof:
    """Merkle inclusion proof for one record at a zero-based record id."""

    record_id: int
    tree_size: int
    root_hash: str
    leaf_hash: str
    proof: list[dict[str, str]]
    hash_algorithm: str = HASH_ALGORITHM
    schema: str = INCLUSION_PROOF_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hash_algorithm": self.hash_algorithm,
            "record_id": self.record_id,
            "tree_size": self.tree_size,
            "root_hash": self.root_hash,
            "leaf_hash": self.leaf_hash,
            "proof": [dict(step) for step in self.proof],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InclusionProof:
        proof = payload.get("proof")
        if not isinstance(proof, list):
            raise MerkleLogError("inclusion proof must contain a proof list")
        steps: list[dict[str, str]] = []
        for step in proof:
            if not isinstance(step, Mapping):
                raise MerkleLogError("inclusion proof step must be an object")
            side = step.get("side", step.get("position"))
            sibling_hash = step.get("hash")
            if side not in {"left", "right"} or not isinstance(sibling_hash, str):
                raise MerkleLogError("invalid inclusion proof step")
            steps.append({"side": str(side), "hash": sibling_hash})
        return cls(
            record_id=int(payload["record_id"]),
            tree_size=int(payload["tree_size"]),
            root_hash=str(payload["root_hash"]),
            leaf_hash=str(payload["leaf_hash"]),
            proof=steps,
            hash_algorithm=str(payload.get("hash_algorithm", HASH_ALGORITHM)),
            schema=str(payload.get("schema", INCLUSION_PROOF_SCHEMA)),
        )


@dataclass(frozen=True)
class ConsistencyProof:
    """RFC 6962 consistency proof from one checkpoint root to a later root."""

    old_size: int
    new_size: int
    old_root: str
    new_root: str
    proof: list[str]
    hash_algorithm: str = HASH_ALGORITHM
    schema: str = CONSISTENCY_PROOF_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hash_algorithm": self.hash_algorithm,
            "old_size": self.old_size,
            "new_size": self.new_size,
            "old_root": self.old_root,
            "new_root": self.new_root,
            "proof": list(self.proof),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConsistencyProof:
        proof = payload.get("proof")
        if not isinstance(proof, list) or not all(isinstance(h, str) for h in proof):
            raise MerkleLogError("consistency proof must contain a list of hashes")
        return cls(
            old_size=int(payload["old_size"]),
            new_size=int(payload["new_size"]),
            old_root=str(payload["old_root"]),
            new_root=str(payload["new_root"]),
            proof=list(proof),
            hash_algorithm=str(payload.get("hash_algorithm", HASH_ALGORITHM)),
            schema=str(payload.get("schema", CONSISTENCY_PROOF_SCHEMA)),
        )


def _coerce_inclusion_proof(proof: InclusionProof | Mapping[str, Any]) -> InclusionProof:
    if isinstance(proof, InclusionProof):
        return proof
    return InclusionProof.from_dict(proof)


def _coerce_consistency_proof(proof: ConsistencyProof | Mapping[str, Any]) -> ConsistencyProof:
    if isinstance(proof, ConsistencyProof):
        return proof
    return ConsistencyProof.from_dict(proof)


def _inclusion_path(leaf_hashes: Sequence[str], record_id: int) -> list[dict[str, str]]:
    def rec(start: int, end: int, idx: int) -> list[dict[str, str]]:
        size = end - start
        if size == 1:
            return []
        split = _largest_power_of_two_less_than(size)
        mid = start + split
        if idx < mid:
            return [
                *rec(start, mid, idx),
                {"side": "right", "hash": _root_from_leaf_hashes(leaf_hashes[mid:end])},
            ]
        return [
            *rec(mid, end, idx),
            {"side": "left", "hash": _root_from_leaf_hashes(leaf_hashes[start:mid])},
        ]

    return rec(0, len(leaf_hashes), record_id)


def _expected_inclusion_sides(record_id: int, tree_size: int) -> list[str]:
    def rec(start: int, end: int, idx: int) -> list[str]:
        size = end - start
        if size == 1:
            return []
        split = _largest_power_of_two_less_than(size)
        mid = start + split
        if idx < mid:
            return [*rec(start, mid, idx), "right"]
        return [*rec(mid, end, idx), "left"]

    return rec(0, tree_size, record_id)


def verify_inclusion(
    record: Any,
    proof: InclusionProof | Mapping[str, Any],
    *,
    root_hash: str | None = None,
) -> bool:
    """Verify that ``record`` is included under a trusted Merkle root.

    The optional ``root_hash`` argument is the out-of-band trusted root. If it
    is omitted, the root embedded in the proof is used.
    """

    try:
        parsed = _coerce_inclusion_proof(proof)
        trusted_root = root_hash or parsed.root_hash
        if parsed.hash_algorithm != HASH_ALGORITHM:
            return False
        if parsed.tree_size <= 0 or not (0 <= parsed.record_id < parsed.tree_size):
            return False
        if not all(_is_hash_hex(h) for h in [parsed.root_hash, parsed.leaf_hash, trusted_root]):
            return False
        if root_hash is not None and parsed.root_hash != root_hash:
            return False
        if leaf_hash(record) != parsed.leaf_hash:
            return False

        expected_sides = _expected_inclusion_sides(parsed.record_id, parsed.tree_size)
        actual_sides = [step.get("side") for step in parsed.proof]
        if actual_sides != expected_sides:
            return False

        current = parsed.leaf_hash
        for step in parsed.proof:
            sibling = step["hash"]
            if not _is_hash_hex(sibling):
                return False
            if step["side"] == "left":
                current = _hash_node_hex(sibling, current)
            elif step["side"] == "right":
                current = _hash_node_hex(current, sibling)
            else:
                return False
        return current == trusted_root
    except Exception:
        return False


def _consistency_subproof(
    old_size: int, leaf_hashes: Sequence[str], include_complete_subtree: bool
) -> list[str]:
    new_size = len(leaf_hashes)
    if old_size == new_size:
        return [] if include_complete_subtree else [_root_from_leaf_hashes(leaf_hashes)]

    split = _largest_power_of_two_less_than(new_size)
    if old_size <= split:
        return [
            *_consistency_subproof(old_size, leaf_hashes[:split], include_complete_subtree),
            _root_from_leaf_hashes(leaf_hashes[split:]),
        ]

    return [
        *_consistency_subproof(old_size - split, leaf_hashes[split:], False),
        _root_from_leaf_hashes(leaf_hashes[:split]),
    ]


def _consistency_proof_hashes(old_size: int, leaf_hashes: Sequence[str]) -> list[str]:
    if old_size == 0:
        return []
    return _consistency_subproof(old_size, leaf_hashes, True)


def verify_consistency(
    proof: ConsistencyProof | Mapping[str, Any],
    *,
    old_root: str | None = None,
    new_root: str | None = None,
) -> bool:
    """Verify that a later Merkle root is an append-only extension.

    The root arguments are the caller's trusted checkpoints. If omitted, the
    roots embedded in the proof are used.
    """

    try:
        parsed = _coerce_consistency_proof(proof)
        trusted_old = old_root or parsed.old_root
        trusted_new = new_root or parsed.new_root
        if parsed.hash_algorithm != HASH_ALGORITHM:
            return False
        if parsed.old_size < 0 or parsed.new_size < parsed.old_size:
            return False
        if not all(_is_hash_hex(h) for h in [parsed.old_root, parsed.new_root, trusted_old, trusted_new]):
            return False
        if old_root is not None and parsed.old_root != old_root:
            return False
        if new_root is not None and parsed.new_root != new_root:
            return False
        if not all(_is_hash_hex(h) for h in parsed.proof):
            return False

        if parsed.old_size == parsed.new_size:
            return parsed.old_root == parsed.new_root == trusted_old == trusted_new and not parsed.proof
        if parsed.old_size == 0:
            return parsed.old_root == trusted_old == EMPTY_ROOT_HASH and not parsed.proof

        fn = parsed.old_size - 1
        sn = parsed.new_size - 1
        hashes = [bytes.fromhex(h) for h in parsed.proof]

        while fn & 1:
            fn >>= 1
            sn >>= 1

        if fn == 0:
            old_hash = bytes.fromhex(parsed.old_root)
            new_hash = bytes.fromhex(parsed.old_root)
        else:
            if not hashes:
                return False
            old_hash = hashes.pop(0)
            new_hash = old_hash

        while fn != 0:
            if fn & 1:
                if not hashes:
                    return False
                sibling = hashes.pop(0)
                old_hash = _hash_node_bytes(sibling, old_hash)
                new_hash = _hash_node_bytes(sibling, new_hash)
            elif fn < sn:
                if not hashes:
                    return False
                sibling = hashes.pop(0)
                new_hash = _hash_node_bytes(new_hash, sibling)
            fn >>= 1
            sn >>= 1

        while sn != 0:
            if not hashes:
                return False
            sibling = hashes.pop(0)
            new_hash = _hash_node_bytes(new_hash, sibling)
            sn >>= 1

        return (
            old_hash.hex() == trusted_old
            and new_hash.hex() == trusted_new
            and not hashes
        )
    except Exception:
        return False


class MerkleLog:
    """Append-only JSONL-backed Merkle log over audit records."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        records: Iterable[Any] | None = None,
        log_id: str = "cc-transparency-log",
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.log_id = log_id
        self._records: list[Any] = []
        self._leaf_hashes: list[str] = []

        if self.path is not None and self.path.exists():
            self._load()
        if records is not None:
            for record in records:
                self.append(record)

    @property
    def tree_size(self) -> int:
        return len(self._leaf_hashes)

    @property
    def records(self) -> list[Any]:
        return list(self._records)

    @property
    def leaf_hashes(self) -> list[str]:
        return list(self._leaf_hashes)

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise MerkleLogError(f"line {line_number}: invalid JSON") from exc
                if not isinstance(entry, Mapping):
                    raise MerkleLogError(f"line {line_number}: log entry must be an object")
                if entry.get("schema") != LOG_RECORD_SCHEMA:
                    raise MerkleLogError(f"line {line_number}: unsupported log record schema")
                expected_id = len(self._records)
                if entry.get("record_id") != expected_id:
                    raise MerkleLogError(
                        f"line {line_number}: record_id {entry.get('record_id')} "
                        f"does not match expected {expected_id}"
                    )
                if "record" not in entry or not isinstance(entry.get("leaf_hash"), str):
                    raise MerkleLogError(f"line {line_number}: missing record or leaf_hash")
                record = entry["record"]
                expected_leaf = leaf_hash(record)
                if entry["leaf_hash"] != expected_leaf:
                    raise MerkleLogError(f"line {line_number}: leaf hash mismatch")
                self._records.append(record)
                self._leaf_hashes.append(expected_leaf)

    def append(self, record: Any) -> int:
        """Append a record and return its zero-based record id."""

        record_id = len(self._records)
        record_leaf_hash = leaf_hash(record)
        self._records.append(record)
        self._leaf_hashes.append(record_leaf_hash)

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "schema": LOG_RECORD_SCHEMA,
                "hash_algorithm": HASH_ALGORITHM,
                "record_id": record_id,
                "record": record,
                "leaf_hash": record_leaf_hash,
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical_json_bytes(entry).decode("utf-8") + "\n")

        return record_id

    def root_hash(self, size: int | None = None) -> str:
        """Return the Merkle root for the current log, or for a prefix size."""

        if size is None:
            size = self.tree_size
        if size < 0 or size > self.tree_size:
            raise MerkleLogError("root size is outside the current log")
        return _root_from_leaf_hashes(self._leaf_hashes[:size])

    def inclusion_proof(self, record_id: int) -> InclusionProof:
        """Return an inclusion proof for a zero-based record id."""

        if not (0 <= record_id < self.tree_size):
            raise MerkleLogError("record_id is outside the current log")
        return InclusionProof(
            record_id=record_id,
            tree_size=self.tree_size,
            root_hash=self.root_hash(),
            leaf_hash=self._leaf_hashes[record_id],
            proof=_inclusion_path(self._leaf_hashes, record_id),
        )

    def consistency_proof(self, old_root: str, new_root: str) -> ConsistencyProof:
        """Return a proof that ``new_root`` extends ``old_root`` as a prefix."""

        prefix_sizes: dict[str, int] = {}
        for size in range(self.tree_size + 1):
            prefix_sizes[self.root_hash(size)] = size

        if old_root not in prefix_sizes:
            raise MerkleLogError("old_root is not a prefix root in this log")
        if new_root not in prefix_sizes:
            raise MerkleLogError("new_root is not a prefix root in this log")

        old_size = prefix_sizes[old_root]
        new_size = prefix_sizes[new_root]
        if new_size < old_size:
            raise MerkleLogError("new_root precedes old_root")

        return ConsistencyProof(
            old_size=old_size,
            new_size=new_size,
            old_root=old_root,
            new_root=new_root,
            proof=_consistency_proof_hashes(old_size, self._leaf_hashes[:new_size]),
        )


__all__ = [
    "CONSISTENCY_PROOF_SCHEMA",
    "EMPTY_ROOT_HASH",
    "HASH_ALGORITHM",
    "INCLUSION_PROOF_SCHEMA",
    "LOG_RECORD_SCHEMA",
    "ConsistencyProof",
    "InclusionProof",
    "MerkleLog",
    "MerkleLogError",
    "leaf_hash",
    "root_from_records",
    "verify_consistency",
    "verify_inclusion",
]
