"""External anchoring for Merkle transparency roots.

This module implements a pluggable witness interface and a concrete Ed25519
witness. In any deployed use, the witness private key should be controlled by
an independent party or service. The local Ed25519 implementation exists so
tests and reference deployments can exercise the protocol without depending on
a specific TSA.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

CHECKPOINT_SCHEMA = "cc/merkle-root-checkpoint.v1"
WITNESS_SIGNATURE_SCHEMA = "cc/witness-signature.v1"
ANCHOR_SCHEMA = "cc/root-anchor.v1"
ANCHOR_MECHANISM = "ed25519-witness"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_hash_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class RootCheckpoint:
    """A witnessable Merkle root checkpoint."""

    log_id: str
    tree_size: int
    root_hash: str
    issued_at: str
    run_nonce: str | None = None
    previous_root_hash: str | None = None
    schema: str = CHECKPOINT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "log_id": self.log_id,
            "tree_size": self.tree_size,
            "root_hash": self.root_hash,
            "issued_at": self.issued_at,
            "run_nonce": self.run_nonce,
            "previous_root_hash": self.previous_root_hash,
        }

    @classmethod
    def create(
        cls,
        *,
        log_id: str,
        tree_size: int,
        root_hash: str,
        run_nonce: str | None = None,
        previous_root_hash: str | None = None,
        issued_at: str | None = None,
    ) -> RootCheckpoint:
        if tree_size < 0:
            raise ValueError("tree_size must be non-negative")
        if not _is_hash_hex(root_hash):
            raise ValueError("root_hash must be a SHA-256 hex digest")
        if previous_root_hash is not None and not _is_hash_hex(previous_root_hash):
            raise ValueError("previous_root_hash must be a SHA-256 hex digest")
        return cls(
            log_id=log_id,
            tree_size=tree_size,
            root_hash=root_hash,
            issued_at=issued_at or _utc_now(),
            run_nonce=run_nonce,
            previous_root_hash=previous_root_hash,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RootCheckpoint:
        return cls(
            log_id=str(payload["log_id"]),
            tree_size=int(payload["tree_size"]),
            root_hash=str(payload["root_hash"]),
            issued_at=str(payload["issued_at"]),
            run_nonce=(None if payload.get("run_nonce") is None else str(payload.get("run_nonce"))),
            previous_root_hash=(
                None
                if payload.get("previous_root_hash") is None
                else str(payload.get("previous_root_hash"))
            ),
            schema=str(payload.get("schema", CHECKPOINT_SCHEMA)),
        )


@dataclass(frozen=True)
class WitnessSignature:
    """A witness signature over a root checkpoint."""

    witness_id: str
    public_key: str
    signature: str
    signed_at: str
    checkpoint_hash: str
    schema: str = WITNESS_SIGNATURE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "witness_id": self.witness_id,
            "public_key": self.public_key,
            "signature": self.signature,
            "signed_at": self.signed_at,
            "checkpoint_hash": self.checkpoint_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WitnessSignature:
        return cls(
            witness_id=str(payload["witness_id"]),
            public_key=str(payload["public_key"]),
            signature=str(payload["signature"]),
            signed_at=str(payload["signed_at"]),
            checkpoint_hash=str(payload["checkpoint_hash"]),
            schema=str(payload.get("schema", WITNESS_SIGNATURE_SCHEMA)),
        )


@dataclass(frozen=True)
class RootAnchor:
    """A checkpoint plus one or more witness co-signatures."""

    checkpoint: RootCheckpoint
    signatures: list[WitnessSignature]
    mechanism: str = ANCHOR_MECHANISM
    schema: str = ANCHOR_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism": self.mechanism,
            "checkpoint": self.checkpoint.to_dict(),
            "signatures": [signature.to_dict() for signature in self.signatures],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RootAnchor:
        signatures = payload.get("signatures")
        if not isinstance(signatures, list):
            raise ValueError("anchor signatures must be a list")
        return cls(
            checkpoint=RootCheckpoint.from_dict(payload["checkpoint"]),
            signatures=[WitnessSignature.from_dict(sig) for sig in signatures],
            mechanism=str(payload.get("mechanism", ANCHOR_MECHANISM)),
            schema=str(payload.get("schema", ANCHOR_SCHEMA)),
        )


class Witness(Protocol):
    """Pluggable third-party root witness interface."""

    witness_id: str

    def witness(self, checkpoint: RootCheckpoint) -> WitnessSignature:
        """Co-sign a checkpoint."""


class Ed25519Witness:
    """Ed25519 witness implementation for checkpoint co-signing."""

    def __init__(self, private_key: Any, *, witness_id: str) -> None:
        self._private_key = private_key
        self.witness_id = witness_id

    @classmethod
    def from_private_key_path(cls, path: Path, *, witness_id: str) -> Ed25519Witness:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519

        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise TypeError("Witness anchoring requires an Ed25519 private key.")
        return cls(key, witness_id=witness_id)

    def public_key_hex(self) -> str:
        from cryptography.hazmat.primitives import serialization

        public_key = self._private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

    def witness(self, checkpoint: RootCheckpoint) -> WitnessSignature:
        payload = checkpoint.to_dict()
        signature = self._private_key.sign(_canonical_json_bytes(payload)).hex()
        return WitnessSignature(
            witness_id=self.witness_id,
            public_key=self.public_key_hex(),
            signature=signature,
            signed_at=_utc_now(),
            checkpoint_hash=_sha256_json(payload),
        )


def anchor_root(checkpoint: RootCheckpoint, witness: Witness) -> RootAnchor:
    """Return a witness anchor for a Merkle root checkpoint."""

    return RootAnchor(checkpoint=checkpoint, signatures=[witness.witness(checkpoint)])


def verify_witness_signature(
    checkpoint: RootCheckpoint,
    signature: WitnessSignature,
    *,
    trusted_public_key: str | None,
    trusted_witness_id: str | None = None,
) -> bool:
    """Verify one witness signature against an expected public key."""

    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        if trusted_witness_id is not None and signature.witness_id != trusted_witness_id:
            return False
        if trusted_public_key is None or signature.public_key != trusted_public_key:
            return False
        if not _is_hash_hex(signature.checkpoint_hash):
            return False
        if signature.checkpoint_hash != _sha256_json(checkpoint.to_dict()):
            return False
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_public_key))
        public_key.verify(
            bytes.fromhex(signature.signature),
            _canonical_json_bytes(checkpoint.to_dict()),
        )
        return True
    except Exception:
        return False


def verify_anchor(
    anchor: RootAnchor | Mapping[str, Any],
    *,
    trusted_witness_public_keys: Mapping[str, str],
    expected_root_hash: str | None = None,
    expected_tree_size: int | None = None,
    expected_log_id: str | None = None,
    expected_run_nonce: str | None = None,
) -> bool:
    """Verify a root anchor against trusted witness public keys."""

    try:
        parsed = anchor if isinstance(anchor, RootAnchor) else RootAnchor.from_dict(anchor)
        if parsed.schema != ANCHOR_SCHEMA or parsed.mechanism != ANCHOR_MECHANISM:
            return False
        checkpoint = parsed.checkpoint
        if checkpoint.schema != CHECKPOINT_SCHEMA:
            return False
        if expected_root_hash is not None and checkpoint.root_hash != expected_root_hash:
            return False
        if expected_tree_size is not None and checkpoint.tree_size != expected_tree_size:
            return False
        if expected_log_id is not None and checkpoint.log_id != expected_log_id:
            return False
        if expected_run_nonce is not None and checkpoint.run_nonce != expected_run_nonce:
            return False
        if not parsed.signatures:
            return False
        for signature in parsed.signatures:
            trusted_key = trusted_witness_public_keys.get(signature.witness_id)
            if verify_witness_signature(
                checkpoint,
                signature,
                trusted_public_key=trusted_key,
                trusted_witness_id=signature.witness_id,
            ):
                return True
        return False
    except Exception:
        return False


__all__ = [
    "ANCHOR_MECHANISM",
    "ANCHOR_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "WITNESS_SIGNATURE_SCHEMA",
    "Ed25519Witness",
    "RootAnchor",
    "RootCheckpoint",
    "Witness",
    "WitnessSignature",
    "anchor_root",
    "verify_anchor",
    "verify_witness_signature",
]
