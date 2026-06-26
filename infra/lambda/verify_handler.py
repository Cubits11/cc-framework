"""Standalone Lambda handler for verify(bundle_id)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any

import boto3

HASH_ALGORITHM = "sha256-rfc6962"
KMS_SIGNING_ALGORITHM = "RSASSA_PKCS1_V1_5_SHA_256"


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


def _leaf_hash(record: Any) -> str:
    return _sha256(b"\x00" + _canonical_json_bytes(record)).hex()


def _hash_node_hex(left: str, right: str) -> str:
    return _sha256(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)).hex()


def _largest_power_of_two_less_than(n: int) -> int:
    return 1 << ((n - 1).bit_length() - 1)


def _expected_sides(record_id: int, tree_size: int) -> list[str]:
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


def _verify_inclusion(record: Any, proof: dict[str, Any], root_hash: str) -> bool:
    if proof.get("hash_algorithm") != HASH_ALGORITHM:
        return False
    if proof.get("root_hash") != root_hash:
        return False
    if _leaf_hash(record) != proof.get("leaf_hash"):
        return False
    if [step.get("side") for step in proof.get("proof", [])] != _expected_sides(
        int(proof["record_id"]),
        int(proof["tree_size"]),
    ):
        return False

    current = str(proof["leaf_hash"])
    for step in proof["proof"]:
        if step["side"] == "left":
            current = _hash_node_hex(str(step["hash"]), current)
        elif step["side"] == "right":
            current = _hash_node_hex(current, str(step["hash"]))
        else:
            return False
    return current == root_hash


def _verify_kms_signature(kms: Any, key_id: str, attestation: dict[str, Any]) -> bool:
    fields = {
        "schema",
        "bundle_id",
        "bundle_sha256",
        "merkle_root",
        "tree_size",
        "sequence_number",
        "signed_at",
    }
    if not fields.issubset(attestation):
        return False
    message = {key: attestation[key] for key in sorted(fields)}
    result = kms.verify(
        KeyId=key_id,
        Message=_canonical_json_bytes(message),
        MessageType="RAW",
        Signature=base64.b64decode(attestation["signature"]),
        SigningAlgorithm=KMS_SIGNING_ALGORITHM,
    )
    return bool(result.get("SignatureValid"))


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    path_params = event.get("pathParameters") or {}
    query_params = event.get("queryStringParameters") or {}
    bundle_id = path_params.get("bundle_id") or query_params.get("bundle_id")
    if not bundle_id:
        return _response(400, {"ok": False, "error": "bundle_id is required"})

    table_name = os.environ["RUN_METADATA_TABLE"]
    bucket_name = os.environ["EVIDENCE_BUCKET"]
    key_id = os.environ["ATTESTATION_KEY_ID"]
    region = os.environ.get("AWS_REGION", "us-east-1")

    dynamodb = boto3.client("dynamodb", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    kms = boto3.client("kms", region_name=region)

    item = dynamodb.get_item(
        TableName=table_name,
        Key={"bundle_id": {"S": bundle_id}, "record_type": {"S": "CHAIN_HEAD"}},
        ConsistentRead=True,
    ).get("Item")
    if item is None:
        return _response(404, {"ok": False, "bundle_id": bundle_id, "error": "not found"})

    object_key = item["object_key"]["S"]
    raw = s3.get_object(Bucket=bucket_name, Key=object_key)["Body"].read()
    bundle = json.loads(raw.decode("utf-8"))
    verification = bundle["verification"]
    records = verification["records"]
    root_hash = verification["trusted_root"]
    inclusion_ok = all(
        _verify_inclusion(record, proof, root_hash)
        for record, proof in zip(records, verification["inclusion_proofs"], strict=True)
    )
    kms_ok = _verify_kms_signature(kms, key_id, bundle.get("enterprise_attestation") or {})
    sequence_ok = int(item["sequence_number"]["N"]) == int(
        bundle["enterprise_attestation"]["sequence_number"]
    )
    ok = inclusion_ok and kms_ok and sequence_ok
    return _response(
        200,
        {
            "ok": ok,
            "bundle_id": bundle_id,
            "merkle_root": root_hash,
            "tree_size": verification["tree_size"],
            "inclusion_ok": inclusion_ok,
            "kms_signature_ok": kms_ok,
            "sequence_ok": sequence_ok,
        },
    )


def _response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body, sort_keys=True),
    }
