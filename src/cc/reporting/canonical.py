"""Canonical JSON helpers for CC report receipts.

The receipt hash is SHA-256 over a strict JSON representation:
sorted object keys, compact separators, UTF-8 bytes, and no NaN/Infinity.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class CanonicalJSONError(TypeError):
    """Raised when a value cannot be represented in canonical report JSON."""


def canonical_json_bytes(obj: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a JSON-native mapping.

    Non-string mapping keys, non-finite floats, bytes, sets, and arbitrary
    objects are rejected instead of being stringified implicitly.
    """

    normalized = _normalize_json_value(obj, path="$")
    if not isinstance(normalized, dict):
        raise CanonicalJSONError("canonical_json_bytes requires a JSON object at the top level")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_canonical(obj: Mapping[str, Any], *, exclude_receipt_hash: bool = True) -> str:
    """Hash canonical JSON bytes with SHA-256.

    When ``exclude_receipt_hash`` is true, ``receipt.canonical_hash`` is removed
    before hashing so reports can contain their own receipt without a circular
    dependency.
    """

    payload: Mapping[str, Any] = _without_receipt_hash(obj) if exclude_receipt_hash else obj
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _without_receipt_hash(obj: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json_value(obj, path="$")
    if not isinstance(normalized, dict):
        raise CanonicalJSONError("report payload must be a JSON object")
    report = dict(normalized)
    receipt = report.get("receipt")
    if isinstance(receipt, dict):
        receipt_copy = dict(receipt)
        receipt_copy.pop("canonical_hash", None)
        report["receipt"] = receipt_copy
    return report


def _normalize_json_value(value: Any, *, path: str) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(f"{path} must be finite, got {value!r}")
        return value
    if isinstance(value, Mapping):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{path} contains non-string key {key!r}")
            norm_key = unicodedata.normalize("NFC", key)
            out[norm_key] = _normalize_json_value(item, path=f"{path}.{norm_key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [
            _normalize_json_value(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)
        ]
    raise CanonicalJSONError(
        f"{path} contains non-JSON-native value of type {type(value).__name__}"
    )
