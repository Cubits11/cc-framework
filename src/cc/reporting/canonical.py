"""Canonical JSON for CC receipts, as versioned profiles.

A receipt identifies a document only up to the kernel of its canonicalizer:
the set of distinct documents that receive the same canonical bytes. Every
member of that kernel is a pair of documents a receipt cannot tell apart. The
job here is to keep that kernel equal to JSON equality -- no larger, and no
smaller.

Two profiles exist.

``cc.canonical.v2`` (:data:`RFC8785`) is the default for new artifacts and
implements RFC 8785, the JSON Canonicalization Scheme. It is what makes an
independent verifier in another language possible: JCS pins number formatting
to the ECMAScript ``Number::toString`` algorithm and key ordering to UTF-16
code-unit order, so a conforming implementation in any language produces the
same bytes.

``cc.canonical.v1`` (:data:`LEGACY_SORT_KEYS`) is the original
``json.dumps(sort_keys=True, ...)`` method. It is retained **read-only** so
receipts written before the migration stay verifiable. Breaking historical
receipts to fix the canonicalizer would trade one integrity failure for
another.

Every receipt names its profile in ``receipt.canonicalization_method``, and
verification dispatches on that name rather than assuming the current default.

Why v1 is not merely "the old one"
----------------------------------

An adversarial census (``scripts/canonicalization_probe.py``) found two
``unintended-kernel`` classes in v1 -- documents a consumer needs
distinguished that received identical bytes:

* v1 applied ``unicodedata.normalize("NFC", ...)`` to every mapping key and
  wrote the results into a fresh dict. Two byte-distinct keys sharing an NFC
  form silently became one, with no error, at top level and nested. The receipt
  then attested to a document with a field missing.
* v1 emitted Python ``repr`` number forms (``1.0``, ``-0.0``, ``1e-07``),
  which diverge from RFC 8785 on five of six probed forms, so no non-Python
  verifier could agree on the bytes.

v2 fixes both by construction rather than by patch. It does **not** normalize:
RFC 8785 is explicit that normalization is the producer's responsibility, and a
canonicalizer that mutates content is not a canonicalizer. Two keys that differ
in Unicode form are two keys, which is what JSON says they are, so there is
nothing left to collide. Producers that want the stricter reading can call
:func:`assert_no_confusable_keys` as a separate lint -- it is deliberately not
part of the hash path.

Both profiles reject non-finite floats, non-string keys, and non-JSON-native
values. v2 additionally rejects integers outside the IEEE-754 safe range (see
:data:`MAX_SAFE_INTEGER`) and strings containing unpaired surrogates.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final, Literal, TypeAlias

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

#: Identifier written into ``receipt.canonicalization_method`` by the original
#: method. Retained verbatim so pre-migration receipts still validate.
LEGACY_SORT_KEYS: Final = (
    "json.dumps(sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False); "
    "receipt.canonical_hash excluded"
)

#: Identifier for the RFC 8785 profile. Default for new artifacts.
RFC8785: Final = "cc.canonical.v2/RFC8785; receipt.canonical_hash excluded"

CanonicalProfile: TypeAlias = Literal[
    "json.dumps(sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False); "
    "receipt.canonical_hash excluded",
    "cc.canonical.v2/RFC8785; receipt.canonical_hash excluded",
]

DEFAULT_PROFILE: Final[CanonicalProfile] = RFC8785

#: ``2**53 - 1``. Integers beyond this are not exactly representable as IEEE-754
#: doubles, so a JavaScript ``JSON.parse`` collapses neighbouring values before
#: any verifier code runs -- the kernel is set by the parser, and no downstream
#: fix reaches it. v2 refuses them rather than emitting bytes that cannot
#: survive a round trip.
MAX_SAFE_INTEGER: Final = 9007199254740991

__all__ = [
    "DEFAULT_PROFILE",
    "LEGACY_SORT_KEYS",
    "MAX_SAFE_INTEGER",
    "RFC8785",
    "CanonicalJSONError",
    "CanonicalProfile",
    "ConfusableKeyError",
    "assert_no_confusable_keys",
    "canonical_json_bytes",
    "sha256_canonical",
    "strict_json_loads",
]


class CanonicalJSONError(TypeError):
    """Raised when a value cannot be represented in canonical report JSON."""


class ConfusableKeyError(CanonicalJSONError):
    """Raised when two byte-distinct keys share a Unicode normal form.

    Under RFC 8785 these are two separate keys and canonicalization proceeds.
    This error is raised only by the opt-in :func:`assert_no_confusable_keys`
    lint, because a document containing both is nearly always a producer bug --
    two fields where one was meant.
    """


class DuplicateJSONKeyError(CanonicalJSONError):
    """Raised when a parsed JSON object repeats a key.

    ``json.loads`` keeps the last occurrence silently, so a receipt computed
    over the parsed result attests to a document the sender did not send.
    """


# --------------------------------------------------------------------------
# RFC 8785 number serialization
# --------------------------------------------------------------------------


def _es_number_to_string(value: float) -> str:
    """Serialize a double per the ECMAScript ``Number::toString`` algorithm.

    RFC 8785 section 3.2.2.3 defers to ECMA-262. Python's ``repr`` already
    yields the shortest round-tripping digits; what differs is the *formatting*
    of those digits, which this function supplies.
    """
    if not math.isfinite(value):
        raise CanonicalJSONError(f"non-finite numbers are not JSON: {value!r}")
    # Covers -0.0, which RFC 8785 requires be emitted as "0".
    if value == 0.0:
        return "0"
    if value < 0:
        return "-" + _es_number_to_string(-value)

    _, raw_digits, exponent = Decimal(repr(value)).as_tuple()
    digits = "".join(str(d) for d in raw_digits)
    exponent = int(exponent)
    # Trailing zeros are an artifact of repr ("1.0" -> digits "10"); the ES
    # algorithm requires the shortest digit string k.
    while len(digits) > 1 and digits.endswith("0"):
        digits = digits[:-1]
        exponent += 1

    k = len(digits)
    n = k + exponent  # value == 0.<digits> * 10**n

    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    sign = "+" if n - 1 >= 0 else "-"
    exp = f"e{sign}{abs(n - 1)}"
    if k == 1:
        return digits + exp
    return digits[0] + "." + digits[1:] + exp


_SHORT_ESCAPES: Final[dict[str, str]] = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _es_quote(text: str) -> str:
    """Quote a string per RFC 8785 section 3.2.2.2.

    Control characters below 0x20 are escaped, using the short forms where they
    exist; everything else is emitted literally as UTF-8. Non-ASCII is **not**
    escaped, and no normalization is applied.
    """
    out = ['"']
    for char in text:
        escape = _SHORT_ESCAPES.get(char)
        if escape is not None:
            out.append(escape)
        elif char < "\x20":
            out.append(f"\\u{ord(char):04x}")
        elif "\ud800" <= char <= "\udfff":
            # An unpaired surrogate cannot be encoded as UTF-8. Emitting one
            # would produce bytes no conforming parser can read, so refuse.
            raise CanonicalJSONError(
                f"string contains an unpaired surrogate U+{ord(char):04X}; "
                "it has no UTF-8 encoding and cannot be canonicalized"
            )
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _utf16_sort_key(key: str) -> bytes:
    """Return a sort key ordering strings by UTF-16 code unit.

    RFC 8785 sorts object keys by UTF-16 code units, **not** by code point.
    The two orders differ above the BMP: U+10000 encodes as the surrogate pair
    D800 DC00, so it sorts *before* U+FFFD under UTF-16 and *after* it under
    code point. Comparing big-endian UTF-16 bytes reproduces the required
    order exactly.
    """
    try:
        return key.encode("utf-16-be")
    except UnicodeEncodeError as exc:
        raise CanonicalJSONError(
            f"object key {key!r} contains an unpaired surrogate and cannot be ordered"
        ) from exc


def _serialize_rfc8785(value: Any, *, path: str) -> str:
    if value is None:
        return "null"
    # bool must precede int: Python's bool is an int subclass.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _es_quote(value)
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalJSONError(
                f"{path} integer {value} exceeds the IEEE-754 safe range "
                f"(|n| <= {MAX_SAFE_INTEGER}). A JSON parser backed by doubles "
                "collapses values beyond it, so no cross-language verifier "
                "could agree on this document. Carry the value as a string."
            )
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError(f"{path} must be finite, got {value!r}")
        return _es_number_to_string(value)
    if isinstance(value, Mapping):
        items = []
        seen: set[str] = set()
        for key in value:
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{path} contains non-string key {key!r}")
            if key in seen:  # pragma: no cover - a dict cannot repeat a key
                raise CanonicalJSONError(f"{path} repeats key {key!r}")
            seen.add(key)
            items.append(key)
        items.sort(key=_utf16_sort_key)
        parts = [
            f"{_es_quote(key)}:{_serialize_rfc8785(value[key], path=f'{path}.{key}')}"
            for key in items
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        parts = [
            _serialize_rfc8785(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
        return "[" + ",".join(parts) + "]"
    raise CanonicalJSONError(
        f"{path} contains non-JSON-native value of type {type(value).__name__}"
    )


# --------------------------------------------------------------------------
# Legacy profile (read-only)
# --------------------------------------------------------------------------


def _normalize_legacy(value: Any, *, path: str) -> JsonValue:
    """Reproduce the v1 normalization exactly, including its NFC key merge.

    This is deliberately bug-compatible. Its purpose is to recompute the hash a
    pre-migration receipt actually carries; "fixing" it here would make old
    receipts unverifiable, which is the failure this profile exists to prevent.
    """
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
            out[norm_key] = _normalize_legacy(item, path=f"{path}.{norm_key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalize_legacy(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    raise CanonicalJSONError(
        f"{path} contains non-JSON-native value of type {type(value).__name__}"
    )


def _serialize_legacy(obj: Mapping[str, Any]) -> bytes:
    normalized = _normalize_legacy(obj, path="$")
    if not isinstance(normalized, dict):
        raise CanonicalJSONError("canonical_json_bytes requires a JSON object at the top level")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------


def canonical_json_bytes(
    obj: Mapping[str, Any],
    *,
    profile: CanonicalProfile = DEFAULT_PROFILE,
) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a JSON-native mapping.

    Args:
        obj: A JSON object. Non-string keys, non-finite floats, bytes, sets,
            and arbitrary objects are rejected rather than stringified.
        profile: Which canonicalization profile to apply. Defaults to
            :data:`RFC8785`. Pass :data:`LEGACY_SORT_KEYS` only to recompute
            the hash of a pre-migration receipt.

    Raises:
        CanonicalJSONError: The value cannot be canonicalized, or (under
            :data:`RFC8785`) carries an integer outside the safe range or an
            unpaired surrogate.
    """
    if profile == LEGACY_SORT_KEYS:
        return _serialize_legacy(obj)
    if profile != RFC8785:
        raise CanonicalJSONError(
            f"unknown canonicalization profile {profile!r}; expected "
            f"{RFC8785!r} or {LEGACY_SORT_KEYS!r}"
        )
    if not isinstance(obj, Mapping):
        raise CanonicalJSONError("canonical_json_bytes requires a JSON object at the top level")
    return _serialize_rfc8785(obj, path="$").encode("utf-8")


def sha256_canonical(
    obj: Mapping[str, Any],
    *,
    exclude_receipt_hash: bool = True,
    profile: CanonicalProfile = DEFAULT_PROFILE,
) -> str:
    """Hash canonical JSON bytes with SHA-256.

    When ``exclude_receipt_hash`` is true, ``receipt.canonical_hash`` is
    removed before hashing so a report can contain its own receipt without a
    circular dependency.
    """
    payload: Mapping[str, Any] = _without_receipt_hash(obj) if exclude_receipt_hash else obj
    return hashlib.sha256(canonical_json_bytes(payload, profile=profile)).hexdigest()


def strict_json_loads(text: str | bytes) -> Any:
    """Parse JSON, refusing any object that repeats a key.

    ``json.loads`` keeps the last occurrence of a repeated key and reports
    nothing. A receipt computed over that result attests to a document the
    sender did not send, so every receipt-covered read goes through here.
    """

    def _hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise DuplicateJSONKeyError(
                    f"JSON object repeats key {key!r}; the value that would "
                    "survive is not the document that was sent"
                )
            seen.add(key)
        return dict(pairs)

    return json.loads(text, object_pairs_hook=_hook)


def assert_no_confusable_keys(obj: Any, *, path: str = "$") -> None:
    """Raise if any object contains two keys sharing a Unicode normal form.

    This is an **opt-in producer lint**, not part of canonicalization. Under
    RFC 8785 such keys are simply two keys and hashing proceeds correctly; but
    a document carrying both is nearly always a bug -- two fields where one was
    intended -- and a producer that would rather fail than ship it can call
    this first.

    Keeping it out of the hash path is deliberate. v1 merged such keys *inside*
    canonicalization and silently dropped a field; the lesson is that a
    canonicalizer must not mutate, and a mutation-free canonicalizer has no
    business deciding a document is wrong.
    """
    if isinstance(obj, Mapping):
        by_normal_form: dict[str, str] = {}
        for key in obj:
            if not isinstance(key, str):
                continue
            normal = unicodedata.normalize("NFC", key)
            previous = by_normal_form.get(normal)
            if previous is not None and previous != key:
                raise ConfusableKeyError(
                    f"{path} contains keys {previous!r} and {key!r}, which are "
                    "byte-distinct but share the Unicode normal form "
                    f"{normal!r}. Under RFC 8785 these are two separate fields."
                )
            by_normal_form[normal] = key
        for key, value in obj.items():
            assert_no_confusable_keys(value, path=f"{path}.{key}")
    elif isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
        for index, item in enumerate(obj):
            assert_no_confusable_keys(item, path=f"{path}[{index}]")


def _without_receipt_hash(obj: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise CanonicalJSONError("report payload must be a JSON object")
    report = dict(obj)
    receipt = report.get("receipt")
    if isinstance(receipt, Mapping):
        receipt_copy = dict(receipt)
        receipt_copy.pop("canonical_hash", None)
        report["receipt"] = receipt_copy
    return report
