# tests/unit/models/test_models_hashing.py

"""
Hashing helper tests for cc.core.models.

Scope:
- _hash_json: determinism, salt behaviour, basic collision sanity,
  Unicode normalization and float stability.
- _hash_text: determinism, salt behaviour, bytes vs str semantics,
  invalid UTF-8 behaviour.

These tests are about **low-level hashing invariants only** and do not
touch any of the model classes.
"""

import json
import unicodedata
from typing import Any, Dict

from hypothesis import assume, given
from hypothesis import strategies as st

from cc.core.models import _hash_json, _hash_text

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """
    Canonical JSON representation for collision sanity tests.

    We use this to ensure that "different" Python objects that serialize
    to the same canonical JSON string also have identical hashes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------
# _hash_json invariants
# ---------------------------------------------------------------------


@given(
    obj=st.dictionaries(
        keys=st.text(min_size=0, max_size=10),
        values=st.integers(),
        max_size=10,
    )
)
def test_hash_json_deterministic_no_salt(obj: Dict[str, int]):
    """
    _hash_json must be deterministic for the same object when no salt is used.

    WHAT:
        Hash the same logical object multiple times, including a deep-cloned
        version via JSON roundtrip.
    WHY:
        Many parts of the system rely on stable hashes for deduplication.
    THREAT:
        If hashing depended on object identity or non-deterministic iteration,
        deduplication and cache keys would become unreliable.
    """
    h1 = _hash_json(obj)
    h2 = _hash_json(obj)

    # Defensive copy through JSON to force a re-materialized dict
    obj_clone = json.loads(json.dumps(obj))
    h3 = _hash_json(obj_clone)

    assert isinstance(h1, str)
    assert h1 == h2 == h3


@given(
    obj=st.dictionaries(
        keys=st.text(min_size=0, max_size=10),
        values=st.integers(),
        max_size=10,
    ),
    salt=st.binary(min_size=1, max_size=16),
)
def test_hash_json_salt_changes_hash(obj: Dict[str, int], salt: bytes):
    """
    Adding a non-empty salt should almost always change the hash.

    WHAT:
        Compare unsalted vs salted hashes for the same object.
    WHY:
        Salting is used to separate hash namespaces (e.g., env vs transcript).
    THREAT:
        If salt were ignored, different contexts could accidentally collide.
    """
    h_plain = _hash_json(obj)
    h_salted = _hash_json(obj, salt=salt)
    assert h_plain != h_salted


def test_hash_json_collision_smoke_for_unique_canonical_inputs():
    """
    Smoke test: 10k distinct canonical JSON strings should not collide.

    WHAT:
        Hash a large family of small dicts and check:
        - identical canonical JSON → identical hash
        - distinct canonical JSON → distinct hash
    WHY:
        Guards against accidental low-entropy mistakes in hashing logic.
    NOTE:
        This is NOT a cryptographic proof, just a regression tripwire.
    """
    objs = [{"k": i, "v": i * 2, "w": i**2} for i in range(10_000)]
    canonicals = {canonical_json(o) for o in objs}
    assert len(canonicals) == len(objs)

    hashes: Dict[str, str] = {}
    for o in objs:
        cj = canonical_json(o)
        h = _hash_json(o)
        if cj in hashes:
            # Same canonical JSON => must be same hash
            assert hashes[cj] == h
        else:
            hashes[cj] = h

    # No collisions across canonical-distinct JSONs
    assert len(set(hashes.values())) == len(hashes)


def test_hash_json_unicode_normalization_nfc_vs_nfd():
    """
    NFC vs NFD normalized strings should hash identically.

    WHAT:
        Compare hashes for {"name": "café"} under NFC and NFD normalization.
    WHY:
        Prevents platform-dependent behaviour in the presence of composed
        vs decomposed Unicode forms.
    THREAT:
        Without normalization, logically identical strings could hash
        differently depending on source.
    """
    text_nfc = unicodedata.normalize("NFC", "café")
    text_nfd = unicodedata.normalize("NFD", "café")

    obj_nfc = {"name": text_nfc}
    obj_nfd = {"name": text_nfd}

    assert _hash_json(obj_nfc) == _hash_json(obj_nfd)


def test_hash_json_float_precision_stability():
    """
    0.1 represented in two mathematically equivalent ways must hash the same.

    WHAT:
        Use literal 0.1 and 1.0/10.0, which share the same IEEE-754 value.
    WHY:
        Documents that JSON float serialization is stable for simple cases.
    THREAT:
        If we changed serialization mode or rounding, hashes might shift
        in subtle ways across platforms or versions.
    """
    o1 = {"x": 0.1}
    o2 = {"x": 1.0 / 10.0}

    # Sanity: Python treats these as the same float
    assert o1["x"] == o2["x"]

    assert _hash_json(o1) == _hash_json(o2)


# ---------------------------------------------------------------------
# _hash_text invariants
# ---------------------------------------------------------------------


@given(
    text=st.text(min_size=0, max_size=50),
    salt=st.binary(min_size=1, max_size=16),
)
def test_hash_text_determinism_and_salt(text: str, salt: bytes):
    """
    _hash_text must be deterministic and salt must perturb the hash.

    WHAT:
        Hash the same text twice (no salt) and once with salt.
    WHY:
        Same semantics as _hash_json: stable hashes plus salt separation.
    """
    h_plain1 = _hash_text(text)
    h_plain2 = _hash_text(text)
    assert h_plain1 == h_plain2

    h_salted = _hash_text(text, salt=salt)
    assert h_salted != h_plain1


@given(st.text(min_size=0, max_size=50))
def test_hash_text_bytes_and_str_match_for_valid_utf8(s: str):
    """
    For valid UTF-8, hashing bytes or the decoded string must match.

    WHAT:
        Generate arbitrary Unicode text (valid by construction),
        encode to UTF-8 bytes, and compare hashes for:
          - raw bytes
          - decoded string
    WHY:
        Ensures we don't accidentally treat "the same text" differently
        depending on input type.
    """
    data = s.encode("utf-8")

    h_bytes = _hash_text(data)
    h_str = _hash_text(s)
    assert h_bytes == h_str


@given(st.binary(min_size=1, max_size=64))
def test_hash_text_invalid_utf8_not_equal_to_replacement(data: bytes):
    """
    For invalid UTF-8, raw bytes hash must differ from 'replacement' decoding.

    WHAT:
        For bytes that *don't* decode as UTF-8, compare:
          - hash(raw bytes)
          - hash(decoded with errors='replace')
    WHY:
        Documents that the hash is ultimately over the raw bytes, not the
        lossy "replacement" text.
    THREAT:
        If these collapsed to the same hash, we could confuse corrupted
        payloads with clean ones.
    """
    try:
        data.decode("utf-8")
        assume(False)  # text is valid; skip this example
    except UnicodeDecodeError:
        pass

    h_bytes = _hash_text(data)
    decoded = data.decode("utf-8", errors="replace")
    h_replaced = _hash_text(decoded)

    assert h_bytes != h_replaced
