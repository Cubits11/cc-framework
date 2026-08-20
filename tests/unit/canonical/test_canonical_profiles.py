"""The canonicalization corpus, enforced.

Every class in ``scripts/canonicalization_probe.py`` is asserted here, so a
regression in the kernel fails the suite rather than only the probe. The
binding rule for this file: **never weaken a case to make a test pass.** A
class that starts failing is a kernel regression, not a corpus problem.

Provenance is ``census``. The classes are curated and adversarial, their number
is an authoring decision, and no coverage claim or confidence interval attaches
to a pass count over them.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from cc.reporting.canonical import (
    DEFAULT_PROFILE,
    LEGACY_SORT_KEYS,
    MAX_SAFE_INTEGER,
    RFC8785,
    CanonicalJSONError,
    ConfusableKeyError,
    DuplicateJSONKeyError,
    assert_no_confusable_keys,
    canonical_json_bytes,
    sha256_canonical,
    strict_json_loads,
)

ROOT = Path(__file__).resolve().parents[3]

KEY_NFC = "é"  # e-acute, precomposed
KEY_NFD = "é"  # e + combining acute


def _probe_json() -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "canonicalization_probe.py"), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    return json.loads(proc.stdout)


# --- the corpus gate ----------------------------------------------------------


def test_probe_reports_no_unintended_kernel_under_the_default_profile():
    """The acceptance gate for W3: zero unintended-kernel, zero rejection-asymmetry."""
    report = _probe_json()
    v2 = report["profiles"]["v2"]
    counts = v2["verdict_counts"]

    assert counts.get("unintended-kernel", 0) == 0, v2["classes"]
    assert counts.get("rejection-asymmetry", 0) == 0, v2["classes"]
    assert counts.get("over-discrimination", 0) == 0, v2["classes"]
    assert v2["failing"] == 0
    # The census must carry its own boundary, not just its counts.
    assert report["provenance"] == "census"
    assert report["confidence_intervals"] is None


def test_probe_still_records_the_legacy_defects():
    """v1 must keep failing. It is retained as history, not as an option.

    If this ever passes trivially it means the legacy profile was "fixed",
    which would silently change the hash of every pre-migration receipt and
    defeat the reason v1 exists at all.
    """
    v1 = _probe_json()["profiles"]["v1"]
    assert v1["verdict_counts"].get("unintended-kernel", 0) == 2
    assert v1["silent_key_loss"]["keys_out"] < v1["silent_key_loss"]["keys_in"]


# --- F-03: the silent key merge -----------------------------------------------


def test_default_profile_keeps_unicode_distinct_keys_distinct():
    assert canonical_json_bytes({KEY_NFC: 1}) != canonical_json_bytes({KEY_NFD: 1})


def test_default_profile_does_not_lose_a_key():
    """Two byte-distinct keys go in; two come out, and neither is dropped."""
    record = {KEY_NFC: 1, KEY_NFD: 2, "plain": 3}
    decoded = strict_json_loads(canonical_json_bytes(record).decode("utf-8"))
    assert len(decoded) == 3
    assert decoded[KEY_NFC] == 1
    assert decoded[KEY_NFD] == 2


def test_nested_unicode_keys_also_survive():
    """A guard inspecting only top-level keys would miss this."""
    assert canonical_json_bytes({"outer": {KEY_NFC: 1}}) != canonical_json_bytes(
        {"outer": {KEY_NFD: 1}}
    )


def test_legacy_profile_still_merges_them():
    """Bug-compatible on purpose: old receipts must stay verifiable."""
    merged = canonical_json_bytes({KEY_NFC: 1, KEY_NFD: 2}, profile=LEGACY_SORT_KEYS)
    assert len(json.loads(merged)) == 1


# --- the opt-in producer lint -------------------------------------------------


def test_confusable_key_lint_flags_what_canonicalization_permits():
    record = {KEY_NFC: 1, KEY_NFD: 2}
    canonical_json_bytes(record)  # canonicalization itself is content-neutral
    with pytest.raises(ConfusableKeyError, match="normal form"):
        assert_no_confusable_keys(record)


def test_confusable_key_lint_descends_into_nested_structures():
    with pytest.raises(ConfusableKeyError):
        assert_no_confusable_keys({"a": [{"b": {KEY_NFC: 1, KEY_NFD: 2}}]})


def test_confusable_key_lint_accepts_ordinary_documents():
    assert_no_confusable_keys({"a": 1, "b": [{"c": 2}], "d": None})


# --- F-04: RFC 8785 number forms ----------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "1"),
        (-0.0, "0"),
        (0.0, "0"),
        (100.0, "100"),
        (0.1, "0.1"),
        (1.5, "1.5"),
        (1e-7, "1e-7"),
        (1e-6, "0.000001"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (1e30, "1e+30"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (-1.5, "-1.5"),
        (1.2345e-10, "1.2345e-10"),
        (1234567890.0, "1234567890"),
    ],
)
def test_number_forms_follow_ecmascript_tostring(value, expected):
    assert canonical_json_bytes({"n": value}).decode("utf-8") == f'{{"n":{expected}}}'


def test_negative_zero_and_zero_share_an_identity():
    """scipy returns -0.0 at a zero lower bound, so this reaches real reports."""
    assert sha256_canonical({"n": -0.0}) == sha256_canonical({"n": 0.0})


def test_int_and_float_of_equal_value_share_an_identity():
    """JSON has one number type; 1 and 1.0 are the same JSON number."""
    assert canonical_json_bytes({"n": 1}) == canonical_json_bytes({"n": 1.0})


# --- F-05: the integer safe range ---------------------------------------------


@pytest.mark.parametrize("value", [2**53, 2**53 + 1, 10**30, -(2**53), -(10**30)])
def test_integers_outside_the_safe_range_are_refused(value):
    """A JS JSON.parse collapses these before any verifier code runs."""
    with pytest.raises(CanonicalJSONError, match="safe range"):
        canonical_json_bytes({"n": value})


@pytest.mark.parametrize("value", [0, 1, -1, MAX_SAFE_INTEGER, -MAX_SAFE_INTEGER])
def test_integers_inside_the_safe_range_are_accepted(value):
    assert canonical_json_bytes({"n": value}) == f'{{"n":{value}}}'.encode()


def test_floats_are_not_subject_to_the_integer_bound():
    """The bound exists because Python ints are unbounded. Floats already are not."""
    assert canonical_json_bytes({"n": 1e300}).decode("utf-8") == '{"n":1e+300}'


# --- F-07: duplicate keys on read ---------------------------------------------


def test_strict_json_loads_refuses_a_repeated_key():
    with pytest.raises(DuplicateJSONKeyError, match="repeats key"):
        strict_json_loads('{"amount":1,"amount":2}')


def test_strict_json_loads_refuses_a_repeated_key_when_nested():
    with pytest.raises(DuplicateJSONKeyError):
        strict_json_loads('{"outer":{"a":1,"a":2}}')


def test_strict_json_loads_accepts_ordinary_json():
    assert strict_json_loads('{"a":1,"b":[1,2],"c":{"d":null}}') == {
        "a": 1,
        "b": [1, 2],
        "c": {"d": None},
    }


# --- string and key handling --------------------------------------------------


def test_control_characters_are_escaped_with_short_forms_where_defined():
    # \u0001 has no short escape, so it must come out as \\u0001; tab and
    # newline do have one and must not be emitted as \\u0009 / \\u000a.
    got = canonical_json_bytes({"k": "a\tb\nc\u0001d"}).decode("utf-8")
    assert got == '{"k":"a\\tb\\nc\\u0001d"}'


def test_non_ascii_is_emitted_literally_not_escaped():
    text = "café"
    assert canonical_json_bytes({"k": text}).decode("utf-8") == f'{{"k":"{text}"}}'


def test_keys_sort_by_utf16_code_unit_not_code_point():
    """The orders disagree above the BMP.

    U+10000 encodes as the surrogate pair D800 DC00, so it sorts *before*
    U+FFFD under UTF-16 and *after* it under code point.
    """
    bmp = "�"
    astral = "\U00010000"
    got = canonical_json_bytes({bmp: 1, astral: 2}).decode("utf-8")
    assert got.index(astral) < got.index(bmp)


def test_unpaired_surrogates_are_refused():
    lone = "\ud800"
    with pytest.raises(CanonicalJSONError, match="surrogate"):
        canonical_json_bytes({"k": lone})
    with pytest.raises(CanonicalJSONError, match="surrogate"):
        canonical_json_bytes({lone: "v"})


# --- profile plumbing ---------------------------------------------------------


def test_default_profile_is_the_rfc8785_one():
    assert DEFAULT_PROFILE == RFC8785


def test_unknown_profile_fails_closed():
    with pytest.raises(CanonicalJSONError, match="unknown canonicalization profile"):
        canonical_json_bytes({"a": 1}, profile="cc.canonical.v99")  # type: ignore[arg-type]


def test_profiles_disagree_which_is_the_point_of_versioning():
    """If these ever agreed, dispatching on the declared profile would be moot."""
    payload = {"n": 1.0, "m": -0.0}
    assert canonical_json_bytes(payload, profile=RFC8785) != canonical_json_bytes(
        payload, profile=LEGACY_SORT_KEYS
    )


# --- shared refusals ----------------------------------------------------------


@pytest.mark.parametrize("profile", [RFC8785, LEGACY_SORT_KEYS])
def test_both_profiles_refuse_non_finite_floats(profile):
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(CanonicalJSONError):
            canonical_json_bytes({"n": bad}, profile=profile)


@pytest.mark.parametrize("profile", [RFC8785, LEGACY_SORT_KEYS])
def test_both_profiles_refuse_non_string_keys(profile):
    with pytest.raises(CanonicalJSONError, match="non-string key"):
        canonical_json_bytes({1: "v"}, profile=profile)  # type: ignore[dict-item]


@pytest.mark.parametrize("profile", [RFC8785, LEGACY_SORT_KEYS])
def test_both_profiles_refuse_non_json_native_values(profile):
    for bad in ({"s"}, b"bytes", object()):
        with pytest.raises(CanonicalJSONError, match="non-JSON-native"):
            canonical_json_bytes({"k": bad}, profile=profile)


# --- determinism --------------------------------------------------------------


def test_key_order_does_not_affect_identity():
    assert canonical_json_bytes({"b": 1, "a": 2}) == canonical_json_bytes({"a": 2, "b": 1})


def test_array_order_does_affect_identity():
    assert canonical_json_bytes({"xs": [1, 2]}) != canonical_json_bytes({"xs": [2, 1]})


def test_canonical_bytes_are_stable_across_repeated_calls():
    payload = {"z": [1, {"b": 2, "a": 1}], "a": "x", "n": 0.1}
    first = canonical_json_bytes(payload)
    assert all(canonical_json_bytes(payload) == first for _ in range(50))
