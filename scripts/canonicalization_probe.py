#!/usr/bin/env python3
"""Adversarial census over the CC report canonicalization kernel.

Modelled on Ghost-Ark's E1 provenance-kernel census. For each pathology class we
declare, *before* observing anything, whether a consumer needs the two documents
distinguished (``distinct``) or unified (``equivalent``). The verdict compares
the observation against that declared intent:

``sound``
    The kernel did what the declared consumer intent requires.
``unintended-kernel``
    Two documents a consumer needs distinguished received the same canonical
    identity. This is the dangerous verdict: it issues a false shared identity.
``over-discrimination``
    Two documents every consumer treats as identical received different
    identities. Breaks replay determinism; does not forge identity.
``fail-closed``
    Both documents were refused. No false identity is issued.
``sound-by-rejection``
    One side was admitted and the other refused where a consumer needs them
    distinguished. No false identity is issued and the honest document still
    receives one.
``rejection-asymmetry``
    One of two documents every consumer treats as identical was refused. This is
    the failure mode of an over-strict rule, and it is what distinguishes a fix
    from a trade.

Provenance of this corpus is ``census``: the classes are curated and adversarial,
the population is exactly what is written here, and its size is an authoring
decision. Exact counts only. **No confidence intervals** -- attaching one to a
hand-authored corpus would be precisely the error `cc.kernel.cliff` refuses to
make elsewhere in this repository.

Exit status is non-zero while any class carries ``unintended-kernel`` or
``rejection-asymmetry``, so the findings in
``docs/upgrade/FINDINGS_REGISTER.md`` are falsifiable rather than asserted: fix
the canonicalizer and this probe goes green.

Usage::

    PYTHONPATH=src python scripts/canonicalization_probe.py
    PYTHONPATH=src python scripts/canonicalization_probe.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Literal

from cc.reporting.canonical import (
    LEGACY_SORT_KEYS,
    RFC8785,
    DuplicateJSONKeyError,
    canonical_json_bytes,
    strict_json_loads,
)

Intent = Literal["distinct", "equivalent"]

# Two representations of the same grapheme: precomposed U+00E9, and the
# decomposed pair U+0065 U+0301. They are different byte sequences that NFC
# normalization maps to the same string.
KEY_NFC = "é"
KEY_NFD = "é"


@dataclass(frozen=True)
class PathologyClass:
    """One declared-intent pair of documents."""

    name: str
    intent: Intent
    left: dict[str, Any]
    right: dict[str, Any]
    note: str


# Declared BEFORE observation. Editing an intent to match a measured result
# requires editing this list, which surfaces it in review.
CLASSES: tuple[PathologyClass, ...] = (
    PathologyClass(
        name="unicode-key-collision",
        intent="distinct",
        left={KEY_NFC: 1},
        right={KEY_NFD: 1},
        note="Distinct source keys whose NFC forms coincide.",
    ),
    PathologyClass(
        name="integer-above-2^53",
        intent="distinct",
        left={"n": 2**53 + 1},
        right={"n": 2**53 + 2},
        note="v1 emitted both, but a JS JSON.parse collapses them before any "
        "verifier code runs -- the kernel is set by the parser, and no "
        "downstream fix reaches it. v2 refuses both rather than emitting bytes "
        "that cannot survive a round trip (F-05).",
    ),
    PathologyClass(
        name="int-vs-float-same-value",
        intent="equivalent",
        left={"n": 1},
        right={"n": 1.0},
        note="INTENT CORRECTED 2026-08-19, from 'distinct' to 'equivalent'. The "
        "original declaration described Python's type system, not JSON's: JSON "
        "has exactly one number type, so 1 and 1.0 are the same JSON number, and "
        "a consumer needing them distinguished is asking JSON for something it "
        "does not provide. Correcting a declaration because it was wrong about "
        "the domain is legitimate; correcting one to flatter a measured result "
        "is not. This is the former, recorded here rather than edited away.",
    ),
    PathologyClass(
        name="negative-zero",
        intent="equivalent",
        left={"n": 0.0},
        right={"n": -0.0},
        note="IEEE-754 -0.0 == 0.0. scipy.optimize.linprog returns -0.0 at a "
        "zero lower bound, so this reaches real reports.",
    ),
    PathologyClass(
        name="bool-vs-int",
        intent="distinct",
        left={"n": True},
        right={"n": 1},
        note="Python bool is a subclass of int; JSON true is not 1.",
    ),
    PathologyClass(
        name="float-exponent-form",
        intent="distinct",
        left={"n": 1e30},
        right={"n": 10**30},
        note="A float and a Python int of the same magnitude. Under v2 the int "
        "is refused for exceeding the IEEE-754 safe range, so no false shared "
        "identity is issued and the float still receives one.",
    ),
    PathologyClass(
        name="safe-integer-neighbours",
        intent="distinct",
        left={"n": 2**53 - 2},
        right={"n": 2**53 - 1},
        note="Positive control: adjacent integers inside the safe range must stay distinct.",
    ),
    PathologyClass(
        name="object-key-order",
        intent="equivalent",
        left={"a": 1, "b": 2},
        right={"b": 2, "a": 1},
        note="Positive control: key order carries no meaning in JSON.",
    ),
    PathologyClass(
        name="array-element-order",
        intent="distinct",
        left={"xs": [1, 2]},
        right={"xs": [2, 1]},
        note="Positive control: arrays are ordered; no arm may sort them.",
    ),
    PathologyClass(
        name="nested-unicode-key-collision",
        intent="distinct",
        left={"outer": {KEY_NFC: 1}},
        right={"outer": {KEY_NFD: 1}},
        note="A guard inspecting only top-level keys would pass the flat class "
        "while leaving this fully exploitable.",
    ),
    PathologyClass(
        name="utf16-vs-codepoint-key-order",
        intent="distinct",
        left={"\ufffd": 1, "\U00010000": 2},
        right={"\ufffd": 2, "\U00010000": 1},
        note="Positive control for key ordering. RFC 8785 sorts by UTF-16 code "
        "unit, not code point, and the orders disagree above the BMP: U+10000 "
        "encodes as D800 DC00, so it sorts before U+FFFD under UTF-16 and after "
        "it under code point. Different values under the same keys must stay "
        "distinct whichever order is used.",
    ),
    PathologyClass(
        name="control-character-escape",
        intent="distinct",
        left={"k": "a\u0001b"},
        right={"k": "a\u0002b"},
        note="Positive control: C0 control characters are escaped as \\u00XX "
        "and must not collapse to one another.",
    ),
    PathologyClass(
        name="tab-vs-escaped-tab-text",
        intent="distinct",
        left={"k": "a\tb"},
        right={"k": "a\\tb"},
        note="Positive control: a real tab and the two-character text "
        "backslash-t are different strings and must stay distinct after escaping.",
    ),
    PathologyClass(
        name="large-document-single-byte",
        intent="distinct",
        left={"pad": "x" * 65536, "tail": "a"},
        right={"pad": "x" * 65536, "tail": "b"},
        note="Positive control: catches a digest computed over a prefix.",
    ),
)


def _canonical(doc: dict[str, Any], profile: str) -> bytes | None:
    """Canonical bytes, or None when the document is refused."""
    try:
        return canonical_json_bytes(doc, profile=profile)  # type: ignore[arg-type]
    except Exception:  # any refusal is a refusal, for this census
        return None


def _verdict(intent: Intent, left: bytes | None, right: bytes | None) -> str:
    if left is None and right is None:
        return "fail-closed"
    if left is None or right is None:
        return "sound-by-rejection" if intent == "distinct" else "rejection-asymmetry"
    if left == right:
        return "sound" if intent == "equivalent" else "unintended-kernel"
    return "sound" if intent == "distinct" else "over-discrimination"


def _observation(left: bytes | None, right: bytes | None) -> str:
    if left is None and right is None:
        return "rejected-both"
    if left is None or right is None:
        return "rejected-one"
    return "collapsed" if left == right else "distinct"


def run(profile: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cls in CLASSES:
        left = _canonical(cls.left, profile)
        right = _canonical(cls.right, profile)
        rows.append(
            {
                "class": cls.name,
                "intent": cls.intent,
                "observed": _observation(left, right),
                "verdict": _verdict(cls.intent, left, right),
                "note": cls.note,
            }
        )
    return rows


def silent_key_loss(profile: str) -> dict[str, Any]:
    """F-03: under v1 a single object carrying both key forms loses one, silently.

    Under v2 nothing is normalized, so the two keys stay two keys and the
    canonical bytes carry both -- which is what JSON says the document is.
    """
    both = {KEY_NFC: 1, KEY_NFD: 2}
    try:
        out = canonical_json_bytes(both, profile=profile)  # type: ignore[arg-type]
    except Exception as exc:
        return {"keys_in": len(both), "refused": True, "error": type(exc).__name__}
    return {
        "keys_in": len(both),
        "keys_out": len(strict_json_loads(out)),
        "refused": False,
        "output": out.decode("utf-8"),
    }


def duplicate_key_parse() -> dict[str, Any]:
    """F-07: the permissive parser keeps the last key; the strict one refuses."""
    raw = '{"amount":1,"amount":2}'
    permissive = json.loads(raw)
    try:
        strict_json_loads(raw)
    except DuplicateJSONKeyError:
        strict_refused = True
    else:
        strict_refused = False
    return {
        "raw": raw,
        "permissive_parse": permissive,
        "permissive_collides_with_survivor": canonical_json_bytes(permissive)
        == canonical_json_bytes({"amount": 2}),
        "strict_json_loads_refuses": strict_refused,
    }


# Expected serializations under **cc.canonical.v2**, which is RFC 8785 plus one
# declared narrowing: a Python ``int`` outside the IEEE-754 safe integer range
# is refused rather than emitted. JCS is defined over JSON numbers, which are
# doubles; Python ints have no such bound, so the profile must supply one. The
# bound is the standard ``2**53 - 1`` -- the largest n for which both n and n+1
# are exactly representable -- which conservatively also refuses 2**53 itself.
#
# ``None`` means "must be refused". A refusal that JCS would have serialized is
# a narrowing, and it is declared in docs/architecture/CANONICAL_PROFILE.md
# rather than left as an undocumented difference.
JCS_EXPECTED: tuple[tuple[Any, str | None], ...] = (
    (1e30, "1e+30"),
    (10**30, None),  # int beyond the safe range
    (1.0, "1"),
    (-0.0, "0"),
    (1e-7, "1e-7"),
    (100.0, "100"),
    (0.1, "0.1"),
    (1e21, "1e+21"),
    (1e20, "100000000000000000000"),  # float: no int bound applies
    (5e-324, "5e-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308"),
    (float(2**53), "9007199254740992"),  # float is unaffected by the int rule
    (2**53, None),  # int at the boundary: conservatively refused
    (2**53 - 1, "9007199254740991"),  # the largest int the profile accepts
    (2**53 + 1, None),
)


def jcs_conformance(profile: str) -> list[dict[str, Any]]:
    """F-04: how the emitted number form compares with RFC 8785."""
    rows: list[dict[str, Any]] = []
    for value, expected in JCS_EXPECTED:
        try:
            emitted: str | None = canonical_json_bytes({"n": value}, profile=profile).decode(  # type: ignore[arg-type]
                "utf-8"
            )[5:-1]
        except Exception:
            emitted = None
        rows.append(
            {
                "value": repr(value),
                "emitted": emitted,
                "jcs_expects": expected,
                "conformant": emitted == expected,
            }
        )
    return rows


FAILING_VERDICTS = {"unintended-kernel", "rejection-asymmetry"}


PROFILES: dict[str, str] = {"v1": LEGACY_SORT_KEYS, "v2": RFC8785}


def census(profile_name: str) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    rows = run(profile)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return {
        "profile": profile_name,
        "profile_id": profile,
        "classes": rows,
        "verdict_counts": counts,
        "failing": sum(counts.get(v, 0) for v in FAILING_VERDICTS),
        "silent_key_loss": silent_key_loss(profile),
        "jcs_conformance": jcs_conformance(profile),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--profile",
        choices=("v2", "v1", "both"),
        default="both",
        help="which canonicalization profile to census (default: both, so the "
        "migration is visible)",
    )
    args = parser.parse_args(argv)

    names = ["v1", "v2"] if args.profile == "both" else [args.profile]
    results = {name: census(name) for name in names}

    payload = {
        "provenance": "census",
        "confidence_intervals": None,
        "confidence_interval_non_claim": (
            "This corpus is the whole population and its size is an authoring "
            "decision. A confidence interval would describe sampling "
            "variability that does not exist here."
        ),
        "profiles": results,
        "duplicate_key_parse": duplicate_key_parse(),
        "gated_profile": "v2",
        "gate_non_claim": (
            "A clean census establishes that the declared classes behave as "
            "declared. It does not establish that the kernel has no other "
            "members: the corpus is curated, and its coverage is an authoring "
            "decision, not a measurement."
        ),
    }

    # Only the default profile gates. v1 is retained read-only for historical
    # receipts and is EXPECTED to fail -- that is why it was replaced.
    failed = results["v2"]["failing"] if "v2" in results else 0

    if args.json:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1 if failed else 0

    print("cc-framework canonicalization kernel census")
    print("provenance: census - exact counts only, no confidence intervals\n")

    for name in names:
        result = results[name]
        gated = " (GATED)" if name == "v2" else " (read-only, historical)"
        print(f"profile {name}{gated}  {result['profile_id']}")
        width = max(len(r["class"]) for r in result["classes"])
        for row in result["classes"]:
            flag = " <<<" if row["verdict"] in FAILING_VERDICTS else ""
            print(
                f"  {row['class']:<{width}}  intent={row['intent']:<10}"
                f"  observed={row['observed']:<13}  {row['verdict']}{flag}"
            )
        counts = result["verdict_counts"]
        print("  counts: " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))

        loss = result["silent_key_loss"]
        if loss["refused"]:
            print(f"  both-key-forms document: refused with {loss['error']}")
        elif loss["keys_out"] < loss["keys_in"]:
            print(
                f"  both-key-forms document: {loss['keys_in']} keys in -> "
                f"{loss['keys_out']} out, NO EXCEPTION  <<< silent key loss"
            )
        else:
            print(
                f"  both-key-forms document: {loss['keys_in']} keys in -> "
                f"{loss['keys_out']} out, no loss"
            )

        nonconf = [r for r in result["jcs_conformance"] if not r["conformant"]]
        print(
            f"  RFC 8785 number forms: "
            f"{len(result['jcs_conformance']) - len(nonconf)}"
            f"/{len(result['jcs_conformance'])} conformant"
        )
        for row in nonconf:
            print(
                f"    diverges {row['value']:<12} emitted={row['emitted']!s:<26}"
                f" jcs={row['jcs_expects']!s}"
            )
        print("")

    dup = payload["duplicate_key_parse"]
    print("duplicate keys on the parse side:")
    print(f"  {dup['raw']} -> json.loads gives {dup['permissive_parse']}")
    print(f"  strict_json_loads refuses it: {dup['strict_json_loads_refuses']}")
    print("")

    if failed:
        print(
            f"FAIL: profile v2 has {failed} class(es) at {' or '.join(sorted(FAILING_VERDICTS))}."
        )
        return 1
    print("PASS: profile v2 has no unintended-kernel or rejection-asymmetry class.")
    print("\n  " + payload["gate_non_claim"].replace("(.{70}) ", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
