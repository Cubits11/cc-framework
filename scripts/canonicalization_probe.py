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

from cc.reporting.canonical import canonical_json_bytes

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
        note="Sound in CPython; a JS JSON.parse collapses both before any "
        "verifier code runs. See F-05.",
    ),
    PathologyClass(
        name="int-vs-float-same-value",
        intent="distinct",
        left={"n": 1},
        right={"n": 1.0},
        note="JSON has one number type; RFC 8785 would unify these.",
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
        note="Same mathematical value, different Python types. RFC 8785 emits 1e+30 for both.",
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
        name="large-document-single-byte",
        intent="distinct",
        left={"pad": "x" * 65536, "tail": "a"},
        right={"pad": "x" * 65536, "tail": "b"},
        note="Positive control: catches a digest computed over a prefix.",
    ),
)


def _canonical(doc: dict[str, Any]) -> bytes | None:
    """Canonical bytes, or None when the document is refused."""
    try:
        return canonical_json_bytes(doc)
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


def run() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cls in CLASSES:
        left = _canonical(cls.left)
        right = _canonical(cls.right)
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


def silent_key_loss() -> dict[str, Any]:
    """F-03: a single object carrying both key forms loses one, silently."""
    both = {KEY_NFC: 1, KEY_NFD: 2}
    try:
        out = canonical_json_bytes(both)
    except Exception as exc:
        return {"keys_in": len(both), "refused": True, "error": type(exc).__name__}
    return {
        "keys_in": len(both),
        "keys_out": len(json.loads(out)),
        "refused": False,
        "output": out.decode("utf-8"),
    }


def duplicate_key_parse() -> dict[str, Any]:
    """F-07: duplicate keys are accepted last-wins on the parse side."""
    raw = '{"amount":1,"amount":2}'
    parsed = json.loads(raw)
    return {
        "raw": raw,
        "parsed": parsed,
        "collides_with_survivor": canonical_json_bytes(parsed)
        == canonical_json_bytes({"amount": 2}),
    }


# RFC 8785 (JCS) reference serializations for the number forms probed. Sourced
# from the JCS number-to-string rules, not from a runtime.
JCS_EXPECTED: tuple[tuple[Any, str], ...] = (
    (1e30, "1e+30"),
    (10**30, "1e+30"),
    (1.0, "1"),
    (-0.0, "0"),
    (1e-7, "1e-7"),
    (100.0, "100"),
)


def jcs_conformance() -> list[dict[str, Any]]:
    """F-04: how the emitted number form compares with RFC 8785."""
    rows: list[dict[str, Any]] = []
    for value, expected in JCS_EXPECTED:
        emitted = canonical_json_bytes({"n": value}).decode("utf-8")[5:-1]
        rows.append(
            {
                "value": repr(value),
                "cc_emits": emitted,
                "jcs_emits": expected,
                "conformant": emitted == expected,
            }
        )
    return rows


FAILING_VERDICTS = {"unintended-kernel", "rejection-asymmetry"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    rows = run()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    payload = {
        "provenance": "census",
        "confidence_intervals": None,
        "confidence_interval_non_claim": (
            "This corpus is the whole population and its size is an authoring "
            "decision. A confidence interval would describe sampling "
            "variability that does not exist here."
        ),
        "classes": rows,
        "verdict_counts": counts,
        "silent_key_loss": silent_key_loss(),
        "duplicate_key_parse": duplicate_key_parse(),
        "jcs_conformance": jcs_conformance(),
    }

    failing = sum(counts.get(v, 0) for v in FAILING_VERDICTS)

    if args.json:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1 if failing else 0

    print("cc-framework canonicalization kernel census")
    print("provenance: census - exact counts only, no confidence intervals\n")
    width = max(len(r["class"]) for r in rows)
    for row in rows:
        flag = " <<<" if row["verdict"] in FAILING_VERDICTS else ""
        print(
            f"  {row['class']:<{width}}  intent={row['intent']:<10}"
            f"  observed={row['observed']:<13}  {row['verdict']}{flag}"
        )

    print("\nverdict counts:")
    for verdict in sorted(counts):
        print(f"  {verdict:<22} {counts[verdict]}")

    loss = payload["silent_key_loss"]
    print("\nF-03 silent key loss (single object carrying both key forms):")
    if loss["refused"]:
        print(f"  refused with {loss['error']} - fixed")
    else:
        print(f"  {loss['keys_in']} keys in -> {loss['keys_out']} key(s) out, no exception")
        print(f"  canonical bytes: {loss['output']}")

    dup = payload["duplicate_key_parse"]
    print("\nF-07 duplicate keys on the parse side:")
    print(f"  {dup['raw']} -> {dup['parsed']}")
    print(f"  canonically identical to the survivor alone: {dup['collides_with_survivor']}")

    print("\nF-04 RFC 8785 (JCS) number-form conformance:")
    for row in payload["jcs_conformance"]:
        mark = "ok" if row["conformant"] else "DIVERGES"
        print(f"  {row['value']:<10} cc={row['cc_emits']:<34} jcs={row['jcs_emits']:<8} {mark}")
    nonconformant = sum(1 for r in payload["jcs_conformance"] if not r["conformant"])
    print(f"  {nonconformant} of {len(payload['jcs_conformance'])} forms diverge")

    if failing:
        print(
            f"\nFAIL: {failing} class(es) at {' or '.join(sorted(FAILING_VERDICTS))}."
            "\nSee docs/upgrade/FINDINGS_REGISTER.md F-03 through F-07."
        )
        return 1

    print("\nPASS: no unintended-kernel or rejection-asymmetry classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
