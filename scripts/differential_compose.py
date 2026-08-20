#!/usr/bin/env python3
"""Differential fuzz between the Python and Node composition kernels.

The conformance corpus is a curated census: it checks the cases someone thought
to write down. This harness checks the cases nobody thought of, by generating
randomized inputs -- including malformed ones -- and requiring both
implementations to agree on the answer *or* on the refusal.

The Node side never sees the Python answers. It receives inputs on stdin and
returns its own results, so it cannot converge on the reference.

Seeds are reported. A disagreement is reproducible with ``--seed``.

**What agreement establishes.** That two implementations of one specification
compute the same values over the sampled region. **What it does not
establish.** That either is correct -- both were authored in the same project
and a wrong specification yields two implementations wrong together. Nor does it
establish coverage: the generator's distribution is stated below and it has
blind spots.

Usage::

    python scripts/differential_compose.py --cases 2000
    python scripts/differential_compose.py --seed 7 --cases 500 --json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from cc.compose import CountermonotoneUndefinedError, compose_bounds

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "verifiers" / "node" / "cc_compose_verify.mjs"

TOLERANCE = 1.0e-12

EVENTS = ("all", "any", "and", "or", "AND", "OR", "intersection", "union")
DEPENDENCE = ("unconstrained", "independent", "comonotone", "countermonotone")

#: What the generator can produce. Stated before the numbers, because a
#: generator's coverage bounds what its agreement means.
GENERATOR_COVERAGE = (
    "Marginals drawn from a mixture: uniform on [0,1], values snapped to the "
    "boundaries 0 and 1, values within 1e-9 of a boundary, and near-tied "
    "values that exercise the binding_event tie rule. Event counts 1-8. All "
    "event-kind aliases and all four dependence assumptions. Roughly one case "
    "in six is deliberately malformed: out-of-range, non-finite, empty, or an "
    "unknown event/dependence value."
)

#: What it cannot produce. A generator's blind spots are part of its result.
GENERATOR_BLIND_SPOTS = (
    "Event counts above 8.",
    "Marginals that are not IEEE-754 doubles (no decimal or rational inputs).",
    "Non-string event names, and names differing only by Unicode normalization "
    "-- the canonicalization kernel's collapse is probed separately by "
    "scripts/canonicalization_probe.py.",
    "Side constraints beyond marginals: the corpus and this harness both cover "
    "the closed-form path only, not the constrained LP path.",
    "Adversarially chosen floating-point values selected to maximize "
    "divergence; the draw is random, not searched.",
)


def _draw_marginal(rng: random.Random) -> float:
    roll = rng.random()
    if roll < 0.08:
        return 0.0
    if roll < 0.16:
        return 1.0
    if roll < 0.24:
        return rng.choice([1e-9, 1.0 - 1e-9, 1e-15, 5e-324])
    return rng.random()


def _generate(rng: random.Random, index: int) -> dict[str, Any]:
    n = rng.randint(1, 8)
    names = [f"E{i}" for i in range(n)]
    values = [_draw_marginal(rng) for _ in range(n)]

    # Near-ties exercise the binding_event tie rule from SPEC section 4.3.
    if n >= 2 and rng.random() < 0.25:
        values[1] = values[0]

    case: dict[str, Any] = {
        "id": f"fuzz-{index}",
        "marginals": dict(zip(names, values, strict=True)),
        "event": rng.choice(EVENTS),
        "dependence": rng.choice(DEPENDENCE),
    }

    # Roughly one in six is malformed, so refusal agreement is exercised too.
    roll = rng.random()
    if roll < 0.04:
        case["marginals"][names[0]] = rng.choice([1.5, -0.1, 2.0, -1e-6])
    elif roll < 0.07:
        case["marginals"][names[0]] = "NaN"
    elif roll < 0.10:
        case["marginals"] = {}
    elif roll < 0.13:
        case["event"] = rng.choice(["xor", "nand", "", "ALL"])
    elif roll < 0.16:
        case["dependence"] = rng.choice(["mixed", "", "Independent"])
    return case


def _python_side(case: dict[str, Any]) -> dict[str, Any]:
    marginals = {k: (math.nan if v == "NaN" else v) for k, v in case["marginals"].items()}
    try:
        bounds = compose_bounds(marginals, event=case["event"], dependence=case["dependence"])
    except CountermonotoneUndefinedError:
        return {"ok": False, "refusal": "countermonotone_undefined"}
    except (ValueError, TypeError) as exc:
        return {"ok": False, "refusal": _classify(str(exc))}
    payload = bounds.to_json()
    return {"ok": True, **payload}


def _classify(message: str) -> str:
    """Map a Python message onto a SPEC section 5 refusal identifier."""
    lowered = message.lower()
    if "at least one event" in lowered or "mapping" in lowered:
        return "no_events"
    if "must be finite" in lowered:
        return "marginal_not_finite"
    if "must lie in [0, 1]" in lowered:
        return "marginal_out_of_range"
    if "unknown event kind" in lowered:
        return "unknown_event_kind"
    if "unknown dependence" in lowered:
        return "unknown_dependence"
    if "real number" in lowered:
        return "marginal_not_finite"
    return f"unclassified:{message[:60]}"


def _node_side(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node = shutil.which("node")
    if node is None:
        raise SystemExit("node is not on PATH; the differential harness needs it")
    proc = subprocess.run(
        [node, str(VERIFIER), "--batch"],
        input=json.dumps({"cases": cases}),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"node verifier failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout)["results"]


_COMPARED = (
    "lower",
    "upper",
    "width",
    "independence_point",
    "independence_regret",
    "understatement_factor",
    "binding_event",
)


def _compare(case: dict[str, Any], py: dict[str, Any], js: dict[str, Any]) -> list[dict[str, Any]]:
    if py["ok"] != js["ok"]:
        return [
            {
                "id": case["id"],
                "field": "<accept/refuse>",
                "python": "accepted" if py["ok"] else py["refusal"],
                "node": "accepted" if js["ok"] else js["refusal"],
                "input": case,
            }
        ]
    if not py["ok"]:
        if py["refusal"] != js["refusal"]:
            return [
                {
                    "id": case["id"],
                    "field": "<refusal reason>",
                    "python": py["refusal"],
                    "node": js["refusal"],
                    "input": case,
                }
            ]
        return []

    out: list[dict[str, Any]] = []
    for field in _COMPARED:
        a, b = py[field], js[field]
        if field == "binding_event" or a is None or b is None:
            if a != b:
                out.append(
                    {"id": case["id"], "field": field, "python": a, "node": b, "input": case}
                )
            continue
        if abs(a - b) > TOLERANCE:
            out.append(
                {
                    "id": case["id"],
                    "field": field,
                    "python": a,
                    "node": b,
                    "delta": abs(a - b),
                    "input": case,
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=2000, help="number of randomized cases")
    parser.add_argument("--seed", type=int, default=None, help="seed; random if omitted")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randrange(2**31)
    rng = random.Random(seed)
    cases = [_generate(rng, i) for i in range(args.cases)]

    py_results = [_python_side(c) for c in cases]
    js_results = _node_side(cases)

    disagreements: list[dict[str, Any]] = []
    for case, py, js in zip(cases, py_results, js_results, strict=True):
        disagreements.extend(_compare(case, py, js))

    accepted = sum(1 for r in py_results if r["ok"])
    report = {
        "seed": seed,
        "cases": len(cases),
        "accepted": accepted,
        "refused": len(cases) - accepted,
        "disagreements": len(disagreements),
        "tolerance": TOLERANCE,
        "generator_coverage": GENERATOR_COVERAGE,
        "generator_blind_spots": list(GENERATOR_BLIND_SPOTS),
        "provenance": "sampled",
        "non_claim": (
            "Agreement over a sampled region does not establish correctness. "
            "Both implementations were authored in the same project and a wrong "
            "specification produces two implementations that are wrong together. "
            "Coverage is bounded by the generator described above."
        ),
        "detail": disagreements[:20],
    }

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if disagreements else 0

    print("differential fuzz — Python vs Node composition kernel")
    print(f"  seed        {seed}   (reproduce with --seed {seed})")
    print(f"  cases       {len(cases)}  ({accepted} accepted, {len(cases) - accepted} refused)")
    print(f"  tolerance   {TOLERANCE:g}")
    print("  provenance  sampled")
    print("")
    if disagreements:
        for row in disagreements[:20]:
            print(f"  DISAGREE {json.dumps(row, default=str)[:220]}")
        if len(disagreements) > 20:
            print(f"  ... and {len(disagreements) - 20} more")
        print(f"\nFAIL: {len(disagreements)} disagreement(s).")
        return 1
    print("PASS: no disagreement.")
    print("\ngenerator coverage:")
    for line in GENERATOR_COVERAGE.split(". "):
        if line.strip():
            print(f"  {line.strip().rstrip('.')}.")
    print("\ngenerator blind spots:")
    for spot in GENERATOR_BLIND_SPOTS:
        print(f"  - {spot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
