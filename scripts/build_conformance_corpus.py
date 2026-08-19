#!/usr/bin/env python3
"""Generate the ``cc-kernel-v1`` conformance corpus.

The corpus is the normative statement of what a CC composition kernel must
compute. It is language-agnostic JSON: an implementation in any language either
reproduces every expected value within tolerance, or it is not a CC kernel.

Expected values are produced here by :mod:`cc.compose`, and every ``accept``
case is independently cross-checked against the finite-atom LP in
:mod:`cc.kernel.sensitivity` before it is written. A case that the closed form
and the LP disagree on is a defect in this repository and the build fails
rather than pinning the disagreement into the corpus.

Usage::

    python scripts/build_conformance_corpus.py            # write the corpus
    python scripts/build_conformance_corpus.py --check    # verify it is current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from cc.compose import CountermonotoneUndefinedError, compose_bounds
from cc.kernel.sensitivity import AssumptionSet, LinearQuery, identified_region
from cc.reporting.canonical import canonical_json_bytes

CORPUS_VERSION = "cc-kernel-v1"
ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "conformance" / CORPUS_VERSION

#: Absolute tolerance an implementation is allowed on every numeric field.
#: Chosen at 1e-12, far above the 3.3e-16 worst disagreement measured between
#: the closed form and the LP, and far below any difference that would change a
#: reported bound.
TOLERANCE = 1.0e-12

# --- accept cases: (id, marginals, event, dependence, description) -----------

_ACCEPT: list[tuple[str, dict[str, float], str, str, str]] = [
    (
        "and-two-classical",
        {"A": 0.1, "B": 0.2},
        "all",
        "unconstrained",
        "Classical two-event conjunction. Lower is max(0, sum-1)=0; upper is min(p)=0.1.",
    ),
    (
        "and-two-overlap-forced",
        {"A": 0.9, "B": 0.8},
        "all",
        "unconstrained",
        "Marginals summing above 1 force a positive lower bound: 0.9+0.8-1 = 0.7.",
    ),
    (
        "and-two-equal-tie",
        {"A": 0.25, "B": 0.25},
        "all",
        "unconstrained",
        "Tied minima: no single event binds, so binding_event is null.",
    ),
    (
        "and-two-extreme-asymmetry",
        {"A": 0.01, "B": 0.99},
        "all",
        "unconstrained",
        "A narrow interval from extreme marginals. Narrowness is not evidence of validity.",
    ),
    (
        "or-two-classical",
        {"A": 0.1, "B": 0.2},
        "any",
        "unconstrained",
        "Classical two-event union. Lower is max(p)=0.2; upper is min(1, sum)=0.3.",
    ),
    (
        "or-two-clipped",
        {"A": 0.7, "B": 0.8},
        "any",
        "unconstrained",
        "Union upper bound clipped at 1 because the marginals sum above it.",
    ),
    (
        "and-four-vinctura-small-chapter",
        {"SELF_REPORTED": 0.30, "STALE": 0.01, "IMMUTABLE": 0.01, "SEPARATION": 0.40},
        "all",
        "unconstrained",
        "Four deterministic controls, assumed evasion rates. Reproduces a published "
        "external result: lower 0, upper 0.01, independence 1.2e-5, factor 833.",
    ),
    (
        "and-four-vinctura-optimistic",
        {"SELF_REPORTED": 0.05, "STALE": 0.01, "IMMUTABLE": 0.01, "SEPARATION": 0.10},
        "all",
        "unconstrained",
        "Same four controls, optimistic assumed rates. Tied minima at 0.01.",
    ),
    (
        "and-four-vinctura-pessimistic",
        {"SELF_REPORTED": 0.50, "STALE": 0.10, "IMMUTABLE": 0.05, "SEPARATION": 0.60},
        "all",
        "unconstrained",
        "Same four controls, pessimistic assumed rates. IMMUTABLE binds at 0.05.",
    ),
    (
        "and-single-event",
        {"A": 0.42},
        "all",
        "unconstrained",
        "Degenerate single-event conjunction. The interval collapses to the marginal.",
    ),
    (
        "or-single-event",
        {"A": 0.42},
        "any",
        "unconstrained",
        "Degenerate single-event union. The interval collapses to the marginal.",
    ),
    (
        "and-zero-marginal",
        {"A": 0.0, "B": 0.5},
        "all",
        "unconstrained",
        "A zero marginal pins the conjunction to zero regardless of the other event.",
    ),
    (
        "and-unit-marginals",
        {"A": 1.0, "B": 1.0},
        "all",
        "unconstrained",
        "Certain events. The conjunction is certain.",
    ),
    (
        "and-two-independent",
        {"A": 0.3, "B": 0.4},
        "all",
        "independent",
        "Independence returns a degenerate interval at the product. A baseline, not an answer.",
    ),
    (
        "or-two-independent",
        {"A": 0.3, "B": 0.4},
        "any",
        "independent",
        "Independent union: 1 - (1-0.3)(1-0.4) = 0.58.",
    ),
    (
        "and-two-comonotone",
        {"A": 0.3, "B": 0.4},
        "all",
        "comonotone",
        "The upper Frechet corner as a degenerate interval.",
    ),
    (
        "and-two-countermonotone",
        {"A": 0.3, "B": 0.4},
        "all",
        "countermonotone",
        "Defined only for exactly two events: max(0, 0.3+0.4-1) = 0.",
    ),
    (
        "and-two-countermonotone-positive",
        {"A": 0.7, "B": 0.8},
        "all",
        "countermonotone",
        "Two-event countermonotone with a positive floor: 0.7+0.8-1 = 0.5.",
    ),
]

# The correlation cliff, as a family: m events all at p=0.1. Independence
# predicts 0.1**m; the sharp upper bound stays at 0.1 for every m. This is the
# repository's headline result, pinned so no implementation can lose it.
for _m in (2, 3, 4, 5, 8, 10):
    _ACCEPT.append(
        (
            f"cliff-homogeneous-m{_m}",
            {f"G{i}": 0.1 for i in range(_m)},
            "all",
            "unconstrained",
            f"Correlation cliff at m={_m}: independence predicts 1e-{_m}, the sharp "
            f"upper bound stays 0.1. Stacking buys nothing under adversarial dependence.",
        )
    )

# --- reject cases ------------------------------------------------------------

_REJECT: list[tuple[str, dict[str, Any], str, str]] = [
    (
        "reject-countermonotone-three-events",
        {
            "marginals": {"A": 0.3, "B": 0.4, "C": 0.5},
            "event": "all",
            "dependence": "countermonotone",
        },
        "countermonotone_undefined",
        "Countermonotonicity is strictly bivariate. The FH lower bound is not a "
        "copula in dimension >= 3, though it stays pointwise sharp. An "
        "implementation that returns a number here is wrong.",
    ),
    (
        "reject-countermonotone-one-event",
        {
            "marginals": {"A": 0.3},
            "event": "all",
            "dependence": "countermonotone",
        },
        "countermonotone_undefined",
        "Countermonotonicity is a relation between two events; with fewer than two "
        "there is nothing to be countermonotone with. Found by differential fuzz "
        "(seed 1): the Python reference raised IndexError and the Node "
        "implementation silently returned NaN. Both were wrong, differently.",
    ),
    (
        "reject-marginal-above-one",
        {"marginals": {"A": 1.5, "B": 0.2}, "event": "all", "dependence": "unconstrained"},
        "marginal_out_of_range",
        "A marginal outside [0, 1] is not a probability.",
    ),
    (
        "reject-marginal-negative",
        {"marginals": {"A": -0.1, "B": 0.2}, "event": "all", "dependence": "unconstrained"},
        "marginal_out_of_range",
        "A negative marginal is not a probability.",
    ),
    (
        "reject-empty-marginals",
        {"marginals": {}, "event": "all", "dependence": "unconstrained"},
        "no_events",
        "A composition over zero events has no meaning.",
    ),
    (
        "reject-unknown-event-kind",
        {"marginals": {"A": 0.1}, "event": "xor", "dependence": "unconstrained"},
        "unknown_event_kind",
        "Only conjunction and union are defined by this corpus.",
    ),
    (
        "reject-unknown-dependence",
        {"marginals": {"A": 0.1, "B": 0.2}, "event": "all", "dependence": "sorta-dependent"},
        "unknown_dependence",
        "An unrecognized dependence assumption must fail closed, not default.",
    ),
    (
        "reject-non-finite-marginal",
        {"marginals": {"A": float("nan"), "B": 0.2}, "event": "all", "dependence": "unconstrained"},
        "marginal_not_finite",
        "NaN is not a probability. Note this case is expressed as a string in JSON "
        "because JSON has no NaN literal; see the spec.",
    ),
]


def _lp_crosscheck(marginals: dict[str, float], event: str) -> tuple[float, float]:
    """Recompute the interval with the finite-atom LP."""
    names = tuple(marginals)
    assumptions = AssumptionSet.empty(names)
    for name in names:
        assumptions = assumptions.with_marginal_interval(name, marginals[name], marginals[name])
    query = (
        LinearQuery.intersection(names, names)
        if event == "all"
        else LinearQuery.union(names, names)
    )
    result = identified_region(query, assumptions)
    return result.lower_bound, result.upper_bound


def build_accept() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, marginals, event, dependence, description in _ACCEPT:
        bounds = compose_bounds(marginals, event=event, dependence=dependence)
        payload = bounds.to_json()

        # Every unconstrained case must agree with the LP. A disagreement is a
        # defect here, not a corpus decision.
        if dependence == "unconstrained":
            lp_lower, lp_upper = _lp_crosscheck(marginals, event)
            drift = max(abs(lp_lower - bounds.lower), abs(lp_upper - bounds.upper))
            if drift > TOLERANCE:
                raise SystemExit(
                    f"REFUSING to pin {case_id}: closed form and LP disagree by "
                    f"{drift:.3e}, above the {TOLERANCE:.0e} tolerance. Fix the "
                    f"kernel before regenerating the corpus."
                )

        cases.append(
            {
                "id": case_id,
                "description": description,
                "input": {
                    "marginals": dict(marginals),
                    "event": event,
                    "dependence": dependence,
                },
                "expect": {
                    "lower": payload["lower"],
                    "upper": payload["upper"],
                    "width": payload["width"],
                    "independence_point": payload["independence_point"],
                    "understatement_factor": payload["understatement_factor"],
                    "binding_event": payload["binding_event"],
                },
            }
        )
    return cases


def build_reject() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, raw_input, reason, description in _REJECT:
        marginals = raw_input["marginals"]
        # Confirm the implementation actually refuses, so the corpus cannot
        # claim a refusal that does not happen.
        try:
            compose_bounds(
                marginals,
                event=raw_input["event"],
                dependence=raw_input["dependence"],
            )
        except (CountermonotoneUndefinedError, ValueError, TypeError):
            pass
        else:
            raise SystemExit(f"REFUSING to pin {case_id}: cc.compose accepted it.")

        serializable = dict(raw_input)
        serializable["marginals"] = {
            k: ("NaN" if isinstance(v, float) and v != v else v) for k, v in marginals.items()
        }
        cases.append(
            {
                "id": case_id,
                "description": description,
                "input": serializable,
                "expect_refusal": reason,
            }
        )
    return cases


def build() -> dict[str, dict[str, Any]]:
    accept = build_accept()
    reject = build_reject()
    composition = {
        "corpus": CORPUS_VERSION,
        "kind": "accept",
        "tolerance": TOLERANCE,
        "case_count": len(accept),
        "cases": accept,
    }
    adversarial = {
        "corpus": CORPUS_VERSION,
        "kind": "reject",
        "case_count": len(reject),
        "cases": reject,
    }
    return {"composition": composition, "adversarial": adversarial}


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    (CORPUS_DIR / "cases").mkdir(exist_ok=True)
    digests: dict[str, str] = {}
    for name, payload in files.items():
        path = CORPUS_DIR / "cases" / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digests[f"cases/{name}.json"] = _digest(payload)

    manifest = {
        "corpus": CORPUS_VERSION,
        "tolerance": TOLERANCE,
        "files": digests,
        "total_cases": sum(p["case_count"] for p in files.values()),
        "non_claims": [
            "Passing this corpus establishes that an implementation computes "
            "the same intervals as the reference on these cases. It does not "
            "establish that either implementation is correct: two "
            "implementations can share a misreading.",
            "The corpus is a curated census, not a sample. Its size is an "
            "authoring decision, it carries no coverage claim, and no "
            "confidence interval may be attached to a pass rate over it.",
            "Passing establishes nothing about safety, calibration, threshold "
            "choice, or whether the supplied marginals mean anything.",
        ],
    }
    (CORPUS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def check(files: dict[str, dict[str, Any]]) -> int:
    manifest_path = CORPUS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("corpus manifest missing; run without --check to generate", file=sys.stderr)
        return 1
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {f"cases/{name}.json": _digest(payload) for name, payload in files.items()}
    if committed.get("files") != expected:
        print("corpus is stale; regenerate with:", file=sys.stderr)
        print("  python scripts/build_conformance_corpus.py", file=sys.stderr)
        for key, digest in expected.items():
            was = committed.get("files", {}).get(key)
            if was != digest:
                print(f"  {key}: {was} -> {digest}", file=sys.stderr)
        return 1
    print(f"corpus current: {committed['total_cases']} cases, {len(expected)} files")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify the committed corpus is current"
    )
    args = parser.parse_args(argv)

    files = build()
    if args.check:
        return check(files)

    manifest = write(files)
    print(f"wrote {manifest['total_cases']} cases to {CORPUS_DIR.relative_to(ROOT)}")
    for name, digest in sorted(manifest["files"].items()):
        print(f"  {name}  {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
