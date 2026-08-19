#!/usr/bin/env python3
"""``cc-guard`` -- the inference guards, reachable without a Python API.

A downstream project asked, in writing, to route its density estimate through
this repository's post-selection refusal so that an optional-stopping error
would be *refused* rather than reported. It could not: the guard existed only
as an in-process Python call, and the caller was JavaScript.

This module is the answer. It exposes the guards two ways, because one is not
enough:

1. ``cc-guard check`` reads a JSON request on stdin and writes a verdict on
   stdout. Any language that can spawn a process can use it.
2. ``cc-guard table`` emits the decision rules as **pure data**, so a caller
   can enforce them with no Python process at all. A subprocess dependency is a
   weaker bridge than a table -- it fails in CI sandboxes, in browsers, and
   anywhere Python is not installed.

The guards refuse claims the supplied evidence cannot support. Refusal is the
product. A guard that can be talked into a confidence claim is not a guard.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["DECISION_TABLE", "check", "main"]

#: The decision rules, as data. This is the normative statement of the guard;
#: :func:`check` implements it and ``tests/unit/cli/test_guard.py`` asserts the
#: two agree case for case, so the table cannot drift from the code.
#:
#: Each rule names the condition, the verdict, whether a confidence claim
#: survives, and the reason. A caller reimplementing this in another language
#: needs nothing beyond this structure.
DECISION_TABLE: dict[str, Any] = {
    "version": "cc-guard-v1",
    "guards": [
        {
            "id": "post_selection_interval",
            "applies_to": "An interval around an estimate that was selected by "
            "a search -- a maximum, a best-fit, a scan over candidates, or any "
            "procedure with optional stopping.",
            "input_fields": ["provenance", "estimate", "ci_low", "ci_high"],
            "rules": [
                {
                    "when": {"provenance": "post-selection"},
                    "verdict": "discovery-only",
                    "confidence_claim_permitted": False,
                    "reason": "The estimate is the maximum over a search, so it "
                    "is biased upward and an interval around it does not attain "
                    "its nominal level for the true parameter. A discovery is a "
                    "hypothesis; only held-out data can certify it.",
                    "remedy": "Re-estimate on data not used for selection, then "
                    "resubmit with provenance='confirmatory'.",
                },
                {
                    "when": {"provenance": "confirmatory"},
                    "verdict": "confirmatory",
                    "confidence_claim_permitted": True,
                    "reason": "The interval was computed on data not used to "
                    "select the target, so its nominal level applies.",
                    "remedy": None,
                },
            ],
            "default": {
                "verdict": "refused",
                "confidence_claim_permitted": False,
                "reason": "Unknown provenance. A guard that defaults to "
                "permitting a claim is not a guard.",
                "remedy": "Declare provenance as 'confirmatory' or 'post-selection'.",
            },
        },
        {
            "id": "census_interval",
            "applies_to": "A proportion computed over a curated corpus rather "
            "than a random sample.",
            "input_fields": ["provenance", "n"],
            "rules": [
                {
                    "when": {"provenance": "census"},
                    "verdict": "no-interval",
                    "confidence_claim_permitted": False,
                    "reason": "A confidence interval describes sampling "
                    "variability under repeated random draws. A hand-authored "
                    "corpus has none: it is the whole population and its size is "
                    "an authoring decision.",
                    "remedy": "Report exact counts and the denominator.",
                },
                {
                    "when": {"provenance": "sampled", "n_below": 30},
                    "verdict": "no-interval",
                    "confidence_claim_permitted": False,
                    "reason": "Below n=30 a normal-approximation interval is not "
                    "trustworthy, and a Wilson interval is wide enough that "
                    "quoting it invites misreading.",
                    "remedy": "Report exact counts, or collect more samples.",
                },
                {
                    "when": {"provenance": "sampled"},
                    "verdict": "interval-permitted",
                    "confidence_claim_permitted": True,
                    "reason": "A genuine sample of adequate size supports a proportion interval.",
                    "remedy": None,
                },
            ],
            "default": {
                "verdict": "refused",
                "confidence_claim_permitted": False,
                "reason": "Unknown provenance. Declare whether the denominator "
                "is a census or a sample.",
                "remedy": "Declare provenance as 'census' or 'sampled'.",
            },
        },
    ],
    "non_claims": [
        "A permitted verdict means this guard found no reason to refuse. It "
        "does not mean the estimate is correct, the model is right, or the "
        "measurement was well designed.",
        "These guards check the provenance a caller declares. They cannot "
        "detect a caller who declares 'confirmatory' for a post-selection "
        "interval.",
    ],
}

#: The minimum sample size below which no proportion interval is permitted.
#: Matches the ``n_below`` threshold in the census guard above.
MIN_N_FOR_PROPORTION_INTERVAL = 30


def _post_selection(request: dict[str, Any]) -> dict[str, Any]:
    provenance = request.get("provenance")
    if provenance == "post-selection":
        return {
            "guard": "post_selection_interval",
            "verdict": "discovery-only",
            "confidence_claim": None,
            "confidence_claim_permitted": False,
            "reason": "The estimate is the maximum over a search, so it is "
            "biased upward and an interval around it does not attain its "
            "nominal level for the true parameter. A discovery is a hypothesis; "
            "only held-out data can certify it.",
            "remedy": "Re-estimate on data not used for selection, then "
            "resubmit with provenance='confirmatory'.",
        }
    if provenance == "confirmatory":
        return {
            "guard": "post_selection_interval",
            "verdict": "confirmatory",
            "confidence_claim": request.get("ci_low") is not None
            and request.get("ci_high") is not None,
            "confidence_claim_permitted": True,
            "reason": "The interval was computed on data not used to select the "
            "target, so its nominal level applies.",
            "remedy": None,
        }
    return {
        "guard": "post_selection_interval",
        "verdict": "refused",
        "confidence_claim": None,
        "confidence_claim_permitted": False,
        "reason": f"Unknown provenance {provenance!r}. A guard that defaults to "
        "permitting a claim is not a guard.",
        "remedy": "Declare provenance as 'confirmatory' or 'post-selection'.",
    }


def _census(request: dict[str, Any]) -> dict[str, Any]:
    provenance = request.get("provenance")
    if provenance == "census":
        return {
            "guard": "census_interval",
            "verdict": "no-interval",
            "confidence_claim_permitted": False,
            "reason": "A confidence interval describes sampling variability "
            "under repeated random draws. A hand-authored corpus has none: it "
            "is the whole population and its size is an authoring decision.",
            "remedy": "Report exact counts and the denominator.",
        }
    if provenance == "sampled":
        n = request.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return {
                "guard": "census_interval",
                "verdict": "refused",
                "confidence_claim_permitted": False,
                "reason": f"A sampled proportion needs a non-negative integer "
                f"denominator; got {n!r}. No proportion without a denominator.",
                "remedy": "Supply 'n' as the number of draws.",
            }
        if n < MIN_N_FOR_PROPORTION_INTERVAL:
            return {
                "guard": "census_interval",
                "verdict": "no-interval",
                "confidence_claim_permitted": False,
                "reason": f"n={n} is below the minimum of "
                f"{MIN_N_FOR_PROPORTION_INTERVAL}. Below that a "
                "normal-approximation interval is not trustworthy, and a Wilson "
                "interval is wide enough that quoting it invites misreading.",
                "remedy": "Report exact counts, or collect more samples.",
            }
        return {
            "guard": "census_interval",
            "verdict": "interval-permitted",
            "confidence_claim_permitted": True,
            "reason": "A genuine sample of adequate size supports a proportion interval.",
            "remedy": None,
        }
    return {
        "guard": "census_interval",
        "verdict": "refused",
        "confidence_claim_permitted": False,
        "reason": f"Unknown provenance {provenance!r}. Declare whether the "
        "denominator is a census or a sample.",
        "remedy": "Declare provenance as 'census' or 'sampled'.",
    }


_GUARDS = {
    "post_selection_interval": _post_selection,
    "census_interval": _census,
}


def check(request: dict[str, Any]) -> dict[str, Any]:
    """Apply the named guard to a request and return its verdict.

    Args:
        request: Must carry ``guard`` naming one of
            ``post_selection_interval`` or ``census_interval``, plus that
            guard's declared ``input_fields``.

    Returns:
        A JSON-native verdict. ``confidence_claim_permitted`` is the field a
        caller acts on; a ``False`` there means the caller must not attach a
        confidence statement to the number, whatever else it does with it.
    """
    guard = request.get("guard")
    handler = _GUARDS.get(guard)
    if handler is None:
        return {
            "guard": guard,
            "verdict": "refused",
            "confidence_claim_permitted": False,
            "reason": f"Unknown guard {guard!r}. Known guards: {', '.join(sorted(_GUARDS))}.",
            "remedy": "Name a guard from 'cc-guard table'.",
        }
    return handler(request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cc-guard",
        description="Inference guards that refuse claims the evidence cannot support.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser("check", help="Apply a guard to a JSON request read from stdin.")
    check_parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit 1 when the verdict forbids a confidence claim, so a shell "
        "caller can gate on it without parsing JSON.",
    )

    sub.add_parser("table", help="Emit the decision rules as pure data.")

    args = parser.parse_args(argv)

    if args.command == "table":
        json.dump(DECISION_TABLE, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    raw = sys.stdin.read()
    try:
        request = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        json.dump(
            {
                "verdict": "refused",
                "confidence_claim_permitted": False,
                "reason": f"Request is not valid JSON: {exc}",
                "remedy": "Send a JSON object on stdin.",
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 2
    if not isinstance(request, dict):
        json.dump(
            {
                "verdict": "refused",
                "confidence_claim_permitted": False,
                "reason": "Request must be a JSON object.",
                "remedy": "Send a JSON object on stdin.",
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 2

    verdict = check(request)
    json.dump(verdict, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.strict_exit and not verdict["confidence_claim_permitted"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
