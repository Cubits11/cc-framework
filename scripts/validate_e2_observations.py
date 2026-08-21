#!/usr/bin/env python
"""Validate an E2 shared-item observation set against the frozen contract.

The E2 preregistration is only meaningful if conformance is checkable rather
than asserted.  This script is that check.  It answers two separate questions:

1. Do the rows conform to ``schemas/cc.e2_observation_row.v1.json``?
2. Do the rows satisfy the *design* conditions that a schema cannot express --
   above all, that every guardrail was evaluated on the same items?

Violations block (exit 1).  Warnings never block, but they are the findings a
reader must see before believing any dependence estimate: post-hoc exclusions,
version drift, upstream moderation, declared shared dependencies, duplicated
items, and replicate disagreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema

DEFAULT_SCHEMA = Path("schemas/cc.e2_observation_row.v1.json")

# Missingness codes whose presence changes how a dependence estimate must be read.
INFORMATIVE_MISSINGNESS = frozenset(
    {"upstream_moderation", "provider_refusal", "unsupported_input", "excluded_post"}
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL observation file into a list of row mappings."""

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit(f"{path}:{number}: each line must be a JSON object")
        rows.append(parsed)
    return rows


def schema_violations(rows: list[dict[str, Any]], schema: dict[str, Any]) -> list[str]:
    """Return one message per schema-invalid row."""

    validator = jsonschema.Draft202012Validator(schema)
    problems: list[str] = []
    for index, row in enumerate(rows, start=1):
        for error in sorted(validator.iter_errors(row), key=lambda err: list(err.path)):
            location = "/".join(str(part) for part in error.path) or "<row>"
            problems.append(f"row {index}: {location}: {error.message}")
    return problems


def design_violations(rows: list[dict[str, Any]]) -> list[str]:
    """Return design-level breaches that a per-row schema cannot detect."""

    problems: list[str] = []

    # Layer B: shared-item pairing. Unpaired items entangle observed dependence
    # with item composition, which is the failure this contract exists to prevent.
    items_by_guardrail: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        items_by_guardrail[str(row.get("guardrail_id"))].add(str(row.get("item_id")))
    if len(items_by_guardrail) > 1:
        universe = set.union(*items_by_guardrail.values())
        for guardrail, items in sorted(items_by_guardrail.items()):
            missing = universe - items
            if missing:
                sample = ", ".join(sorted(missing)[:5])
                problems.append(
                    f"shared-item violation: guardrail {guardrail!r} has no row for "
                    f"{len(missing)} item(s) other guardrails cover (e.g. {sample}). "
                    "Every guardrail must be offered every item; absence must be "
                    "recorded as a row with a missingness_code, not omitted."
                )

    # Layer C: one frozen reduction h per study, or the binary event is undefined.
    normalizers = {
        str(row.get("normalizer_version"))
        for row in rows
        if row.get("normalizer_version") is not None
    }
    if len(normalizers) > 1:
        problems.append(
            f"normalizer_version is not frozen: found {sorted(normalizers)}. "
            "A single study must collapse raw outcomes under exactly one versioned h."
        )

    # Row identity must be unique, or derived pair tables are ill-defined.
    keys = Counter(
        (str(row.get("item_id")), str(row.get("guardrail_id")), row.get("replicate_id"))
        for row in rows
    )
    for key, count in sorted(keys.items()):
        if count > 1:
            problems.append(
                f"duplicate observation for item={key[0]!r} guardrail={key[1]!r} "
                f"replicate={key[2]}: {count} rows"
            )
    return problems


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the disclosures a reader needs before trusting an estimate."""

    guardrails = sorted({str(row.get("guardrail_id")) for row in rows})
    missingness: dict[str, Counter[str]] = {
        guardrail: Counter(
            str(row.get("missingness_code"))
            for row in rows
            if str(row.get("guardrail_id")) == guardrail
        )
        for guardrail in guardrails
    }

    versions: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        guardrail = str(row.get("guardrail_id"))
        versions[guardrail].add(str(row.get("guardrail_version")))
        group = row.get("shared_dependency_group")
        if group is not None:
            groups[str(group)].add(guardrail)

    # Same content under different ids is benchmark duplication, which inflates
    # apparent agreement between guardrails.
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hashes[str(row.get("item_hash"))].add(str(row.get("item_id")))

    # Disagreement across replicates of the same (item, guardrail) is guardrail
    # stochasticity, which no single-shot estimate represents.
    replicate_outcomes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        if row.get("normalized_outcome") is not None:
            key = (str(row.get("item_id")), str(row.get("guardrail_id")))
            replicate_outcomes[key].add(int(row["normalized_outcome"]))
    unstable = sorted(key for key, seen in replicate_outcomes.items() if len(seen) > 1)

    complete_items = _complete_case_items(rows, guardrails)
    all_items = {str(row.get("item_id")) for row in rows}

    return {
        "rows": len(rows),
        "guardrails": guardrails,
        "items": len(all_items),
        "complete_case_items": len(complete_items),
        "complete_case_fraction": (len(complete_items) / len(all_items)) if all_items else 0.0,
        "missingness_by_guardrail": {k: dict(v) for k, v in missingness.items()},
        "guardrail_versions": {k: sorted(v) for k, v in versions.items()},
        "shared_dependency_groups": {k: sorted(v) for k, v in groups.items() if len(v) > 1},
        "duplicate_item_content": {k: sorted(v) for k, v in hashes.items() if len(v) > 1},
        "replicate_unstable_pairs": [list(key) for key in unstable],
    }


def _complete_case_items(rows: list[dict[str, Any]], guardrails: list[str]) -> set[str]:
    """Items observed on every guardrail -- the naive analysis population."""

    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if str(row.get("missingness_code")) == "observed":
            observed[str(row.get("item_id"))].add(str(row.get("guardrail_id")))
    needed = set(guardrails)
    return {item for item, seen in observed.items() if needed <= seen}


def warnings_from(report: dict[str, Any]) -> list[str]:
    """Turn audit findings into the disclosures a reader must not miss."""

    notes: list[str] = []
    for guardrail, counts in sorted(report["missingness_by_guardrail"].items()):
        for code, count in sorted(counts.items()):
            if code in INFORMATIVE_MISSINGNESS:
                notes.append(
                    f"{guardrail}: {count} row(s) with missingness_code {code!r}; "
                    "complete-case analysis may itself manufacture or suppress dependence."
                )
    for guardrail, seen in sorted(report["guardrail_versions"].items()):
        if len(seen) > 1:
            notes.append(
                f"{guardrail}: version drift within one run ({', '.join(seen)}); "
                "repeated runs may be experiments on different mechanisms."
            )
        if "unknown" in seen:
            notes.append(f"{guardrail}: version is 'unknown'; this limits any replication claim.")
    for group, members in sorted(report["shared_dependency_groups"].items()):
        notes.append(
            f"declared shared dependency {group!r} spans {', '.join(members)}; "
            "observed dependence may be mechanism-induced rather than a property of composition."
        )
    for digest, ids in sorted(report["duplicate_item_content"].items()):
        notes.append(f"duplicate item content {digest[:12]}... under ids {', '.join(ids)}.")
    if report["replicate_unstable_pairs"]:
        notes.append(
            f"{len(report['replicate_unstable_pairs'])} (item, guardrail) pair(s) disagree "
            "across replicates; the guardrail is stochastic."
        )
    fraction = float(report["complete_case_fraction"])
    if report["items"] and fraction < 1.0:
        notes.append(
            f"complete-case population is {report['complete_case_items']}/{report['items']} "
            f"items ({fraction:.1%}); the missingness sensitivity analysis is mandatory, not optional."
        )
    return notes


def main() -> int:
    """Validate rows, print an audit, and block on conformance violations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, help="JSONL file of E2 observation rows")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--json", action="store_true", help="Emit the audit report as JSON")
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    rows = load_rows(args.observations)
    if not rows:
        print("E2 conformance error: no observation rows", file=sys.stderr)
        return 1

    violations = schema_violations(rows, schema) + design_violations(rows)
    report = audit(rows)
    notes = warnings_from(report)

    if args.json:
        print(json.dumps({"report": report, "violations": violations, "warnings": notes}, indent=2))
    else:
        print(f"rows={report['rows']}  items={report['items']}  guardrails={report['guardrails']}")
        print(
            f"complete-case items: {report['complete_case_items']}/{report['items']} "
            f"({report['complete_case_fraction']:.1%})"
        )
        for note in notes:
            print(f"  warning: {note}")
        for violation in violations:
            print(f"  VIOLATION: {violation}", file=sys.stderr)

    if violations:
        print(f"\nE2 conformance FAILED with {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("\nE2 conformance OK (warnings are disclosures, not passes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
