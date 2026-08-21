#!/usr/bin/env python3
"""Emit evidence cards and the site manifest an Evidence Atlas consumes.

Reads the claim-boundary manifest, binds each claim to its artifacts by digest
and to the current source revision, and writes one evidence card per claim plus
a ``cc.site_evidence_manifest.v1`` bundle.

The default verdict is ``not-run``. A ``pass`` is reachable only with ``--run``,
which executes each claim's commands and records what actually happened. A card
cannot acquire a passing verdict by being written confidently.

Nothing here computes an aggregate. Counts are reported per label, because a
site reporting "7/8 passing" would be describing something nobody measured.

Usage::

    python scripts/build_evidence_cards.py                  # verdicts: not-run
    python scripts/build_evidence_cards.py --run            # execute commands
    python scripts/build_evidence_cards.py --check          # verify committed output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from cc.evidence_card import ArtifactRef, EvidenceCard, cards_to_site_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "claims" / "claim_boundary_manifest.v0.1.json"
OUT_DIR = ROOT / "evidence-cards"
SITE_MANIFEST = OUT_DIR / "site-evidence-manifest.v1.json"

#: Commands that are safe to execute under --run. Anything not matching is
#: recorded as unverifiable rather than being shelled out blindly: a card
#: generator that runs arbitrary strings from a data file is a code-execution
#: surface, not an evidence tool.
RUNNABLE_PREFIXES = ("pytest ", "python ", "make ")


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        )
        revision = out.stdout.strip()
    except OSError:
        revision = ""
    if not revision:
        return "unknown-revision"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    ).stdout.strip()
    # A dirty tree is recorded, not hidden: a card bound to "abc123" when the
    # working tree differs from abc123 is bound to the wrong thing.
    return f"{revision}-dirty" if dirty else revision


def _artifact(path_str: str) -> ArtifactRef:
    path = ROOT / path_str
    if not path.is_file():
        return ArtifactRef(path=path_str, missing=True)
    data = path.read_bytes()
    return ArtifactRef(
        path=path_str,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes_len=len(data),
    )


#: pytest exit code 1 means "tests ran and some failed" -- the one non-zero exit
#: this harness can confidently attribute to a check performing and not
#: succeeding. Codes 2-5 mean usage error, internal error, interruption, or no
#: tests collected: in each the check did not run, which is `unverifiable`, not
#: `fail`. Everything else is likewise unattributable.
PYTEST_TESTS_FAILED = 1


def _run_one(command: str, env: dict[str, str]) -> tuple[str, str]:
    """Run a single command. Returns (outcome, one-line record)."""
    if not command.startswith(RUNNABLE_PREFIXES):
        return (
            "unverifiable",
            f"{command} -> not executed: outside the runnable allowlist {RUNNABLE_PREFIXES}",
        )
    argv = shlex.split(command)
    is_pytest = argv[0] == "pytest"
    if is_pytest:
        argv = [sys.executable, "-m", *argv]
    elif argv[0] == "python":
        argv = [sys.executable, *argv[1:]]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=ROOT, env=env)
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    summary = (tail[-1] if tail else "<no output>")[:160]
    record = f"{command} -> exit {proc.returncode}: {summary}"
    if proc.returncode == 0:
        return "pass", record
    if is_pytest and proc.returncode == PYTEST_TESTS_FAILED:
        return "fail", record
    return "unverifiable", record + " | exit not attributable to a check running"


def _run_commands(commands: list[str]) -> tuple[str, str]:
    """Run every command and derive the card verdict conservatively.

    Each command is recorded with its own outcome, so a claim whose tests pass
    but whose `make` target cannot run here reports both facts rather than
    collapsing to one. The card verdict is the worst outcome present:

    * any `fail`         -> the card fails
    * else any `unverifiable` -> the card is unverifiable
    * else               -> the card passes

    The harness never guesses a `fail`. A non-zero exit becomes `fail` only
    when it can be attributed to a check running and not succeeding -- pytest
    exit code 1. Everything else is `unverifiable`, which is a different fact
    about the world.

    This is not pedantry. On the first run of this script `make test-kernel`
    exited non-zero because its dependency-install step could not reach the
    network, and the kernel tests never executed. Reporting that as `fail`
    would have published a failure nobody observed.
    """
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    outcomes: list[str] = []
    records: list[str] = []
    for command in commands:
        outcome, record = _run_one(command, env)
        outcomes.append(outcome)
        records.append(record)

    if "fail" in outcomes:
        verdict = "fail"
    elif "unverifiable" in outcomes:
        verdict = "unverifiable"
    else:
        verdict = "pass"

    tally = ", ".join(
        f"{o}={outcomes.count(o)}" for o in ("pass", "fail", "unverifiable") if outcomes.count(o)
    )
    return verdict, f"[{tally}] " + " | ".join(records)


def build(*, run: bool) -> tuple[list[EvidenceCard], dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    revision = _git_revision()
    cards: list[EvidenceCard] = []

    for claim in manifest["claims"]:
        commands = tuple(claim["supporting_tests_or_commands"])
        if run:
            verdict, detail = _run_commands(list(commands))
        else:
            verdict, detail = "not-run", None

        cards.append(
            EvidenceCard(
                card_id=claim["id"],
                claim=claim["claim_text"],
                maturity=claim["level"],
                source_revision=revision,
                command=commands,
                artifacts=tuple(_artifact(p) for p in claim["supporting_files"]),
                falsifier=claim["falsifier"],
                assumptions=tuple(claim["assumptions"]),
                non_claims=tuple(claim["non_claims"]),
                evidence_state=claim["evidence_state"],
                verdict=verdict,
                # Every card starts as a draft. Release is a human decision made
                # in the publication workflow, not a side effect of generation.
                publication_state="draft",
                result_detail=detail,
            )
        )

    note = "Generated from docs/claims/claim_boundary_manifest.v0.1.json. " + (
        "Verdicts reflect commands executed during generation."
        if run
        else "Verdicts are 'not-run': no command was executed during generation."
    )
    return cards, cards_to_site_manifest(cards, source_revision=revision, generated_note=note)


def _write(site: dict[str, Any], cards: list[EvidenceCard]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "cards").mkdir(exist_ok=True)
    for card in cards:
        path = OUT_DIR / "cards" / f"{card.card_id}.json"
        path.write_text(
            json.dumps(card.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    SITE_MANIFEST.write_text(
        json.dumps(site, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _stable(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that legitimately change between runs, for --check."""
    out = json.loads(json.dumps(payload))
    out.pop("source_revision", None)
    out.pop("generated_note", None)
    for card in out.get("cards", []):
        card.pop("source_revision", None)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute each claim's commands and record real verdicts. Results are "
        "host-specific, so the COMMITTED cards are always the not-run scaffold; "
        "regenerate without --run before committing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed cards match the manifest (ignoring revision)",
    )
    args = parser.parse_args(argv)

    cards, site = build(run=args.run)

    if args.check:
        if not SITE_MANIFEST.exists():
            print("site manifest missing; run without --check", file=sys.stderr)
            return 1
        committed = json.loads(SITE_MANIFEST.read_text(encoding="utf-8"))
        if _stable(committed) != _stable(site):
            print(
                "evidence cards are stale; regenerate with:\n"
                "  python scripts/build_evidence_cards.py",
                file=sys.stderr,
            )
            return 1
        print(f"evidence cards current: {site['card_count']} cards")
        return 0

    _write(site, cards)
    print(f"wrote {site['card_count']} evidence cards to {OUT_DIR.relative_to(ROOT)}")
    if args.run:
        print(
            "  NOTE: verdicts are specific to this host and toolchain. The "
            "committed\n        cards are the not-run scaffold -- regenerate "
            "without --run before committing."
        )
    for label, counts in site["counts_by_label"].items():
        shown = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        print(f"  {label:20s} {shown}")
    print("\n  Counts are per label and orthogonal. There is no aggregate score,")
    print("  and any surface computing one from this file is misusing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
