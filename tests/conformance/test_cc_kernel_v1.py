"""The ``cc-kernel-v1`` conformance corpus, enforced.

Three checks, in increasing strength:

1. The committed corpus is current -- it has not drifted from the kernel.
2. The Python reference passes every case in it.
3. An independent Node implementation, written from the specification rather
   than from this source, passes it too, and reproduces an external consumer's
   published numbers.

Check 3 is skipped when ``node`` is absent, and the skip is loud: a green run
without it establishes strictly less.

**What passing establishes.** That the implementations compute the same values
on these cases. **What it does not establish.** That any of them is correct:
the Python and Node sides were authored in the same project and can share a
misreading of the specification.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cc.compose import CountermonotoneUndefinedError, compose_bounds

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "conformance" / "cc-kernel-v1"
VERIFIER = ROOT / "verifiers" / "node" / "cc_compose_verify.mjs"

MANIFEST = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
TOLERANCE = MANIFEST["tolerance"]

ACCEPT = json.loads((CORPUS / "cases" / "composition.json").read_text(encoding="utf-8"))
REJECT = json.loads((CORPUS / "cases" / "adversarial.json").read_text(encoding="utf-8"))

_requires_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is not installed; cross-implementation agreement is NOT established",
)


# --- 1. the corpus is current -------------------------------------------------


def test_committed_corpus_is_current():
    """A stale corpus pins yesterday's kernel and silently stops testing today's."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_conformance_corpus.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_manifest_counts_match_the_case_files():
    assert ACCEPT["case_count"] == len(ACCEPT["cases"])
    assert REJECT["case_count"] == len(REJECT["cases"])
    assert MANIFEST["total_cases"] == len(ACCEPT["cases"]) + len(REJECT["cases"])


def test_manifest_carries_non_claims():
    assert MANIFEST["non_claims"]
    assert any("does not establish" in claim for claim in MANIFEST["non_claims"])


def test_case_ids_are_unique():
    ids = [c["id"] for c in ACCEPT["cases"]] + [c["id"] for c in REJECT["cases"]]
    assert len(ids) == len(set(ids))


# --- 2. the Python reference passes -------------------------------------------


@pytest.mark.parametrize("case", ACCEPT["cases"], ids=lambda c: c["id"])
def test_python_reference_reproduces_accept_case(case):
    bounds = compose_bounds(
        case["input"]["marginals"],
        event=case["input"]["event"],
        dependence=case["input"]["dependence"],
    )
    got = bounds.to_json()
    for field, want in case["expect"].items():
        have = got[field]
        if field == "binding_event" or want is None or have is None:
            assert have == want, f"{case['id']}.{field}"
        else:
            assert have == pytest.approx(want, abs=TOLERANCE), f"{case['id']}.{field}"


@pytest.mark.parametrize("case", REJECT["cases"], ids=lambda c: c["id"])
def test_python_reference_refuses_reject_case(case):
    marginals = {k: (math.nan if v == "NaN" else v) for k, v in case["input"]["marginals"].items()}
    with pytest.raises((CountermonotoneUndefinedError, ValueError, TypeError)):
        compose_bounds(
            marginals,
            event=case["input"]["event"],
            dependence=case["input"]["dependence"],
        )


# --- 3. cross-implementation agreement ----------------------------------------


@_requires_node
def test_independent_node_implementation_agrees():
    """The check that moves cross-implementation agreement off zero."""
    proc = subprocess.run(
        ["node", str(VERIFIER), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    report = json.loads(proc.stdout)
    assert report["accept"]["failed"] == 0, report["accept"]["failures"]
    assert report["reject"]["failed"] == 0, report["reject"]["failures"]
    assert report["accept"]["total"] == len(ACCEPT["cases"])
    assert report["reject"]["total"] == len(REJECT["cases"])
    assert proc.returncode == 0


@_requires_node
def test_node_implementation_reproduces_the_external_oracle():
    """The one check here whose implementation is independent of this code.

    The oracle numbers were published by a separate project that implemented
    these bounds for its own purposes, without reference to this corpus. That
    makes the *implementation* independent; it does not make the *author*
    independent, because the two projects share one. This catches divergent
    arithmetic between two codebases. It cannot catch a shared misreading of the
    specification, which is what author independence would buy and what this
    repository still lacks entirely.
    """
    proc = subprocess.run(
        ["node", str(VERIFIER), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    report = json.loads(proc.stdout)
    assert report["external_oracle"]["total"] >= 1
    assert report["external_oracle"]["failed"] == 0, report["external_oracle"]["failures"]


@_requires_node
def test_agreement_report_states_what_it_does_not_establish():
    """Agreement that does not name its limits invites being over-read."""
    proc = subprocess.run(
        ["node", str(VERIFIER), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    report = json.loads(proc.stdout)
    assert "does not establish that either is correct" in report["agreement_non_claim"]
    assert report["provenance"] == "census"


@_requires_node
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_differential_fuzz_finds_no_disagreement(seed):
    """Randomized cases explore what the curated corpus does not.

    Seeds are fixed here so CI is deterministic. The harness accepts any seed;
    a wider unseeded sweep belongs in the scheduled empirical lane, not in the
    per-commit gate.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "differential_compose.py"),
            "--cases",
            "400",
            "--seed",
            str(seed),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    report = json.loads(proc.stdout)
    assert report["disagreements"] == 0, report["detail"]
    assert report["cases"] == 400
    # A sweep that refused everything would agree trivially.
    assert report["accepted"] > 0
    assert report["refused"] > 0
    # Coverage and blind spots must be stated, not implied.
    assert report["generator_coverage"]
    assert report["generator_blind_spots"]
    assert report["provenance"] == "sampled"
