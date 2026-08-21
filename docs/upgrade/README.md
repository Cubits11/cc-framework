# Epistemic Upgrade — cc-framework

A deep excavation of `Cubits11/cc-framework` at commit `3e22c39`, measured
against the standard set by the flagship sibling repository
[`PSUCyberSecurityLab/ghost-ark`](https://github.com/PSUCyberSecurityLab/ghost-ark),
and a plan to close the gap.

> **Status: baseline measurements, plan, and delivery records.** The measurements
> are real and reproducible at their recorded revision. W3 and W5 are delivered;
> the remaining workstreams are proposals. Nothing here claims cc-framework is
> safe, correct, or production-ready, and nothing here relaxes a boundary in
> [`NON_CLAIMS.md`](../research/NON_CLAIMS.md).

---

## Read in this order

| # | Document | What it is |
|---|---|---|
| 1 | [BASELINE_MEASUREMENTS.md](BASELINE_MEASUREMENTS.md) | Every number, with the command that produced it. Read first — everything else rests on it. |
| 2 | [FINDINGS_REGISTER.md](FINDINGS_REGISTER.md) | 20 findings, 5 at severity S1, each with reproduction and remedy. |
| 3 | [COMMITTEE_SCORECARD.md](COMMITTEE_SCORECARD.md) | 12 dimensions scored, each naming what would raise it and what would lower it. |
| 4 | [VERIFICATION_LADDER.md](VERIFICATION_LADDER.md) | The tiers this repository should have, and what each does *not* establish. |
| 5 | [DOWNSTREAM_CONTRACTS.md](DOWNSTREAM_CONTRACTS.md) | What ghost-ark, vinctura, and b2b-spatial need. Contracts C1–C7. |
| 6 | [EPISTEMIC_UPGRADE_PLAN.md](EPISTEMIC_UPGRADE_PLAN.md) | Ten workstreams, acceptance gates, sequencing. |

---

## The short version

**The mathematics is exact and nobody can use it.**

The LP recovers Fréchet–Hoeffding to 1.11e-16 worst case. It reproduces the
correlation cliff in twenty seconds: eighteen guardrails at p = 0.1 each,
composed as a conjunction, bound to `[0, 0.1]` where independence predicts
1e-18. That result is the project's thesis and it works.

Three separate projects reimplemented the calculus rather than depend on this
repository, and one wrote down why: the most discoverable composition entry point
imposes an ROC/detector ontology on events that are deterministic predicates. A
consumer read the source, found the mathematics right and the API wrong, and
wrote 214 lines of JavaScript instead.

So the scorecard reads:

| | At excavation | Now |
|---|---:|---:|
| Mathematical correctness | 8.9 | 8.9 |
| Claim discipline — prose | 8.8 | 8.8 |
| Consumability as a library | 2.6 | **6.8** |
| Claim discipline — enforcement | 2.1 † | **4.0** † |
| Cross-implementation agreement | 0.0 | **6.5** |

† Corrected, not improved — see [F-12](FINDINGS_REGISTER.md#f-12).

Three failures, stated once:

1. **Reachability.** Correct and unreachable — wrong ontology, Python-only, no
   conformance corpus. Everybody rewrites it. **Largely closed by
   [W5](EPISTEMIC_UPGRADE_PLAN.md#w5-delivery-record).**
2. **Enforcement.** Prose 8.8, enforcement 4.0. Strict typing declared over 90
   files, enforced on 7. No forbidden-phrase scanner anywhere. The claim ledger
   itself *is* enforced — an earlier draft said otherwise and was wrong.
3. **Unexamined foundations (baseline; W3 delivered).** The baseline
   canonicalization kernel silently merged Unicode-distinct keys, diverged from
   RFC 8785, and had never been attacked. W3 shipped `cc.canonical.v2` and gates
   its declared census in CI. That is repair evidence, not proof that the kernel
   has no other flaws.

---

## What has been built

[W5](EPISTEMIC_UPGRADE_PLAN.md#w5-delivery-record) is delivered. Reproduce it:

```bash
make test-compose     # the ROC-free surface, 34 tests
make conformance      # corpus current + an independent Node implementation agrees
make differential     # 4,000 randomized cases through both implementations
make acceptance       # an external consumer's PUBLISHED numbers, from this library
```

The acceptance gate is the one that matters. It is not "the API exists" — it is
a real consumer's published four-control result reproduced from `cc.compose`:
interval `[0, 0.01]`, independence baseline `1.2e-5`, understatement factor
`833×`, all three of their scenarios, and their sensitivity finding. Their 214
lines of JavaScript could be deleted.

The differential fuzzer found a real bug on its first run — `countermonotone`
with one event, where the Python raised `IndexError` and the Node silently
returned `NaN`. Both wrong, differently; the curated corpus had not thought to
ask.

**The honest limit:** both implementations were authored in the same project, so
this is a differential-testing instrument, not an independent replication. The
one genuinely non-same-author check is a single external oracle on a single
scenario family.

---

## Verify the delivered canonicalization repair (W3)

The historical canonicalization defects —
[F-03](FINDINGS_REGISTER.md#f-03) through [F-07](FINDINGS_REGISTER.md#f-07) —
remain visible in the read-only v1 census. The gated v2 profile is verified in
one command:

```bash
PYTHONPATH=src python scripts/canonicalization_probe.py
```

The probe exits non-zero if v2 carries an `unintended-kernel` or
`rejection-asymmetry` verdict. A green result establishes only that its declared
cases behave as declared; the corpus is curated and does not prove completeness.

Historical v1 behavior, preserved only so legacy receipts remain verifiable:

```python
>>> canonical_json_bytes({"é": 1, "é": 2})   # U+00E9 key, then U+0065 U+0301 key
b'{"\xc3\xa9":2}'
```

Two distinct input keys. One output key. No exception. Under v2, the same input
keeps both keys, and the profile conforms to RFC 8785 for the census's tested
number forms.

---

## What this excavation did not do

Absence of a finding is not evidence of absence.

- No fuzzing, no mutation testing, no dependency audit in this pass.
- `redteam/dependence_search.py` — 709 statements at 65.6% coverage, performing
  the search whose post-selection bias `cliff.py` refuses to certify — was **not
  examined**. It is the most likely home of a subtle statistical defect.
- No external reviewer was involved. Every finding here was produced by the same
  kind of process that produced the code.
- One finding was **wrong on first publication**. F-12 claimed the
  claim-boundary manifest was "validated by nothing"; the original grep covered
  `.github/` and `Makefile` but not `tests/`, where the validator is in fact
  called. It is corrected in place, with the correction visible rather than
  quietly edited away, and the enforcement score was revised from 2.1 to 4.0.
