# Findings Register — cc-framework Epistemic Excavation

> **Status: measured findings.** Every finding below was reproduced by a command
> against commit `3e22c39`. Each carries the reproduction, the impact, the
> proposed remedy, and the workstream that owns it.
>
> **Non-claim.** This register is not a security audit and not a completeness
> claim. It records what one excavation pass found. The absence of a finding is
> not evidence of absence — see [Not searched](#what-was-not-searched).

Severity scale:

| | Meaning |
|---|---|
| **S1** | An artifact this repository signs or publishes can be wrong or misleading, silently. |
| **S2** | A gate the repository claims to enforce is not enforced. |
| **S3** | A consumer cannot use the repository for its stated purpose. |
| **S4** | Hygiene, drift, or duplication that will become S1–S3 if left. |

---

## Index

| ID | Severity | Finding | Owner |
|---|---|---|---|
| [F-01](#f-01) | S2 | The declared `[test]` extra cannot run the test suite | W0 |
| [F-02](#f-02) | S2 | `strict = true` mypy is declared for `cc`, enforced on 7 of 90 files | W1 |
| [F-03](#f-03) | **S1** | Canonicalization silently merges Unicode-distinct keys | W3 |
| [F-04](#f-04) | **S1** | Canonical form is not RFC 8785; five of six number forms diverge | W3 |
| [F-05](#f-05) | **S1** | Cross-language receipt divergence above 2^53 is undetectable | W3, W5 |
| [F-06](#f-06) | S2 | `-0.0` and `0.0` produce different receipts for identical numbers | W3 |
| [F-07](#f-07) | S2 | Duplicate JSON keys are accepted last-wins on the parse side | W3 |
| [F-08](#f-08) | **S3** | A real consumer read the composition API and rejected it | W5 |
| [F-09](#f-09) | **S3** | The kernel calculus is independently reimplemented three times | W5 |
| [F-10](#f-10) | **S3** | Vinctura's requested cross-language guard surface does not exist | W5 |
| [F-11](#f-11) | S3 | Ghost-Ark's binding ingest rule has no enforcing code here | W6 |
| [F-12](#f-12) | S2 | The claim-boundary manifest is validated by nothing | W2 |
| [F-13](#f-13) | S2 | No coverage gate; the report CLI is at 0% | W4 |
| [F-14](#f-14) | S2 | `core/stats.py` is 38% covered across 756 statements | W4 |
| [F-15](#f-15) | S4 | Two divergent theorem ledgers | W2 |
| [F-16](#f-16) | S4 | 17.8 MB of render binaries tracked in git | W8 |
| [F-17](#f-17) | S2 | Experiment and performance lanes have never run in CI | W7 |
| [F-18](#f-18) | S3 | A sibling repository demoted the core contribution to "engineering" | W9 |
| [F-19](#f-19) | S4 | Schema surface is two files with a non-resolvable `$id` | W6 |
| [F-20](#f-20) | S2 | A regression test shells out to run the whole unit suite | W0 |

---

## F-01

**The declared `[test]` extra cannot run the test suite.** — S2, owner W0

`pyproject.toml`'s `[project.optional-dependencies].test` omits `build`,
`setuptools`, and `wheel`. `tests/unit/packaging/test_wheel_boundary.py`
subprocesses `python -m build --no-isolation`, which needs all three.

```bash
uv venv .venv && uv pip install -e '.[test]' && pytest
# 1 failed, 685 passed, 9 skipped, 3 errors
# E   No module named build
# then, after installing build:
# E   BackendUnavailable: Cannot import 'setuptools.build_meta'
```

**Impact.** A stranger following the documented install path gets a red suite and
has no way to tell a real regression from a packaging gap. CI passes only
because `[dev]` happens to pull `build`, so the defect is invisible to the
people who would fix it. This is the single cheapest fix in the register and it
is the first thing an external reviewer hits.

**Remedy.** Add `build`, `setuptools`, `wheel` to `[test]`. Add a CI job that
installs `.[test]` *specifically* — not `.[dev]` — and runs the suite, so the
extra that strangers are told to use is the extra that is tested.

---

## F-02

**Strict typing is declared for the package and enforced on 7 files.** — S2, owner W1

`pyproject.toml` declares `packages = ["cc"]` with `strict = true`. CI and
pre-commit both override this with an explicit 7-file list.

```bash
mypy                       # at the declared scope
# Found 279 errors in 47 files (checked 101 source files)
```

**Impact.** The configuration asserts a discipline the repository does not have.
Any reader who checks `pyproject.toml` to see how strict the project is will be
misled by 279 errors' worth. This is claim inflation *in the build
configuration* — the same defect class the project's own documents exist to
prevent, expressed in TOML instead of prose.

**Remedy.** Do not delete `strict = true`, and do not silence 279 errors in one
pass. Ratchet: introduce `docs/upgrade/typing-ratchet.json` recording the
current per-module error count, add a CI job that fails if any module's count
*increases*, and burn the list down module by module, kernel first. The declared
scope becomes true incrementally and cannot regress.

---

## F-03

**Canonicalization silently merges Unicode-distinct keys.** — **S1**, owner W3

`_normalize_json_value` applies `unicodedata.normalize("NFC", key)` to every
mapping key and writes results into a fresh `dict`. When two distinct source
keys share an NFC form, the second overwrites the first.

```python
>>> from cc.reporting.canonical import canonical_json_bytes
>>> canonical_json_bytes({"é": 1, "é": 2})   # U+00E9 key, then U+0065 U+0301 key
b'{"\xc3\xa9":2}'
```

Two keys in. One key out. **No exception.**

The committed census (`scripts/canonicalization_probe.py`) finds **two**
`unintended-kernel` classes, not one. Widening the corpus from the hand-probed
six classes to eleven surfaced `nested-unicode-key-collision` — the same
collision one level down, inside a nested object:

```python
>>> canonical_json_bytes({"outer": {"é": 1}}) == canonical_json_bytes({"outer": {"é": 1}})
True
```

This matters on its own: a guard that inspected only top-level keys would pass
the flat class while leaving the defect fully exploitable. It also reproduces
Ghost-Ark's E1 lesson exactly — *widening the alphabet found more defects* — and
it is the reason the remedy below is a corpus rather than a patch.

**Impact.** This is the most serious finding in the register. The receipt hash is
computed over the *normalized* document, so:

1. A report can lose a field between construction and hashing, and the receipt
   will faithfully attest to the truncated document.
2. Two semantically different payloads can be constructed with the same receipt
   hash — a collision in the provenance kernel, reachable with no cryptography
   and no privileged access.
3. Ghost-Ark's E1 census classifies exactly this pattern as an *unintended kernel
   member*. cc-framework has one, in the module every signed artifact routes
   through, and has never looked.

The irony is load-bearing: `b2b-spatial-intelligence-engine` ships a research
module titled `01-canonicalization-collapse.html`. The sibling repositories study
this defect class. This repository has it.

**Remedy.** Normalization must **detect** collision rather than resolve it. After
normalizing, compare key-set cardinality; if it shrank, raise
`CanonicalJSONError` naming both colliding keys. Fail closed. Add the pair to a
committed adversarial corpus (W3) so the fix cannot silently regress.

---

## F-04

**The canonical form is not RFC 8785, and diverges on five of six number
forms.** — **S1**, owner W3

| Value | cc-framework | RFC 8785 (JCS) |
|---|---|---|
| `10**30` | `1000000000000000000000000000000` | `1e+30` |
| `1.0` | `1.0` | `1` |
| `-0.0` | `-0.0` | `0` |
| `1e-7` | `1e-07` | `1e-7` |
| `100.0` | `100.0` | `100` |
| `1e30` | `1e+30` | `1e+30` |

The implementation is `json.dumps(sort_keys=True, separators=(",",":"))`, which
is a Python convention, not an interoperable standard.

**Impact.** Any non-Python verifier that implements JCS — the obvious choice for
an independent verifier, and the family Ghost-Ark surveyed in
`CANONICALIZATION_LAYER_SURVEY.md` — computes a different digest for the same
report. Independent verification is impossible not because of a bug but because
the two sides never agreed on what the bytes are.

**Remedy.** Choose deliberately and document the choice with its consequences:
either adopt RFC 8785 number serialization, or declare `cc.canonical.v1` as an
explicitly non-JCS profile with a written rationale and a conformance corpus.
Either is defensible. Silence is not — and the choice must be made **before** an
independent verifier is written, not after.

---

## F-05

**Cross-language receipt divergence above 2^53 is undetectable here.** — **S1**, owner W3, W5

```python
>>> canonical_json_bytes({"n": 2**53 + 1})
b'{"n":9007199254740993}'
```

Python's arbitrary-precision integers preserve this. A JavaScript or TypeScript
verifier's `JSON.parse` collapses both `2**53+1` and `2**53+2` to
`9007199254740992` before any verifier code runs — the collapse happens inside
the parser, which is Ghost-Ark's E1 corollary C1 exactly: *the kernel is set by
the parser, and auditing the canonicalizer alone cannot find it.*

**Impact.** Ghost-Ark holds a TypeScript reimplementation of this calculus
(`packages/research-frontier/src/ccCorrelation.ts`). If a CC report carrying a
large integer count is verified there, the two sides disagree — and nothing in
either repository would notice, because there is no differential test spanning
them.

**Remedy.** Two parts, both required. (1) Constrain the schema: integers in
receipt-covered positions are bounded to the IEEE-754 safe range, or carried as
strings. (2) Build the differential harness (W5) so divergence is *caught*, not
argued about.

---

## F-06

**`-0.0` and `0.0` produce different receipts for the same number.** — S2, owner W3

```python
>>> canonical_json_bytes({"n": -0.0})
b'{"n":-0.0}'
```

`-0.0 == 0.0` is true in IEEE-754 and both denote zero. Two numerically identical
reports receive different receipt hashes.

**Impact.** Over-discrimination, in Ghost-Ark's E1 vocabulary. Less dangerous than
F-03 — it produces false *differences* rather than false *identities* — but it
breaks replay determinism whenever an LP solver returns a negative zero, which
`scipy.optimize.linprog` does routinely at a lower bound of zero. A replay that
should be byte-identical will not be.

**Remedy.** Normalize `-0.0` to `0.0` before serialization. Add both to the
corpus, and add the positive control that genuinely distinct near-zero values
stay distinct — a strict rule that rejects honest documents is not a fix, it is
a trade.

---

## F-07

**Duplicate JSON keys are accepted last-wins on the parse side.** — S2, owner W3

```python
>>> json.loads('{"amount":1,"amount":2}')
{'amount': 2}
```

Reports read from disk pass through `json.loads`, which silently keeps the last
value. `canonical_json_bytes` then attests to the survivor.

**Impact.** The same class as F-03 and reachable from any file the repository
reads. Ghost-Ark's E1 records `duplicate-key-last-wins` as an unintended kernel
member in four of five arms; the one sound arm was the one with a different
parser.

**Remedy.** Read receipt-covered JSON through a strict loader with an
`object_pairs_hook` that raises on repeated keys. Applies to the report reader,
the evidence-bundle reader, and the claim-envelope reader alike.

---

## F-08

**A real consumer read the composition API and rejected it.** — **S3**, owner W5

`vinctura/scripts/compose-bounds.js`, in its file header, records the decision
not to use this repository:

> The obvious move was to call `cc.core.composition_theory`, which implements FH
> bounds for guardrail composition. Reading it rather than its symbol list: it
> operates on **ROC POINT SETS** and bounds the **Youden J statistic**
> (J = TPR − FPR). That is a DETECTOR framing — it assumes each guardrail is a
> classifier with a threshold and an operating curve.
>
> Vinctura's controls are not classifiers. `SELF_REPORTED` refuses if and only if
> `loggedBy === memberId`. There is no threshold, no operating point, and no
> false-positive rate to trade against. Forcing a deterministic refusal rule into
> an ROC shape would produce numbers with the form of a measurement and none of
> the content.

The consumer then wrote 214 lines of JavaScript to do it themselves.

**Impact.** This is the most valuable single artifact the excavation found, because
it is a *rejected-adoption report written by someone who read the source*. It
names the defect precisely: the most discoverable composition entry point,
`cc.core.composition_theory`, imposes a detector ontology on events that have
none. Deterministic predicates — a refusal rule, a schema check, a signature
verification — are binary failure events with marginals and no ROC curve. They
are squarely inside the mathematics and outside the API.

Note also what the consumer *did* keep: "the FH inequality is applied directly to
the events. **That is the part that transfers**; the ROC machinery is not." The
consumer correctly identified the kernel and correctly identified the packaging
as the obstacle.

**Remedy.** W5 ships `cc.compose` — a marginals-in, bounds-out surface with no ROC
concept anywhere in its signature — and re-derives Vinctura's four-control result
from it as an acceptance test. The gate is not "the API exists"; the gate is
"Vinctura's own numbers reproduce from the library, and their 214 lines can be
deleted."

---

## F-09

**The kernel calculus is independently reimplemented three times.** — **S3**, owner W5

| Repository | File | Language | What it reimplements |
|---|---|---|---|
| cc-framework | `src/cc/kernel/` | Python | normative source |
| ghost-ark | `packages/research-frontier/src/ccCorrelation.ts` | TypeScript | FH pairwise bounds, Wilson intervals, phi |
| vinctura | `scripts/compose-bounds.js` (214 LOC) | JavaScript | FH n-ary conjunction bounds |

No cross-checking. No shared corpus. No agreement test. Three implementations of
one theorem, each trusted because it looks right.

**Impact.** Ghost-Ark treats *its own* cross-language verifier agreement as a
critical architectural invariant — "breaking independent verifier agreement is a
critical architectural event." Across the CC calculus, the same project has
three implementations and zero agreement tests. Any one of them can drift and
nothing detects it.

There is also an opportunity here that is larger than the defect. Ghost-Ark's E5
(cross-language verifier agreement) and E7 (differential fuzz) exist and work.
Pointing that machinery at the CC kernel costs far less than inventing it.

**Remedy.** W5 publishes `conformance/cc-kernel-v1/` — a language-agnostic corpus
of `(marginals, constraints, query) → [L, U]` cases with exact expected values,
plus an adversarial section (infeasible sets, degenerate marginals, boundary
values). Any implementation in any language either passes or is not a CC kernel.
cc-framework runs it in CI; the corpus is published for the others.

---

## F-10

**Vinctura's requested cross-language guard surface does not exist.** — **S3**, owner W5

`vinctura/docs/research/program/ultracode/UC-10-KERNEL-BRIDGE.md` §4.3:

> Optional stopping invalidates the density estimate entirely — and the sibling
> project `cc-framework` already refuses confidence claims on post-selection
> intervals at `src/cc/kernel/cliff.py:321`. **Use that.** Route the density
> estimate through the same provenance-tagged machinery so that a post-selection
> interval is refused rather than reported. Making one repository's guardrail
> catch another repository's error is the strongest possible demonstration that
> the guardrail is real.
>
> That last point is the single most elegant thing available in this program. Do it.

The refusal exists and is correct — `cliff_certificate(..., provenance="post-selection")`
returns `regime="discovery-only"` with no confidence claim. It is reachable only
from Python, in-process.

**Impact.** A named downstream consumer has a written, dated request to route
through a specific guard in this repository, and cannot, because there is no
callable surface across the language boundary. The most compelling demonstration
available to the whole program is blocked on packaging, not on science.

**Remedy.** W5 ships `cc-guard` — a stdin/stdout JSON subcommand exposing the
provenance-tagged guards, and the same logic as a pure-data decision table in the
conformance corpus so a JS caller can enforce it without a Python process at all.
Acceptance: Vinctura's G3 gate (`probe.post-selection-refused`) passes against
cc-framework's guard.

---

## F-11

**Ghost-Ark's binding ingest rule has no enforcing code here.** — S3, owner W6

`ghost-ark/docs/research/CLAIM_EVIDENCE_MATRIX.md` states as a binding rule:

> CC-Framework must not consume naked binary labels from Ghost-Ark. Binary
> variables must be tied to a discretization rule, threshold, comparator,
> calibration digest, scoring digest, validity window, and parent evidence
> lineage.

Ghost-Ark specifies the object (`ghost.discretization_rule_receipt.v1`), the
monotonic risk invariant, and eleven verification preconditions. cc-framework
has **no ingest module, no schema, and no test** for any of it. The rule binds a
repository that cannot honour it.

**Impact.** The bridge between the two repositories is currently prose in one of
them. Every guarantee the contract offers — that a `Z_i = 1` means what it says,
that the comparator matches score polarity, that the observation is inside the
rule's validity window — is unenforced at the point of consumption.

**Remedy.** W6 ships `cc.ingest.discretization`: a fail-closed reader for
`ghost.discretization_rule_receipt.v1` implementing all eleven preconditions,
with a negative corpus for each. A marginal that arrives without lineage is
refused, not defaulted.

---

## F-12

**The claim-boundary manifest is validated by nothing.** — S2, owner W2

`docs/claims/CLAIM_BOUNDARY_MANIFEST.md` and its JSON companion define C0–C5
claim levels and map every public claim to evidence, tests, files, and
non-claims. It is one of the best-designed artifacts in the repository.
`scripts/validate_claim_boundary_manifest.py` exists to check it.

```bash
grep -rn "validate_claim_boundary_manifest" .github/ Makefile
# (no match)
```

**Impact.** The manifest can drift from the code it describes — a renamed test, a
moved module, a deleted file — and nothing reports it. An unenforced truth table
degrades into a historical document, and the failure is silent: it still *looks*
authoritative.

**Remedy.** W2 wires the validator into CI and pre-commit, extends it to assert
that every named test path and source file exists, and adds a test that fails
when a claim row references a nonexistent witness.

---

## F-13

**No coverage gate; the report CLI is at 0%.** — S2, owner W4

Total coverage 69.91%. `src/cc/reporting/cli.py` — the `cc-report` console entry
point, 204 statements — is at **0.00%**. `src/cc/cli/manifest.py`, 68 statements,
also 0.00%. No CI job measures or enforces coverage.

**Impact.** `cc-report build-report` and `cc-report verify-claim-governance` are
the commands an external reviewer would actually run. They are the repository's
product surface and they are entirely unexercised by the suite. The verifier that
issues PASS verdicts on claim-governance packages has no test covering its own
CLI path.

**Remedy.** W4 adds CLI golden-output tests (build a report, verify it, verify a
tampered copy fails), then sets a coverage floor at the measured value and
ratchets it upward. A floor set below the current value is theatre; set it at
69.91% and raise it only with evidence.

---

## F-14

**`core/stats.py` is 38.16% covered across 756 statements.** — S2, owner W4

434 statements in the statistics module are never executed by the suite.

**Impact.** In a repository whose entire purpose is to be honest about
statistical claims, the statistics module is its least-tested large component.
Untested statistical code does not fail loudly — it returns a plausible number.
That is the specific failure mode this project exists to prevent, located inside
the project.

**Remedy.** W4 treats `core/stats.py` as the highest-priority coverage target,
with property-based tests (Hypothesis is already a declared dependency and
currently ungated) over interval coverage, monotonicity, and boundary behaviour
rather than example-based tests alone.

---

## F-15

**Two divergent theorem ledgers.** — S4, owner W2

`docs/theory/theorem_ledger.md` (10 KB, T1–T6 with proof status, implementation
witness, test witness, and non-claims per theorem) and
`docs/research/THEOREM_LEDGER.md` (3 KB). Different content, same name, no
cross-reference declaring which governs.

**Impact.** Ghost-Ark's matrix names its source of truth explicitly and states the
tie-break: "If the ladder and this matrix disagree, downgrade the claim." Two
ledgers with no such rule means a reader cannot tell which one binds, and an
author can satisfy whichever is convenient.

**Remedy.** `docs/theory/theorem_ledger.md` is the ledger. The other becomes a
pointer, or is deleted. Add a test asserting exactly one file matches
`*theorem*ledger*`.

---

## F-16

**17.8 MB of render binaries tracked in git.** — S4, owner W8

Seven Blender renders in `visual_identity/claim_observatory/renders/`, 2.2–2.8 MB
each. Four 636 KB JSONL checkpoints under `docs/archive/generated-checkpoints/`.
`.git` is 43 MB against a 54 MB working tree.

**Impact.** Clone cost for external reviewers, for no verification benefit. The
renders are brand assets, not evidence.

**Remedy.** Move to a release asset or LFS. Keep one small preview in-tree. Not
urgent; do it during a release, not mid-workstream.

---

## F-17

**Experiment and performance lanes have never run in CI.** — S2, owner W7

Nine tests skip behind `CC_RUN_EXPERIMENTS`, `CC_RUN_PERF`, or optional imports
(`guardrails`, `fastavro`, `protobuf`, `SQLAlchemy`). No workflow sets any of them.

**Impact.** The lanes that produce empirical numbers are the lanes that never run.
`tests/performance/test_adapter_perf.py` and
`tests/experiments/test_experiment_leak_metrics.py` are, operationally, dead code
that looks like coverage.

**Remedy.** W7 adds a scheduled workflow that sets both variables and installs the
optional extras, publishing results as artifacts. A lane that runs weekly and
reports honestly beats a lane that is skipped daily and looks green.

---

## F-18

**A sibling repository demoted the core contribution to "engineering."** — **S3**, owner W9

`b2b-spatial-intelligence-engine/docs/latent-research-program.md` runs a
kill-ledger over the program's research concepts. The row that matters:

> **Fréchet ceiling + reachability sharpening** — "Adversarial robustness ≠
> average-case robustness" — standard in ML security since 2014. **DEMOTED to
> engineering.** The mathematics is 1935; the sharpening is a known distinction.
> Survives as reporting practice, not as science.

**Impact.** This is the hardest single piece of feedback in the excavation, it
comes from inside the program, and it is **substantially correct**.
Fréchet–Hoeffding is 1935. Restating it for guardrails is not a contribution.
Any plan that positions cc-framework's value as "we compute FH bounds" is
answering a criticism that has already been made and sustained.

It is also *not* fatal, and the reason matters. What is not 1935:

- **Statistical inference for partial-identification bounds under estimated
  marginals** — a confidence band around `[L, U]` rather than a point interval,
  which is thin in the literature and is VISION Pillar I.
- **Measured co-failure dependence on real guardrail stacks** — nobody has
  published how badly independence lies in practice, which is Pillar V.
- **A conformance corpus that makes three independent implementations agree** —
  engineering, yes, but engineering nobody has done for this calculus.
- **Refusing to certify** — the post-selection guard at `cliff.py:321` is a
  design commitment most statistical software does not make.

**Remedy.** W9 rewrites the framing to concede the theorem and claim the
measurement. The repository's public position becomes: *the mathematics is
classical and we say so; the contribution is the estimation loop, the measured
atlas, the certificate, and the refusal.* Concede early, in the README, above the
fold. A criticism you state yourself cannot be used against you.

---

## F-19

**Schema surface is two files with a non-resolvable `$id`.** — S4, owner W6

`schemas/cc_report.schema.json` and `schemas/evidence/evidence-item.schema.json`.
Both `$id` values point at `https://cc-framework.local/...`. Version lives in
`title` ("CC Report v0.3.1"), not in the `$id`.

For comparison, Ghost-Ark carries 182 JSON files with versioned identifiers of the
form `ghost.discretization_rule_receipt.v1.json`.

**Impact.** A consumer cannot resolve a schema, cannot pin a version, and cannot
tell from an artifact which schema version produced it. Version-in-title means a
schema can change without its identifier changing.

**Remedy.** W6 adopts `cc.<object>.v<N>` identifiers, moves version into `$id`,
and adds a test asserting every emitted artifact names a resolvable, versioned
schema.

---

## F-20

**A regression test shells out to run the whole unit suite.** — S2, owner W0

`tests/regression/week2/test_week2_deliverables.py::test_unit_tests_pass`
subprocesses `pytest tests/unit -q` and asserts the return code is zero.

**Impact.** Every unit-test failure is reported twice — once truthfully, once as a
meaningless `assert 1 == 0` in an unrelated file. With `--maxfail=1` in
`addopts`, whichever fires first can mask the real one. This is how F-01
presented: the visible failure was a week-2 deliverables test, and the actual
cause was a missing `build` module three directories away.

**Remedy.** Delete the test. If the intent was "week 2 deliverables still exist,"
assert that directly — check the files and the entry points, not the exit code of
a nested test runner.

---

## What was not searched

Stated so that absence is not read as a null result:

- **No fuzzing.** No fuzz target exists for the report reader, the LP input path,
  or the canonicalizer.
- **No mutation testing.** Mutation score is unknown, not low.
- **No adversarial corpus.** There is no committed set of malformed reports that
  must be rejected. The negative tests that exist are hand-written per-feature.
- **No dependency audit in this pass.** `pip-audit` runs in `security.yml`; its
  current output was not collected here.
- **No review of `_legacy`, `exp/`, `theory/`, or the notebooks** beyond size.
- **No numerical audit of `redteam/dependence_search.py`** (709 statements,
  65.6% covered), which performs the search whose post-selection bias
  `cliff.py` refuses to certify. It is the most likely location of a subtle
  statistical defect and it was not examined in this pass.
