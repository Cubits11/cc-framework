# Baseline Measurements — cc-framework

> **Status: measured.** Every number in this document was produced by a command
> run against commit `3e22c39` on 2026-08-19, on the host described in
> [Host](#host). Numbers that are unflattering are reported anyway. Where a
> number was *not* measured, the row says so rather than estimating.
>
> **Non-claim.** These measurements describe this repository on one host at one
> commit. They are not a statement about correctness, safety, or the behaviour
> of any deployed system. A passing test suite is evidence of regression
> resistance, not of correctness.

Regenerate with `scripts/upgrade_baseline.py` (see [Reproduction](#reproduction)).

---

## Host

| Field | Value |
|---|---|
| Platform | Linux 6.18.5, x86-64 |
| Python | 3.11.15 (CPython) |
| NumPy / SciPy | 2.x / 1.17.1 |
| Environment | ephemeral container, `uv venv`, `uv pip install -e '.[test]'` |
| Commit | `3e22c39` |
| Date | 2026-08-19 |

A latency figure without a machine is not reproducible. Every timing below is
host-relative and must be re-measured before it is cited.

---

## 1. Size

| Metric | Value | Command |
|---|---:|---|
| Python files in `src/cc` | 90 | `find src -name '*.py' \| wc -l` |
| Lines in `src/cc` | 41,645 | `find src -name '*.py' -exec cat {} + \| wc -l` |
| Test files | 108 | `find tests -name '*.py' \| wc -l` |
| Markdown files (repo) | 136 | `find . -name '*.md' -not -path './.git/*' \| wc -l` |
| Markdown files (`docs/`) | 90 | `find docs -name '*.md' \| wc -l` |
| JSON schemas | 2 | `ls schemas/*.json schemas/evidence/*.json` |
| Working tree (excl. `.git`) | 54 MB | `du -sh --exclude=.git .` |
| `.git` | 43 MB | `du -sh .git` |

### Per-package line counts

| Package | Files | Lines |
|---|---:|---:|
| `src/cc/core` | 13 | 10,462 |
| `src/cc/evidence` | 11 | 9,946 |
| `src/cc/kernel` | 12 | 6,389 |
| `src/cc/analysis` | 9 | 4,384 |
| `src/cc/cartographer` | 9 | 2,896 |
| `src/cc/redteam` | 2 | 1,413 |
| `src/cc/reporting` | 4 | 1,343 |
| `src/cc/adapters` | 6 | 1,191 |
| `src/cc/evals` | 3 | 993 |
| `src/cc/guardrails` | 6 | 726 |
| `src/cc/exp` | 1 | 710 |
| `src/cc/enterprise` | 2 | 524 |
| `src/cc/io` | 2 | 204 |
| `src/cc/utils` | 6 | 236 |
| `src/cc/cli` | 2 | 117 |
| `src/cc/_legacy` | 1 | 89 |

`cc.core` and `cc.evidence` together are 20,408 lines — 49% of the source tree —
against a `cc.kernel` of 6,389 lines. The mathematical core the project exists
to provide is 15% of its own source.

---

## 2. Test suite

### Out of the box, from the declared test extra

```bash
uv venv .venv && uv pip install -e '.[test]' && pytest
```

| Result | Count |
|---|---:|
| Passed | 685 |
| **Failed** | **1** |
| **Errors** | **3** |
| Skipped | 9 |
| Wall time | 87 s |

Failures:

- `tests/unit/packaging/test_wheel_boundary.py` — 3 errors, `No module named build`
- `tests/regression/week2/test_week2_deliverables.py::test_unit_tests_pass` — cascaded

After installing `build`, a second, distinct failure appears:
`BackendUnavailable: Cannot import 'setuptools.build_meta'`.

**The `[test]` extra cannot run the test suite.** It omits `build`, `setuptools`,
and `wheel`, all of which `tests/unit/packaging/` requires. See finding
[F-01](FINDINGS_REGISTER.md#f-01).

### After adding the three undeclared dependencies

| Result | Count |
|---|---:|
| Passed | 689 |
| Failed | 0 |
| Skipped | 9 |
| Wall time | 90 s |

This is the green baseline the rest of the plan builds on.

### Skipped tests (9)

Skips are gated by environment variables or optional dependencies:
`CC_RUN_EXPERIMENTS`, `CC_RUN_PERF`, `guardrails`, `fastavro`, `protobuf`,
`SQLAlchemy`. No CI job sets any of them, so **no CI run has ever executed the
experiment or performance lanes.**

---

## 3. Coverage

`pytest --cov=src/cc --cov-report=term-missing`

| Metric | Value |
|---|---:|
| Statements | 15,149 |
| Missed | 3,904 |
| Branches | 4,762 |
| Partial branches | 1,070 |
| **Total coverage** | **69.91%** |

There is **no coverage gate in CI**. This number has never been enforced.

### Modules below 60%

| Module | Stmts | Cover | Note |
|---|---:|---:|---|
| `reporting/cli.py` | 204 | **0.00%** | the `cc-report` entry point — the product surface |
| `cli/manifest.py` | 68 | **0.00%** | |
| `core/audit_runner.py` | 199 | 17.45% | |
| `guardrails/semantic_filter.py` | 38 | 18.75% | |
| `guardrails/toy_threshold.py` | 42 | 21.43% | |
| `enterprise/aws_reference.py` | 143 | 23.87% | |
| `guardrails/regex_filters.py` | 138 | 24.18% | |
| `analysis/reporting.py` | 153 | 26.42% | |
| **`core/stats.py`** | **756** | **38.16%** | **the statistics engine; 434 statements unexercised** |
| `evals/run_bench.py` | 189 | 49.39% | |
| `cartographer/bounds.py` | 272 | 52.99% | |
| `utils/artifacts.py` | 50 | 53.70% | |
| `analysis/cc_estimation.py` | 118 | 56.08% | |
| `core/guardrail_api.py` | 95 | 57.48% | |
| `core/registry.py` | 79 | 57.89% | |
| `core/metrics.py` | 244 | 59.15% | |

The two largest uncovered surfaces are the **CLI that emits reports** (0%) and
the **statistics module** (38% of 756 statements). Both are on the path from
evidence to a published number.

### Modules above 90%

`kernel/frechet_classes.py` 98.87%, `evidence/permission_compiler.py` 95.28%,
`reporting/canonical.py` 91.89%, `core/manifest.py` 90.74%,
`evidence/role_ontology.py` 90.53%.

The kernel is well covered. The plumbing around it is not.

---

## 4. Type checking

`pyproject.toml` declares:

```toml
[tool.mypy]
packages = ["cc"]
strict = true
```

Running mypy **at its own declared scope**:

```
Found 279 errors in 47 files (checked 101 source files)
```

CI runs mypy on an explicit list of **7 files**:

```
src/cc/adapters/base.py  src/cc/cartographer/audit.py
src/cc/cartographer/bounds.py  src/cc/cartographer/intervals.py
src/cc/io/storage.py  src/cc/utils/artifacts.py  src/cc/utils/timing.py
```

`.pre-commit-config.yaml` uses the same 7-file list.

The repository declares strict typing across the `cc` package and enforces it on
7.8% of the files. See finding [F-02](FINDINGS_REGISTER.md#f-02).

---

## 5. Kernel numerics

### Fréchet–Hoeffding recovery

`identified_region` versus the closed form `[max(0, pA+pB-1), min(pA, pB)]`:

| `(pA, pB)` | LP interval | Closed form | Abs. error |
|---|---|---|---:|
| (0.1, 0.2) | [0, 0.1] | [0, 0.1] | 0 |
| (0.5, 0.5) | [0, 0.5] | [0, 0.5] | 0 |
| (0.9, 0.8) | [0.7, 0.8] | [0.7, 0.8] | 1.11e-16 |
| (0.01, 0.99) | [0, 0.01] | [0, 0.01] | 0 |
| (1e-6, 1e-6) | [0, 1e-6] | [0, 1e-6] | 0 |
| (0.999999, 0.999999) | [0.999998, 0.999999] | same | 0 |

Worst absolute error across the six configurations: **1.11e-16** — one unit in
the last place of a float64. **The LP kernel is numerically exact on this
family.** This is the strongest single result in the repository and it is
currently unadvertised.

Coverage caveat: six marginal configurations is a hand-chosen census, not a
sample. It carries no confidence interval and does not establish exactness
outside these points. Property-based testing over the marginal simplex is
workstream [W4](EPISTEMIC_UPGRADE_PLAN.md#w4).

### Scaling: atoms = 2^m

`m` guardrails, all marginals fixed at 0.1, query = intersection of all:

| m | atoms | wall time | interval |
|---:|---:|---:|---|
| 2 | 4 | 0.003 s | [0, 0.1] |
| 6 | 64 | 0.005 s | [0, 0.1] |
| 10 | 1,024 | 0.039 s | [0, 0.1] |
| 12 | 4,096 | 0.146 s | [0, 0.1] |
| 14 | 16,384 | 0.766 s | [0, 0.1] |
| 15 | 32,768 | 1.738 s | [0, 0.1] |
| 16 | 65,536 | 3.642 s | [0, 0.1] |
| 17 | 131,072 | 8.962 s | [0, 0.1] |
| 18 | 262,144 | 19.978 s | [0, 0.1] |

Single measurement per point, no repetition, so **no dispersion is reported and
none should be inferred**. Repeated-measurement timing with p50 and IQR is
workstream [W7](EPISTEMIC_UPGRADE_PLAN.md#w7).

Practical ceiling for the exact LP on this host: **m ≈ 18**, roughly doubling
per added guardrail. This is the measured basis for the relaxation hierarchy in
VISION Pillar VI — not an assumption.

### The correlation cliff, reproduced

Eighteen guardrails, each failing at p = 0.1, composed as a conjunction:

- independence would predict `0.1^18 = 1e-18`
- the sharp identified interval is **[0, 0.1]**

Under adversarial dependence, **eighteen stacked filters bound no better than
one**. The upper bound is `min_i p_i`, independent of `m`. This is the
project's thesis and it reproduces in 20 seconds on a laptop-class host.

---

## 6. Canonicalization kernel

Probing `cc.reporting.canonical.canonical_json_bytes` in the style of
Ghost-Ark's E1 provenance-kernel census. Intent is declared per class; the
verdict compares observation to intent.

```bash
PYTHONPATH=src python scripts/canonicalization_probe.py
```

| Class | Intent | Observed | Verdict |
|---|---|---|---|
| unicode-key-collision (NFC vs NFD) | distinct | collapsed | **unintended-kernel** |
| nested-unicode-key-collision | distinct | collapsed | **unintended-kernel** |
| negative-zero | equivalent | distinct | over-discrimination |
| integer-above-2^53 | distinct | distinct | sound (CPython only — see below) |
| int-vs-float-same-value | distinct | distinct | sound |
| bool-vs-int | distinct | distinct | sound |
| float-exponent-form | distinct | distinct | sound |
| safe-integer-neighbours | distinct | distinct | sound (positive control) |
| object-key-order | equivalent | collapsed | sound (positive control) |
| array-element-order | distinct | distinct | sound (positive control) |
| large-document-single-byte | distinct | distinct | sound (positive control) |

**Verdict counts: 8 sound, 2 unintended-kernel, 1 over-discrimination.**
Provenance is `census` — exact counts, no confidence intervals.

The four positive controls pass, which is what makes the two failures credible
rather than an artifact of an over-strict probe.

Findings [F-03](FINDINGS_REGISTER.md#f-03) through
[F-07](FINDINGS_REGISTER.md#f-07) record these in full. The headline:

```python
>>> canonical_json_bytes({"é": 1, "é": 2})   # NFC key, NFD key
b'{"\xc3\xa9":2}'
```

Two distinct input keys, one output key, **no error raised**. The receipt hash
covers a document that is not the document supplied.

The probe exits non-zero while any class carries `unintended-kernel` or
`rejection-asymmetry`, so this measurement is falsifiable rather than asserted:
fix the canonicalizer and the probe goes green.

### RFC 8785 (JCS) conformance

| Value | cc-framework emits | JCS emits | |
|---|---|---|---|
| `1e30` | `1e+30` | `1e+30` | ok |
| `10**30` | `1000000000000000000000000000000` | `1e+30` | **diverges** |
| `1.0` | `1.0` | `1` | **diverges** |
| `-0.0` | `-0.0` | `0` | **diverges** |
| `1e-7` | `1e-07` | `1e-7` | **diverges** |
| `100.0` | `100.0` | `100` | **diverges** |

Five of six number forms diverge from RFC 8785. cc-framework's canonical form is
`json.dumps(sort_keys=True)`, which is a *convention*, not a *standard*, and it
is not the convention its sibling repositories use.

---

## 7. Claim-discipline surface

### What exists

| Artifact | Lines | Enforced by |
|---|---:|---|
| `docs/research/NON_CLAIMS.md` | — | prose + verifier substring engine |
| `docs/claims/CLAIM_BOUNDARY_MANIFEST.md` (C0–C5 levels) | — | **nothing** |
| `docs/claims/claim_boundary_manifest.v0.1.json` | — | **nothing** |
| `scripts/validate_claim_boundary_manifest.py` | — | **nothing** |
| `src/cc/evidence/permission_compiler.py` forbidden-phrase list | 202 stmts | unit tests |
| `docs/theory/theorem_ledger.md` | 10 KB | test witnesses named per theorem |
| `docs/research/THEOREM_LEDGER.md` | 3 KB | — (second, divergent ledger) |

`grep` across `.github/workflows/` and `.pre-commit-config.yaml` for
`claim`/`scan` returns only Bandit and detect-secrets. **No claim gate runs in
CI.** `scripts/validate_claim_boundary_manifest.py` is referenced by no
workflow, no Makefile target, and no test.

### Proto-scan: Ghost-Ark's forbidden patterns applied here

| Pattern | Hits |
|---|---:|
| `guarantee` | 70 |
| `production[- ]ready` | 7 |
| `enterprise[- ]ready` | 2 |
| `tamper[- ]proof` | 1 |
| `enterprise[- ]grade`, `formally verified`, `zero risk`, `secure by default`, `unbreakable`, `audit complete`, `one-click compliance` | 0 |

Inspection of the hits matters more than the count:

- The `guarantee` hits are **overwhelmingly negated** — `does not guarantee`,
  `no guarantee of`. A naive scanner ported from Ghost-Ark would produce 70
  false positives on this repository and be switched off within a week.
- The `production[- ]ready` and `tamper[- ]proof` hits are mostly **inside the
  claim-discipline machinery itself** (`permission_compiler.py`'s forbidden list,
  `NON_CLAIMS.md`, the release checklist saying the project is *not*
  production-ready).

So the honest reading is: **cc-framework's prose discipline is already good; its
enforcement is absent.** The scanner this repository needs is not Ghost-Ark's —
it needs negation and allowlist handling from day one. See
[W2](EPISTEMIC_UPGRADE_PLAN.md#w2).

---

## 8. CI gates

| Gate | Present | Scope |
|---|---|---|
| ruff lint | yes | whole repo |
| ruff format | yes (pre-commit) | whole repo |
| mypy | yes | **7 files of 90** |
| pytest | yes | full suite, Python 3.10–3.13 |
| coverage threshold | **no** | — |
| claim scanner | **no** | — |
| claim-boundary manifest validation | **no** | — |
| artifact boundary | yes | `scripts/check_artifact_boundary.py --static` |
| package build + twine check | yes | — |
| enterprise smoke (moto) | yes | — |
| minimal example | yes | `examples/minimal/run_bounds.py` |
| Bandit / detect-secrets / pip-audit | yes | `security.yml` |
| mutation testing | **no** | — |
| property-based test gate | **no** | hypothesis is a declared dep; no gate |
| cross-language verifier agreement | **no** | no second implementation exists |
| adversarial receipt corpus | **no** | — |
| reproducibility replay manifest | partial | `scripts/reproduce_paper.py`, not a manifest |

---

## 9. Repository hygiene

| Item | Measurement |
|---|---|
| Largest tracked files | 7 Blender renders, 2.2–2.8 MB each, **17.8 MB total** |
| Archived JSONL checkpoints | 4 × 636 KB in `docs/archive/generated-checkpoints/` |
| `.git` size | 43 MB against a 54 MB working tree |
| Duplicate theorem ledgers | `docs/theory/theorem_ledger.md` (10 KB) and `docs/research/THEOREM_LEDGER.md` (3 KB) |
| Legacy shim | `src/cc/_legacy/` — 89 lines, one file |
| `build/` directory on disk | present, untracked (0 files in git) |
| Schema `$id` host | `cc-framework.local` — not resolvable |

---

## Reproduction

Every table above is regenerated by:

```bash
python scripts/upgrade_baseline.py --out docs/upgrade/baseline.json
```

The script is a deliverable of workstream [W0](EPISTEMIC_UPGRADE_PLAN.md#w0). Until
it exists, this document is a **hand-recorded measurement**, and it says so
here rather than implying automation that is not present.

## What was not measured

Reported so that absence is not mistaken for a null result:

- **Mutation score.** No mutation testing has ever been run on this repository.
- **Cross-language agreement.** No second implementation of the kernel exists
  inside this repository to disagree with.
- **External reviewer reproduction.** No stranger has ever reproduced these
  artifacts. Every adversary in this tree was written by the author of the code
  it attacks.
- **Timing dispersion.** Every timing above is a single measurement.
- **Real guardrail marginals.** No `p_i` in this repository was measured against
  a production guardrail. All are supplied, synthetic, or assumed.
