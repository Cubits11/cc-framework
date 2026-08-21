# E1 — Dependence-Evidence Value Study

> Not a system that knows more. A system that makes it harder for any of us to
> pretend we know more than the evidence permits.

**Frozen protocol — controlled synthetic study complete; real-world pilot
UNTESTED.** This document is the frozen wager. The verdict and its corrections
are recorded in [E1_DECISION_RECORD.md](E1_DECISION_RECORD.md); the successor
wager is frozen in [E2_MEASUREMENT_CONTRACT.md](E2_MEASUREMENT_CONTRACT.md).
This document is the sole active empirical wager for the current
research cycle. It replaces neither the kernel nor the repository's
non-claims, and it does not license a deployment-safety claim.

## Wager contract

| Field | Frozen declaration |
| --- | --- |
| Proposition | For declared finite synthetic joint laws, the existing atom-LP returns sharp intervals and feasible endpoint witnesses under evidence regimes I0–I3. Count-derived I0–I2 intervals have the stated simultaneous-coverage scope under fixed-query, iid sampling assumptions. More evidence can narrow an interval; it need not identify a three-way event. |
| Population | IID draws from each explicitly declared synthetic three-indicator joint law. There is no real-world population in E1. |
| Unit | One synthetic row with three binary failure indicators, `g1`, `g2`, and `g3`. |
| Failure variable | `Z_i=1` means guardrail failure / unsafe pass. This is a study convention, not a claim about a named product or adapter. |
| Primary event | `P(g1=1, g2=1, g3=1)`: all three guardrails fail, so an OR-blocking stack passes the row. |
| Evidence regimes | **B0:** product baseline, explicitly assuming independence; **I0:** singleton moments; **I1:** I0 plus the predeclared lexical pair `g1&g2`; **I2:** all pairwise moments; **I3:** full joint record, used only as a measured synthetic reference. |
| Sampling claim | I0–I2 use simultaneous Hoeffding intervals at `delta=0.10` for fixed-before-sampling queries and iid rows from the named synthetic population. This is an outer-coverage statement conditional on those assumptions. |
| Falsifiers | A failed endpoint witness; an LP interval excluding its declared synthetic truth; failure of the parity oracle; an invalid manifest/hash; or a coverage row whose predeclared 95% Wilson lower limit falls below nominal coverage minus 0.06. |
| Non-claims | No real guardrails, real prompts, labels, attack success, causal mechanism, representativeness, adapter comparison, or deployment risk has been measured. I3 is not deployment truth. |
| Decision rule | A credible real-data pilot is absent, so the only permitted decision after E1 is **Narrow**: retain a mathematically and synthetically validated claim; do not expand it to external safety or robustness. |

The product baseline is always reported but is never silently substituted for
joint evidence. The design, target event, pair selection, sample sizes,
replicates, seed, and coverage rule are all fixed in the artifact before its
output is interpreted.

## Controlled suite and results

The implementation reuses
[`cc.evals.dependence_benchmark`](https://github.com/Cubits11/cc-framework/blob/main/src/cc/evals/dependence_benchmark.py),
not a parallel evaluation system. Its full deterministic output is in
[`artifacts/empirical/e1/study.json`](https://github.com/Cubits11/cc-framework/tree/main/artifacts/empirical/e1)
with a hash manifest and replay verifier.

| Surface | Declared law | True primary event | B0 product | I0 interval | I1 interval | I2 interval | I3 interval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | independent Bernoulli(0.5) | 0.125 | 0.125 | [0, 0.5] | [0, 0.25] | [0, 0.25] | [0.125, 0.125] |
| S2 | common cause: 000/111 each 0.5 | 0.5 | 0.125 | [0, 0.5] | [0, 0.5] | [0.5, 0.5] | [0.5, 0.5] |
| S3 | mutually exclusive failures | 0 | 0.015625 | [0, 0.25] | [0, 0] | [0, 0] | [0, 0] |
| S4a | uniform even parity | 0 | 0.125 | [0, 0.5] | [0, 0.25] | [0, 0.25] | [0, 0] |
| S4b | uniform odd parity | 0.25 | 0.125 | [0, 0.5] | [0, 0.25] | [0, 0.25] | [0.25, 0.25] |

S4 is the adversarial oracle. Even and odd parity have identical singleton
rates (0.5) and identical pairwise overlap rates (0.25), yet their three-way
unsafe-pass probabilities are 0 and 0.25. Therefore all singleton and all
pairwise evidence still leave the same `[0, 0.25]` interval. This is a direct
counterexample to treating pairwise agreement as identification of a higher
order event.

S5 is the finite-sample check: five declared laws × two events × I0/I1/I2 ×
sample sizes 64 and 256 × 128 deterministic replicates. All 60 frozen grid
cells passed the coverage decision rule. The smallest observed coverage was
0.984375 and the smallest Wilson 95% lower limit was 0.944819, above the
predeclared 0.84 threshold for nominal 0.90 coverage. This supports the stated
simulation diagnostic; it does not prove performance beyond this generator and
finite grid.

S6 is the broken-data control suite: 14 focused test functions across
`tests/unit/evals/test_dependence_evidence.py` (3),
`tests/unit/evals/test_e1_artifact_identity.py` (10), and
`tests/integration/test_e1_dependence_evidence.py` (1). They reject malformed
designs and tampered artifacts and pin artifact identity. The verifier itself
reconstructs exact regime bounds, rechecks endpoint witnesses, checks parity,
validates the rendered CSV against the JSON study, hashes both files, and can
regenerate the full study in memory — these are verifier capabilities exercised
by those tests, not additional test functions.

## Identification is not sampling is not measurement

- **Identification validity:** Given the declared moment constraints, the LP
  returns sharp feasible bounds and verified witness distributions. Parity
  shows that I2 can remain non-identifying.
- **Sampling validity:** The count-derived intervals use a simultaneous
  Hoeffding union bound only for fixed, iid Bernoulli moments from the named
  population. Adaptive target selection, clustering, label error, and shift
  are outside this statement.
- **Measurement validity:** E1 has no real measurement instrument. A later
  pilot must establish data provenance, row definition, label policy,
  guardrail versions/configuration, missingness, deduplication, and population
  limits before its numbers can be called empirical evidence.

## Real-pilot data contract — superseded

**This template has been superseded by the frozen
[E2 measurement contract](E2_MEASUREMENT_CONTRACT.md)**, which specifies the
same obligations at row-schema resolution and makes conformance machine-checkable
(`make verify-e2-contract`). The summary below is retained for continuity.

The real-pilot gate remains closed. If it opens, the pilot must supply this
small contract before a model is run:

| Required field | Minimum content |
| --- | --- |
| Population and sampling frame | Named population, geography/time window, inclusion/exclusion rules, and whether sampling can support any intended generalization. |
| Unit and event | One immutable row identifier; prompt/response version; label definition; exact unsafe-pass event; outcome adjudication and uncertainty policy. |
| Guardrail measurement | Versions, configurations, order/stack semantics, failures/errors, retries, review policy, and per-row output records. |
| Evidence plan | Predeclared event, I0/I1/I2 moments, alpha/multiplicity method, sample target, missing-data policy, and holdout or split-sample plan for any selection. |
| Rights and safety | Provenance, licence/terms, PII and sensitive-content handling, retention, access controls, and red-team safety procedures. |
| Falsification and release | Known failing examples, independent replay instructions, raw-to-derived hash chain, non-claims, and an explicit Hold/Narrow/Proceed decision owner. |

Until that contract and a legally usable dataset exist, the pilot is **UNTESTED**
and the decision remains Narrow. Adaptive attack work is an exploratory later
track, not evidence for this initial validation.

## Reproduction and challenge packet

```bash
make reproduce-empirical-e1
make verify-empirical-e1
make test-empirical-e1
```

An external challenger should: (1) run those commands in a clean environment;
(2) inspect the parity atoms and confirm equal singleton/pairwise moments;
(3) independently implement the eight-atom LP or check the supplied endpoint
witnesses; (4) alter one artifact byte and confirm verification fails; and (5)
try to find a declared synthetic law for which the coverage or sharpness claim
fails. A passing replay establishes artifact and conditional mathematical
consistency—not external validity.

Replay into **any** output directory. `manifest_payload_sha256` covers the
study identity (design, schema versions, and content hashes) and is invariant
under relocation; the absolute path is recorded separately under
`execution_provenance` and is deliberately excluded from the digest. A digest
that differs across two clean checkouts is a genuine divergence, not a path
artifact.

## Source ledger

| Source | Role in E1 | What it supports | Limit |
| --- | --- | --- | --- |
| Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables* (1963), DOI [10.1080/01621459.1963.10500830](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830) | Theory | Bounded iid concentration is the basis for the declared count-interval construction. | Does not establish iid sampling, representativeness, label quality, or deployment safety. |
| Fréchet, *Généralisation du théorème des probabilités totales* (1935), DOI [10.4064/fm-25-1-379-387](https://www.impan.pl/en/publishing-house/journals-and-series/fundamenta-mathematicae/all/25/0/93246/generalisation-du-theoreme-des-probabilites-totales) | Theory | The study's finite coupling/partial-identification framing. | Does not supply an empirical evaluation design. |
| Nasr et al., *The Attacker Moves Second* (USENIX Security 2026) [paper](https://www.usenix.org/system/files/usenixsecurity26-nasr.pdf) | Context | Static evaluations alone are insufficient for a robustness claim; stronger adaptive evaluation belongs in a later, explicit track. | It does not validate CC's synthetic study or its partial-identification engine. |

## Claim ledger

| Claim | Status | Evidence boundary |
| --- | --- | --- |
| The existing finite atom-LP produces the declared exact E1 bounds and feasible endpoint witnesses. | Supported in controlled synthetic suite | Five declared distributions, deterministic artifacts, and witness verifier. |
| Singleton and pairwise evidence can fail to identify a three-way event. | Supported by constructive counterexample | The parity pair only; it is a mathematical fact for this specified finite setting. |
| Artifact identity is invariant under relocation and sensitive to design mutation. | Supported | Manifest schema v2; pinned by `tests/unit/evals/test_e1_artifact_identity.py`. |
| Count-derived E1 I0–I2 intervals met the frozen Monte Carlo coverage rule. | Supported as a simulation diagnostic | Five generators, two sample sizes, two events, fixed seed/grid; conditional on generator and method. |
| CC improves real guardrail safety or robustness. | **Not supported** | No credible real-world pilot or adaptive evaluation has been performed. |

**Decision: Narrow.** Retain the controlled mathematical and synthetic claim,
publish no external-world safety conclusion, and make a future real-data pilot
earn its own contract, provenance, and red-team challenge.
