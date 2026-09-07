# Identical Shadows — what layer counts do and do not establish

**Status:** research note plus a reproducible script; imported browser prototype
retained locally for review.
**Evidence class:** local and elementary. The Python script recomputes reference
intervals using SciPy/HiGHS and declared constants. The browser prototype has a
separate simplex implementation; this does not establish independent authorship
or external reproduction. The novelty claimed is in the framing and the
artifact, not in the mathematics.
**Companion artifact:** interactive prototype, not included in this public release.


---

## 0.0 Revision 2 — corrections

Nine claims in revision 1 were overstated or wrong. All are fixed below; each was
re-checked computationally before being changed (`identical_shadows.py`).

| Revision 1 said | Verdict | Revision 2 says |
|---|---|---|
| Independence is the interval's midpoint | wrong | True only at m = 0.5. In the symmetric regime the interval is [0, m²] and the product is m³, so independence sits exactly **m** of the way across. At m = 5% it is 5% up from the floor. |
| Ceiling = min over pairs | not general | Inclusion–exclusion, t ≤ 1 − Σmᵢ + Σpᵢⱼ, binds strictly below min-pair in a substantial share of feasible instances. The LP is the authority. |
| "Identified at k=2 and never again" | wrong | The moment map is globally injective at k=2, non-injective for k≥3. Particular fibers still collapse — S2 pins the answer exactly at k=3. |
| "5.4% of the joint law is pinned" | category error | Affine degrees of freedom. Dimension ratios are not information fractions. |
| "Every layer after the second multiplies the claim" | over-scoped | Holds in the symmetric regime. In general a layer **does** tighten the ceiling when its measured pairwise overlap is below any existing pair. |
| "The reducer is where the information dies" | unfair to Inspect | Inspect stores each scorer separately per sample and ships a `collect` reducer that "preserves the individual values as a list — for example to feed an inter-rater agreement metric." Withdrawn. |
| Scorer vectors = guardrail joint law | conflation | A scorer evaluates; a guardrail blocks. A semantics/topology bridge is a precondition. |
| The interval *is* definitional uncertainty | over-claimed | Analogy only. VIM's quantity comes from finite specification of the measurand; this one from partial identification given retained summaries. |
| "The all-miss corner is on no face" | geometrically false | It lies on three faces and three edges; it is pooled, never isolated. |
| "No instance of anyone computing a joint" | withdrawn | Multimodal Safeguard Bench builds block-on-either ensembles and argues for complementary blind spots. |

**The sharper statement that replaced the midpoint error.** In the symmetric regime
independence sits at exactly `m` of the way across [0, m²]. So the *better* the
guardrails, the *more optimistic* the independence assumption is relative to what the
evidence permits: at m = 1% it sits one percent up from the most favourable world
consistent with the data. The correction improved the result.

---

## 0.1 External evidence: two real guard pairs

A "block on either" ensemble figure is not a summary of the joint law — it **is** one
moment of it, since `P(either blocks) = 1 − P(all of S miss)`. Two detection rates plus
one ensemble rate therefore give three equations for the three free parameters of a
2×2 table. **At two guards the published aggregates determine the joint exactly** — the
k=2 exact-identification result being used, not circumvented. No per-item data required.

Source: Multimodal Safeguard Bench, README result tables (`results/full_run/metrics.json`,
`ensemble.json` hold the aggregates; raw `gen_*.jsonl` are gitignored by the authors
because they may contain harmful text).

**Case A — Llama-Guard-4 + LlamaGuard-3-Vision, harmful text, n = 200.**
Published: 92.5%, 89.0%, block-on-either 96.0%. Implies 15 and 22 misses, **8 both**
(4.00%). Independence predicts 0.825% ≈ 1.65 items. Ratio **×4.85**, φ = 0.3853,
Fisher exact p = 2.32e−5.

**Case B — Llama-Guard-4 + ShieldGemma-2, harmful image, n = 200.**
Published: 82.0%, 87.0%, union 97.0%. Implies 36 and 26 misses, **6 both** (3.00%).
Independence predicts 2.34% ≈ 4.68 items. Ratio **×1.28**, φ = 0.0511,
Fisher exact p = 0.43 — **not distinguishable from independence.**

The contrast is the finding: dependence is not a property of guardrails, it is a
property of a *pairing*.

**Residual coverage.** Of Llama-Guard-4's 15 text misses, LG3V catches 7 (46.7%), at an
over-refusal cost rising from 11.8% to 56.4%. Of its 36 image misses, SG2 catches 30
(83.3%) for 11.8% → 14.8%. ShieldGemma-2 has *zero* text detection and is still the
better addition. **Standalone guard quality does not rank guards by what they add.**

**Boundaries.** One benchmark, 200 harmful items per modality, particular models,
policies and thresholds. Two pairs are not a survey. Neither case is an identification
puzzle — at k=2 nothing is unidentified — so these measure whether the independence
*assumption* is safe, not the k≥3 gap. The 2×2 tables are reconstructed algebraically
on the assumption that each ensemble figure is an item-level OR over the same items;
every rate lands on a whole item count, which is consistent, but per-item records are
deliberately unpublished and the assumption is unchecked. LG3V refuses 55% of benign
items, which makes it a strange object to call a guardrail. Not verified with the authors.

## 0.2 The ensemble table is the channel

Harmful-content evaluations withhold per-item records for good reasons and will keep
doing so. They do not need to be released. Each ensemble row over a guard subset is one
joint moment. These aggregates avoid releasing raw item text, but are not a
privacy guarantee; small cells and combinations of disclosures require review:

| Published for 3 guards | Moments | Degrees of freedom left |
|---|---:|---:|
| detection rates only | 4 | 4 |
| + one pairwise ensemble | 5 | 3 |
| + all three pairwise ensembles | 7 | 1 |
| + all pairs and the 3-way | 8 | **0 — fully determined** |

The subset-sum transform is invertible, so this holds for every k: a benchmark reporting
block-on-either for all 2^k − 1 subsets has published its entire joint failure law.
**The ask to a benchmark author is one line: publish every subset's ensemble row, not
just the pairing you recommend.**

---

## 0. The finding in one paragraph

Per-layer rates can discard distinctions relevant to joint failure. In the
displayed symmetric regime, every miss rate is 5% and every pairwise co-miss
rate is 0.25%. The sharp upper bound on "every layer misses the same item" is
0.25% for three through eight layers, while the independence product decreases.
This flat ceiling is conditional on those measurements; it is not a theorem
about arbitrary stacks. One full-stack block-on-any rate identifies the all-miss
target under the same-item OR semantics. Per-item vectors or all subset moments
support the larger task of reconstructing the full joint law.

## 1. What was already here

E1 established that pairwise evidence need not identify a three-way event, using
the parity oracle (S4a/S4b): two declared joint laws with identical singleton
rates (0.5) and identical pairwise overlaps (0.25) whose three-way unsafe-pass
probabilities are 0 and 0.25. That result is recorded in
`docs/research/E1_DEPENDENCE_EVIDENCE_STUDY.md` and frozen in
`artifacts/empirical/e1/study.json`.

This note adds four things E1 did not state.

### 1.1 The fiber is exactly one-dimensional, and independence is its midpoint

For three binary guardrails the constraint system — total mass, three marginals,
three pairwise moments — has rank 7 over 8 atom probabilities. The affine
solution set is therefore a **line**, and its intersection with the simplex is a
segment. Its two endpoints, in the equal-rate case, are the two regular tetrahedra
inscribed in the cube `{0,1}^3`: the even-parity corners and the odd-parity
corners. The null direction is exactly the parity sign vector
`(-1,+1,+1,-1,+1,-1,-1,+1)`.

In this particular case (m = 0.5) the independent law is `(even + odd) / 2` —
the midpoint of the segment. **That is special to m = 0.5**; see §0.0. The product
baseline is not a conservative default or an approximation; it is one arbitrary
interior point of a line whose position was never measured, and it happens to be
the centre.

### 1.2 The gap opens at exactly three layers

The percentages below are affine dimension shares, not information fractions.

| Layers | Atoms | Free parameters | Pinned by singles + pairs | Unmeasured dimensions | Dimension share |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 3 | 3 | **0** | 100.00% |
| 3 | 8 | 7 | 6 | 1 | 85.71% |
| 4 | 16 | 15 | 10 | 5 | 66.67% |
| 5 | 32 | 31 | 15 | 16 | 48.39% |
| 6 | 64 | 63 | 21 | 42 | 33.33% |
| 8 | 256 | 255 | 36 | 219 | 14.12% |
| 10 | 1024 | 1023 | 55 | 968 | 5.38% |

Two binary events are **globally identified** by two marginals and one co-miss
rate: three unknowns, three measurements, nothing left to assume. The evidence
map of singleton and pairwise moments is globally injective at k = 2 and
non-injective for k >= 3. Particular feasible fibers may still be singletons,
as S2 demonstrates. The measurement count grows quadratically in k; the number
of atom probabilities grows exponentially.

### 1.3 The bound is flat in the number of layers

Hold every layer at miss rate `m` and measure every pairwise co-miss rate at
exactly `m²` — the friendliest data the world could hand you.

| Layers | Independence claim | Sharp bound | Overstatement |
|---:|---:|---:|---:|
| 2 | 0.250% | [0.25%, 0.25%] | — |
| 3 | 0.0125% | [0, 0.250%] | ×20 |
| 4 | 0.000625% | [0, 0.250%] | ×400 |
| 5 | 0.0000313% | [0, 0.250%] | ×8,000 |
| 6 | 0.00000156% | [0, 0.250%] | ×160,000 |

*(m = 0.05.)*

In this symmetric regime the upper bound does not move, because it equals
`min over pairs = m²` and a new layer contributes no better-measured pair. Change
the regime and it does move: a layer whose measured overlap with an existing layer
is below every current pair tightens the ceiling to exactly that value. Note also
that `min over pairs` is not the general sharp bound (see §0.0). The fourth and fifth layers change the honest ceiling by exactly
zero. This is one of the Fréchet–Hoeffding bounds and is entirely elementary; the
point is what it implies about the artifact organisations actually publish.

### 1.4 The unification with the Provenance Kernel Problem

`docs/research/PROVENANCE_KERNEL_PROBLEM.md` (Ghost-Ark) and this note are the
same theorem applied to two different maps.

|  | Ghost-Ark | CC-Framework |
|---|---|---|
| Map | canonicalizer `C : documents → bytes` | moment map `M : joint laws → measured rates` |
| Many-to-one residue | `ker(C)` | the Fréchet class / fiber |
| Decision | consumer interpretation `⟦·⟧_p` | `P(all guardrails miss)` |
| Failure condition | decision does not factor through `C` | query does not factor through `M` |
| Central artifact | a colliding document pair | an endpoint witness pair |
| Suppressed argument | the consumer population `P` | the joint dependence structure |

Both projects' central computational object is the **production of a witness
pair**: two things the system cannot separate that the decision must separate.
Ghost-Ark's differential suite exhibits colliding documents; CC's LP exhibits
endpoint witnesses. The shared discipline is one sentence, already stated in the
Ghost-Ark note: *demonstrate the kernel, do not assert its emptiness.*

The general test:

> **Does the decision depend only on what was measured? If not, a witness pair
> exists — and it can be produced.**

## 2. The outside-field borrow: definitional uncertainty

JCGM 200:2008 (VIM), entry 2.27 defines **definitional uncertainty** as the
"component of measurement uncertainty resulting from the finite amount of detail
in the definition of a measurand", and notes:

> "Definitional uncertainty is the practical minimum measurement uncertainty
> achievable in any measurement of a given measurand."

The interval in §1.3 is a definitional uncertainty. It is not sampling error;
more items do not shrink it. It is the price of having defined the measurand as
three per-layer rates when the decision depends on a joint law. Metrology's
enforcement mechanism is the one worth importing: a calibration laboratory may
not report a result without first declaring the measurand, must give the
uncertainty budget component by component, and must state that the result
applies to the item *as received*.

Reliability engineering learned the modelling half the harder way: redundant
channels in reactor protection systems were found to fail together, and
common-cause failure models (beta-factor and successors) exist precisely because
a count of redundant channels had been treated as evidence of independence.

## 3. What would change a decision

1. **Log the vector, not the rates.** Store `{item_id, missed_by: [1,0,1]}` per
   item rather than three separate rates. The joint law is then observed and the
   interval collapses to sampling error. This is a logging-schema change, not a
   research program, and the data already exists in memory in every guardrail
   evaluation run.
2. **Reprice the marginal layer.** Under the evidence regime organisations
   actually collect, the k-th layer for k ≥ 3 provably does not tighten the
   bound. Budget spent on a fourth guardrail buys claim, not evidence; budget
   spent on joint logging buys evidence.
3. **State the topology.** These bounds describe an OR-block stack (an item
   passes only if every layer misses). A cascade, where a later stage inspects
   only what an earlier stage flagged, is worse on this axis: the later stage
   never sees the earlier stage's misses and cannot reduce them at all. "Defense
   in depth" is ambiguous between topologies with opposite implications.

## 4. Non-claims

- **No new mathematics.** Fréchet–Hoeffding bounds date to the 1930s; pairwise
  independence failing to imply mutual independence is the textbook Bernstein
  example; partial identification is a mature field. The dimension count is
  linear algebra.
- **No survey has been conducted.** The stacks in the artifact are constructed
  witnesses, not observed systems. This establishes that the ambiguity is
  *available*, not how often it occurs in deployment. Measuring the frequency
  would require joint failure logs — which is exactly the thing not published.
  This is the same limitation `evidence-scope-inflation.md` states about itself.
- **No named product is alleged to be miscalibrated.** The demonstrations are
  against this project's own canonicalizer and this project's own study.
- **The interval is not a risk estimate.** Its lower end is zero in most
  regimes. A wide interval means the evidence is uninformative about total
  failure, not that failure is likely.
- **Sampling error is excluded.** All rates are treated as exactly known. Finite
  samples widen the interval further.
- **The k = 2 exact-identification result assumes both marginals and the pair are
  measured on the same items.** Rates from separate evaluations do not compose.

## 5. Reproduction

```
python3 identical_shadows.py          # all tables in this note, from scratch
```

Requires numpy and scipy. The script re-derives the E1 reference intervals for
S1–S4b (regimes I0 and I2) and fails if they differ from the constants declared
in its `E1` table. It does not load `artifacts/empirical/e1/study.json`; that
source relationship must not be mistaken for a live file-binding check. The
browser prototype carries a separate two-phase simplex and client-side checks.

Release review, 2026-09-07: moved the script's entry point after all definitions
and connected the Revision 2 calculations. Previously the process exited before
those functions were defined, so the advertised full command never ran them.
The reference-only `--check` command retains its narrower scope. The imported
browser prototype is retained as REVIEW, not deployed as a verified site page.

## 6. Sources

1. CC-Framework, `artifacts/empirical/e1/study.json`; `docs/research/E1_DEPENDENCE_EVIDENCE_STUDY.md`; `src/cc/kernel/sensitivity.py`.
2. Ghost-Ark, `docs/research/PROVENANCE_KERNEL_PROBLEM.md`; `tests/differential/provenanceKernel.test.ts`.
3. JCGM 200:2008, *International Vocabulary of Metrology*, entries 2.3 (measurand) and 2.27 (definitional uncertainty).
4. promptfoo, "Testing and Validating Guardrails" — reports attack block rate, false-positive rate, indeterminate rate and latency per guardrail; does not treat layered composition.
