# E2 dry run — instrument validated, measurement is a null, real E2 still owed

**Status: DRY RUN — not E2. E2 remains FROZEN and UNTESTED.**
Date: 2026-08-23. Artifacts: `experiments/e2_dryrun/` (harness, frozen corpus,
and the reference observation set, in the repository root tree).

This records a rehearsal of the full [E2 measurement contract](E2_MEASUREMENT_CONTRACT.md)
pipeline against three *toy* guardrail mechanisms. It exists to answer one
question honestly and early: **does the E2 instrument actually run end to end,
and what would real E2 cost?** It is deliberately not a result about deployed
guardrails, and it does not move E2 off the `UNTESTED` rung.

## What was run

Three genuinely distinct mechanism families, each an existing guardrail in this
repository, calibrated to a common operating point (target FPR 0.10) on a frozen
22-item benign subset *before* any harmful-item outcome was treated as data:

| id | mechanism | family |
| --- | --- | --- |
| `gr.keyword` | `cc.guardrails.keyword_blocker` | lexical (exact + fuzzy keyword) |
| `gr.regex` | `cc.guardrails.regex_filters` | structural (regex patterns) |
| `gr.semantic` | `cc.guardrails.semantic_filter` | distributional (TF-IDF cosine) |

Population: 22 harmful items (12 plain + 10 obfuscated), authored and frozen in
`experiments/e2_dryrun/corpus.jsonl`. The frozen reduction `h.unsafe_pass.v1`
maps a decision to the failure event `Y = 1` iff the guardrail *allowed* a
harmful item (an unsafe pass). One `h`, one operating point per mechanism, all
frozen in `experiments/e2_dryrun/run_dryrun.py` before outcomes were inspected.
Whatever `Δ` resulted is reported here, including its sign and its
insignificance.

## What the instrument established (Observed → Derived)

- **Conformance is real.** The 66 emitted rows pass
  `scripts/validate_e2_observations.py` with **zero violations**; all 22 items are
  complete-case across all three guardrails (the Layer B shared-item rule holds).
  The measurement pipeline the contract specifies is executable, not aspirational.
- **Unsafe-pass marginals:** keyword 0.591, regex 0.545, semantic 0.591. All three
  toy mechanisms miss a majority of the obfuscated harmful items — the corpus was
  built to make the mechanisms disagree, and it does.
- **Excess joint-failure Δ = p11 − pA·pB, per pair:**

  | pair | Δ | reading |
  | --- | --- | --- |
  | keyword & regex | **−0.050** | joint failure *below* independence — the product baseline is conservative here |
  | keyword & semantic | **+0.014** | joint failure *above* independence — the product baseline is optimistic here |
  | regex & semantic | −0.004 | indistinguishable from independence |

  One pair each direction: E1's central qualitative lesson — *a product baseline
  is neither reliably conservative nor reliably optimistic* — reproduces even in
  toy mechanisms.

## What the instrument refused to let us claim (Interpreted)

- **At n = 22 the observed dependence is within sampling noise.** The simultaneous
  Hoeffding radius for the rate estimates at δ = 0.05 is **0.353** — roughly seven
  times the largest |Δ| observed. The pre-registered **column-permutation control**
  (which destroys real dependence while preserving marginals) produced a max |Δ| of
  **0.087**, *larger* than the observed max |Δ| of 0.050. The instrument correctly
  reports its own null: none of these deltas is distinguishable from independence
  at this scale.
- **The negative controls behaved exactly as specified.** The duplicate-column
  common-cause control gave Δ = pA(1−pA) = 0.2417 to within 1e-9; an independently
  written recompute of Δ matched the matrix computation. The controls that were
  supposed to move, moved; the one that was supposed to hit a closed form, hit it.

## What this is not (Claimed)

- **Not E2.** These are Python filters in this repository, not deployed guardrails
  on a named real population. There are no data rights, no external annotation, and
  the reference labels were authored here rather than adjudicated before outcomes by
  an independent party. The E2 defeaters name this directly.
- **No claim about real guardrail dependence.** The Δ values are scoped entirely to
  this authored corpus and these frozen configs.
- **The missingness machinery is unexercised.** This deterministic local run has
  zero missingness, so nothing here tests the Layer E handling that real E2 most
  needs.

## Decision under the contract — HOLD, with two valuable results

Per the contract's practical defeaters, this is a **HOLD**: "the interval is too
wide to affect any bounded decision" at this scale, and the mechanisms and
population are not the ones E2 is about. Two results are worth banking, and the
contract explicitly counts them as results rather than failures:

1. **The instrument is validated.** The E2 pipeline — schema, shared-item rule,
   frozen reduction, estimand, and pre-registered controls — runs end to end and
   conforms. When real data arrives, the measurement path is already proven.
2. **The cost of real E2 is now a number.** To estimate the failure rates to
   ε = 0.05 at δ = 0.05 requires **≈ 1,097 shared items** run through every
   guardrail. That is the concrete bar the next step must clear, and it is what
   `/now/` on the public site should track against.

A third, incidental: exercising the semantic mechanism surfaced a real defect —
`SemanticFilter.score`/`calibrate` raised on any populated template set under
modern SciPy (`not <sparse matrix>` is ambiguous). Fixed in this change with a
regression test; the guardrail was previously unusable end to end.

## What real E2 still requires (the documented failure-to-obtain)

1. Deployed, externally hosted guardrails with recorded version identity — not
   local filters.
2. A named real population with clear data rights and an explicit sampling frame.
3. Reference labels adjudicated *before* guardrail outcomes are visible, with
   inter-annotator uncertainty — not author-declared labels.
4. Scale on the order of a thousand shared items per the number above.
5. Real missingness (`upstream_moderation`, `timeout`, `provider_refusal`, …)
   represented rather than absent.

Until those exist, E2 is `UNTESTED`, and this document is the honest record of
how far the instrument reaches without them.
