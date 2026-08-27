# Breakthrough audit — four candidates, all dead, and the one constraint that killed them

**Date: 2026-08-25. Verdict: NO BREAKTHROUGH in the guardrail-dependence direction.**

Commissioned as an explicit search for a defensible contribution CC could make now. Four
candidates were investigated against primary sources, then attacked by two adversarial
auditors briefed that *manufacturing hope would be the worst available outcome*. Both
auditors independently returned **NO_BREAKTHROUGH**.

This document records the kill, the verification I did myself, and the one thing that would
change the answer.

---

## 0. First: a claim from the audit that I checked and REFUTED

The novelty auditor instructed: *"`tau` at `dependence_search.py:1051` is not Kendall's tau,
it is the Pearson phi coefficient. Flag the misnomer before any reviewer sees it."*

**That is wrong. Do not act on it.** For a 2×2 table, Kendall's tau-b, the Pearson φ
coefficient, the Matthews correlation coefficient, and the Pearson correlation of the two 0/1
indicators are **all the same number** — concordant pairs are `n11·n00`, discordant are
`n10·n01`, and tau-b's denominator reduces to `sqrt(p1(1−p1)p2(1−p2))`. Verified numerically
across 8 random paired-Bernoulli designs; all four estimators agree to <1e-9.

`src/cc/redteam/dependence_search.py:1051` is **correctly named**. There is no error for a
reviewer to catch. Recorded because a wrong correction propagating into the repo would be a
real cost, and because it is a standing caution about the rest of the audit.

**What I verified as TRUE**, to 1e-17, on 4 random designs:

    Δ = p11 − pA·pB  =  Cov(A,B)  =  φ · sqrt( pA(1−pA) · pB(1−pB) )

CC's preregistered estimand ([`E2_MEASUREMENT_CONTRACT.md:223`](E2_MEASUREMENT_CONTRACT.md))
is exactly the Bernoulli covariance, and a monotone rescaling of φ. That part stands.

---

## 1. The four candidates

| # | Candidate | Verdict | What killed it |
|---|---|---|---|
| 1 | Injection benchmarks overstate effective n; their CIs are too narrow | **DEAD — PRIOR ART** | arXiv:2510.05709 *is the candidate*: 1.3–5.6× overestimate, on HarmBench and garak. Plus arXiv:2411.00640 §2.2 "Clustered Questions" (Anthropic) supplying the cluster-robust machinery, over Kish 1965 |
| 2 | Guards don't implement a common event, so comparisons are incommensurable | **DEAD — PRIOR ART** | arXiv:2501.15145 (PromptShield, Berkeley) published matched-FPR calibration in this exact subfield and retro-applied it to competitors |
| 3 | Contamination becomes checkable via guards publishing training corpora | **WEAK** | Of 24 guards screened, ~zero publish record-level training data *and* a cutoff. The ones that do are academic rather than deployed — buying checkability with ecological validity |
| 4 | A population exists with both ancestry and an attack-success oracle | **WEAK/DEAD** | Oracle benchmarks score whether an *agent* was compromised, not whether a *text classifier* flagged an injection. Incompatible with a classifier-pair study |

**Every citation above was verified to exist** by fetching its arXiv abstract page. Titles
match. This is not hallucinated prior art.

---

## 2. Where the audit OVERREACHED, stated so it is not over-believed

The novelty auditor claimed arXiv:2504.11168 (Hackett et al., *Bypassing LLM Guardrails*)
is *"your observed-p11-on-a-shared-frame experiment... and it found p11 = 1."*

I read the abstract. **It is an evasion-attack paper**, not a paired dependence measurement.
It demonstrates character-injection and AML evasion against six systems including Azure
Prompt Shield and Meta Prompt Guard, reaching up to 100% evasion. It does not preregister
`Δ`, does not report a joint table, and does not test independence.

So it does **not** literally scoop E2's estimand. What it does is worse for motivation: it
already shows, in-domain, that **one attack class defeats six guards at once**. The
qualitative finding E2 was chasing — guards fail together — is on the record. Measuring Δ
afterwards answers a question the field no longer considers open.

Separately: arXiv:2605.29800 (*Nine Judges, Two Effective Votes*) **is** the full method —
9 models, 7 families, shared items, Kish n_eff, "the 9 judges effectively provide only about
2 independent votes' worth of information," 8–22pp shortfall against independent voting.
That is CC's thesis, with CC's machinery, in an adjacent domain, published May 2026. The
auditor was right about this one.

---

## 3. The structural reason, which matters more than any single scoop

A 2×2 contingency table has only a handful of functionals, and essentially all of them were
named between 1900 and 1965. Δ is the covariance. φ is its normalization. `p11/min(p1,p2)`
is p11 against its Fréchet bound. **Any estimand CC preregisters over two binary guard
outcomes will land on a named classical statistic.**

This explains the whole run of deaths rather than treating them as bad luck:

- S4 → Bernstein 1946 / Hailperin 1965
- Candidate 1 → Kish 1965
- Candidate 2 → Provost & Fawcett 2001, ISO/IEC 19795-1
- Tribunal item 4 ("Fréchet width shrinks with guard quality") → this is the marginal-
  dependence of the covariance, the century-old defect that caused Yule to introduce Q and
  Pearson to introduce φ in the first place

**Rule going forward:** any candidate posed as a functional of a 2×2 table should be assumed
dead until 1900–1965 statistics has been searched, not just arXiv.

---

## 4. The binding constraint — both auditors converged here independently

> **No public prompt-injection benchmark releases per-item detector outcomes.**

Verified for WAInjectBench, PromptShield, and Gate AI: datasets and code are released,
predictions are generated at runtime. Cluster-robust SEs, ICC, and Δ all require per-item
paired outcomes.

Therefore **"no model loader" and "produce a result anyone must respond to" are mutually
exclusive.** The no-loader constraint was adopted to avoid contamination exposure. It has
instead guaranteed that no result can be produced. That is the actual finding of this audit,
and it is not a statement about prompt injection — it is a statement about CC's own design.

Worse: most of these papers report bare percentages with no interval at all, so there is
often no CI to show is too narrow. The methods critique has no constructive payload.

---

## 5. What survives, without inflation

- **The claim-package compiler** is working engineering. **Prior art was NOT searched** for
  it — in-toto/SLSA, Sigstore, RO-Crate, W3C PROV, ACM artifact badging, OSF preregistration
  and nanopublications all occupy that space. Do not lean on it until that search is done.
- **The process record** — a dry run reporting its own null and issuing its own correction, a
  court finding that `preflight.py` executes 6 of its 8 declared gates, two non-activations
  preserved with full provenance — is unusually disciplined. It is a **process artifact, not
  a scientific claim.** A workshop note is its ceiling, and that is a real ceiling, not a
  consolation prize.

## 6. What must be abandoned

1. **Δ as a scientific object.** Not narrowed, not moved to a new corpus.
2. **The belief that no-loader is a differentiator.** It is the binding constraint.
3. **Repairing E2 by fixing its seven blockers.** A perfectly executed E2 now produces
   arXiv:2605.29800's design with k=2 instead of k=9, on a pair whose common event the
   tribunal's own ontology reviewer already returned FAIL on.

## 7. The one thing that would change this answer

Not a cleverer no-loader idea. **Running inference on a defensible frame and publishing the
per-item paired outcomes** — the artifact nobody in the field currently releases, and which
CC's measurement contract, row schema, missingness taxonomy and pairing-integrity validator
were built to produce.

That is a resource contribution, not a statistical one, and it requires abandoning the
no-loader constraint deliberately rather than drifting past it. Whether that is worth doing
is a judgement about cost and appetite, not a question this audit can settle.

**Note on sources.** arXiv IDs 2605.*, 2606.*, 2607.*, 2608.* postdate my knowledge cutoff;
I verified they exist and read two abstracts directly, but I have no independent view of
their quality. The auditor itself flagged arXiv:2607.13918 and arXiv:2605.29800 as
single-author unreviewed preprints. **The verdict survives deleting both**: Hackett
(2504.11168), PromptShield (2501.15145), Llewellyn (2510.05709), Miller (2411.00640) and
Kish 1965 carry it on their own.
