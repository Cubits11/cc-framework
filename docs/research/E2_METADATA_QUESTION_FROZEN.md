# Frozen metadata question — observational unit of the candidate E2 frame

**Frozen: 2026-08-24, BEFORE any external contact and BEFORE any research into whether
the answer already exists.** This document exists so that the external conversation cannot
turn into improvisational study design. It is sealed by the same discipline as
`E2-VNEXT`: the question, the admissible evidence, the inadmissible evidence, and **all**
consequence branches are fixed in advance, so no answer can be reinterpreted after arrival.

`E2-VNEXT` remains sealed and terminally NOT ACTIVATED. This question does not revive it.

---

## 1. The question

> **What is the appropriate observational and resampling unit for the candidate finite
> dataset frame?**

Not "does a column named `template_id` exist." The underlying question is about
**generation ancestry**, because the unit could sit at, above, or below any single column:

```
  source seed  ->  template  ->  instantiated record
                                  (the 2,750 rows)
```

If multiple records descend from a shared parent, then the rows are not independent
observations, and Δ = Cov(A,B) computed over rows is a weighted average whose weights are
the generator's expansion cardinalities rather than a property of any detector pair.

**This question precedes models.** It is not metadata housekeeping; it determines whether
a legitimate sampling/inference design exists at all.

## 2. Why it is being asked now

Adjudicated in [`E2_VNEXT_ADMISSIBILITY_COURT.md`](E2_VNEXT_ADMISSIBILITY_COURT.md) §4.
The dataset card at the pinned revision declares the attack items were built from
human-designed templates *"authored with slot variables and expanded deterministically with
scattered mixed-radix sampling,"* then *"deduplicated and split-isolated by template."*

Split-isolation **by template** concedes that templates recur across records — the
constraint is vacuous otherwise. The released schema
`[id, text, label, class, attack_family, technique, severity, language]` contains **no
parent identifier**, and the template count is disclosed nowhere.

## 3. Evidence that WILL be accepted

- Maintainer-authored documentation describing the generation ancestry.
- Stable, revision-specific metadata mapping `record_id -> parent/template id`.
- A published generator, config, or manifest from which ancestry is recoverable **without
  reading record text**.
- An authoritative maintainer statement that **no shared parent structure exists**, with
  enough description of the generation procedure to make that statement checkable.

## 4. Evidence that will NOT count — and may not be produced

Recording these in advance so they cannot be rationalised into admissibility later:

- semantic clustering performed by us;
- embeddings or vector similarity over records;
- lexical / n-gram / prefix similarity heuristics;
- template structure inferred from record text;
- **any inspection of record text whatsoever**;
- model output similarity;
- clustering derived from any detector's scores;
- a third party's reconstruction not endorsed by the maintainer.

**Rationale.** Every item above requires reading the study population, which would destroy
the precommitment this entire court exists to protect. If ancestry can only be recovered by
looking at the data, then for this study it **cannot be recovered at all**.

## 5. Consequence branches — all fixed in advance

| World | Answer | Consequence |
|---|---|---|
| **A** | Stable parent IDs exist and are supplied at the pinned revision | Redesign analysis around the hierarchy: unique-parent count, expansion cardinalities, cluster sizes, cluster-aware variance, cluster-level resampling. **Re-derive power at the proper unit.** Frame-capacity gate re-evaluated on units, not rows |
| **B** | Templates exist but the mapping was not preserved / cannot be supplied | The independent unit is unrecoverable without content inspection. **Reject this population for this experiment.** Do NOT manufacture parent IDs algorithmically — that would be a separate preregistered measurement problem, not a fix |
| **C** | No shared parent structure in the statistical sense (e.g. independently parameterised draws from a grammar with no stable parent) | Our clustering assumption was wrong and the source corrected us — a genuinely good outcome. A row-level design may remain possible, **subject to the generation semantics actually described** |
| **D** | Hierarchical structure exists above the template (e.g. `source seed -> template -> record`) | `template_id` alone is **insufficient**. The resampling unit sits higher; design must be re-derived at the top of the ancestry |
| **—** | Ambiguous, partial, or no response | **Remain RED.** No inference is drawn from silence. `NO PUBLIC EVIDENCE FOUND`, never "there is no structure" |

## 6. Standing constraints during this phase

Unchanged and non-negotiable:

- **No loader.** No model artifact retrieval. No dataset rows, previews, viewer,
  `/rows`, or `/first-rows` endpoints.
- **No threshold work, no calibration, no candidate replacement, no `E2-VNEXT.1`.**
- The sealed `E2-VNEXT` artifacts stay byte-identical.
- **No outreach is sent without explicit human authorization**, per message and per venue.

## 7. Order of operations

1. Freeze this question. *(this document)*
2. Determine from **public primary sources only** whether the answer already exists. If it
   does, **no outreach is needed** and the branch is decided without contacting anyone.
3. Only if unresolved: draft a single narrowly-scoped public request, and obtain explicit
   authorization before posting it under the project's identity.
4. Record the answer against the branch table in §5 **without amendment**.

**The request must not signal a preferred answer.** *"There are no templates"* has to be as
easy and as welcome to give as any other response, or the answer is worthless.
