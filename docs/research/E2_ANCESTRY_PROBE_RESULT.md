# Ancestry probe — result against the frozen question

**Probed 2026-08-24, public primary sources only. No row, preview, viewer, or `/rows`
endpoint was opened. No model was retrieved. Nothing was posted anywhere.**

Answers [`E2_METADATA_QUESTION_FROZEN.md`](E2_METADATA_QUESTION_FROZEN.md), whose branches
were sealed **before** this probe ran.

---

## Branch determination: **D — hierarchy sits ABOVE the template** (medium confidence)

**The frozen question's own framing was vindicated.** It insisted the question was about
generation ancestry, *"not merely a column named `template_id`,"* because *"the unit could
sit above the template."* That is exactly what the evidence shows. **Asking this maintainer
for a `template_id` would have been the wrong ask.**

Branch **C is affirmatively refuted** — shared parent structure is not a hypothesis here;
it is documented in the author's own published code. Branch **A is not met** at the pinned
revision. Branch **B's condition currently holds** (mapping unrecoverable from public
artifacts without touching rows), but B alone understates the finding.

---

## What the study population itself publishes

Verified directly against the HF API at pinned revision `e658d94d…`:

| Fact | Value |
|---|---|
| Entire file tree | `.gitattributes`, `CITATION.cff`, `README.md`, `data/{train,validation,test}.parquet`, `sample_preview.jsonl` — **no generator, no scenario source, no `metadata/` directory** |
| Repo history | **exactly two commits, three seconds apart**, both 2026-08-11T23:43:2xZ. The pinned revision *is* HEAD. **There is no earlier revision to mine** |
| Splits | train 3,923 · validation 467 · test 545 = **4,935 total** (corroborates 2,750 attack + 2,185 benign) |
| Community discussions | **ZERO**, all statuses, all types. No maintainer statement on provenance exists anywhere |
| Downloads / likes | 175 / 3 — a two-week-old, individually authored corpus with no independent scrutiny |
| Author | Enes Deniz, ORCID `0009-0006-9491-3565`, affiliation **AltaySec** — independently verified across three primary sources |

### A ninth column the frozen screen never recorded

`CANDIDATE_SCREEN.json` states the schema as eight fields. The datasets-server reports
**nine**: `text, label, class, attack_family, technique, severity, language, `**`source`**`, id`.

The card documents only eight and **omits `source`**. This matters: in the author's Python-era
sibling, `source` is a **struct** carrying exactly the ancestry fields
(`method`, `generator`, `seed`, `record_seed`, `scenario_family`). Here it is flattened to a
bare string. **Whether it retains any provenance content is unknown and cannot be checked
without reading rows** — so it is a question only the maintainer can answer, and it is the
single cheapest thing that could unblock the design.

---

## What the same author publishes when documenting properly

The decisive evidence is **the author's own homologous sibling corpora**, which expose the
full generation ancestry.

### `agentic-prompt-injection-boundary-pairs` (same author, same agentic-injection line)

Its published `metadata/stats.json` states, verbatim:

```
total_rows: 1200 ·  pair_count: 600 ·  scenario_count: 50
```

Its `metadata/schema.json` **requires** per-record ancestry:
`scenario_id: ^scn_[0-9]{3}$` · `pair_id: ^pair_[0-9]{4}$` · `id: ^apibp_[0-9]{4}_[ab]$`

Its **generator is public** — `scripts/build_dataset.rb`, `BUILD_SEED = 20_260_713`,
running **12 hardcoded template functions CROSSED against 50 scenarios** from
`source/scenarios.jsonl`, with `variation = scenario_index % 5`, and **the split assigned at
the scenario level** by `split_for_scenario(index)`.

So the real ancestry is:

```
build seed -> scenario (50) x template family (12) -> variation -> pair -> row (1200)
```

**Template and scenario are crossed, not nested.** A `template_id` alone would not identify
the resampling unit. That is Branch D.

### `turkish-daily-dialogues-5k` (same author, Python era)

Publishes `source.scenario_family`, described verbatim as *"source family used for
group-aware splitting,"* with *"Splitting happens at the source scenario-family level"* and
*"The split is therefore intentionally not row-random."*

### And the study population's own card names both levels in one breath

> *"Splits (scenario-isolated — **no template leaks across splits**)"*

Two distinct units — scenario and template — in a single sentence, with the **scenario** as
the isolation unit.

### The author's own harness expects a key this corpus does not ship

`injectbench` (created 2026-08-11T23:37:35Z, minutes before this dataset) defines `pair_id`
as a first-class field and a `pairs_from()` function grouping benign/attack tuples. **The 5k
ships no `pair_id`, so that function silently skips every row.** The pairing demonstrably
exists in generation; the key was dropped at release.

---

## Quantified consequence, using the author's own published ratio

Anchor: **1,200 rows / 50 scenarios = 24 rows per scenario**, published by this author.

If the candidate's 2,750 attack rows expand at a comparable ratio, the independent unit
count is ~115, not 2,750 — and with near-deterministic classifiers on slot-variants
(intracluster correlation → 1) the effective *n* collapses to the cluster count:

| rows/parent | implied parents | effective *n* | min detectable \|Δ\| | vs the designed 0.020 |
|---|---|---|---|---|
| 1 | 2,750 | 2,750 | 0.017 | OK |
| 5 | 550 | 550 | 0.038 | 1.9× worse |
| 10 | 275 | 275 | 0.054 | 2.7× worse |
| **24** *(author's published ratio)* | **115** | **115** | **0.082** | **4.1× worse** |
| 50 | 55 | 55 | 0.115 | 5.8× worse |

**At the author's own ratio the frame is short by a factor of ~18** — the design's 2,025
*independent units* would require ~48,600 rows at 24× expansion, against 2,750 available.

> **Status: this is a quantified hypothesis, not a measurement.** The 24× ratio is a
> published fact about a *different* corpus. The candidate's true parent count is UNKNOWN
> and is precisely what the frozen question asks. Nothing here substitutes for the
> maintainer's answer, and nothing here may be cited as a property of the study population.

---

## The upstream lead is CLOSED

The card says the corpus was built *"in the spirit of garak, PyRIT and promptfoo."* The
predecessor's generator imports only `base64`/`digest`/`fileutils`/`json` and reads only the
author's own scenarios file. **Those projects supply no templates**, so ancestry cannot be
recovered from upstream. No third party has mirrored, forked, or analysed the corpus, and no
paper or DOI exists for it.

---

## Probe coverage — honest gaps

Four probes were launched; **two completed, two died on infrastructure errors** (machine
sleep, dropped response), not on logic.

- `probe:repo` — **lost, but fully superseded.** I performed the repository forensics
  directly and independently; the ninth-column finding above is mine, not an agent's.
- `probe:field` — **lost and not recovered.** This was the survey of whether publishing
  per-record parent identity is normal practice across public injection benchmarks. Its
  absence does not affect the branch determination; it would only have informed how
  unusual the request is. **Recorded as a gap, not silently dropped.**

---

## Consequence under the frozen branch table

Branch **D** ⇒ a `template_id` alone is insufficient; the required ancestry is a **tuple**:
build seed · scenario / scenario-family id · template / family id · variation index · pair id.

Branch **B** also holds at present ⇒ if that tuple cannot be supplied by the maintainer,
the population is **rejected for this experiment**, and it must **not** be reconstructed
algorithmically — clustering, embeddings, and lexical similarity are all inadmissible under
§4 of the frozen question.

**The study remains RED.** No inference is drawn from the absence of a public answer.
