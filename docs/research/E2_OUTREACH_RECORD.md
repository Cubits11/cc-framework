# Outreach record — SENT

**Status: POSTED 2026-08-25T13:41:48Z, on explicit human authorization.**

| | |
|---|---|
| Thread | [discussions/1](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k/discussions/1) |
| Title | Provenance question: is there a per-record scenario / generation identifier? |
| Posted by | `Cubits11` (Pranav Bhave) — identity verified against `/api/whoami-v2` before submission |
| Venue | HuggingFace Community discussion on the dataset repo — the first and only thread on it |
| Body | exactly the text below, verified on screen before submission; unmodified since |

**Response status: AWAITING REPLY. No answer received. The study remains RED.**

Per [`E2_METADATA_QUESTION_FROZEN.md`](E2_METADATA_QUESTION_FROZEN.md) §5, silence and
ambiguity both map to **remain RED**, and **no inference is drawn from non-response**. The
maintainer has never participated in a discussion on any of their repositories, so silence
is the modal expected outcome and must not be read as evidence of anything.

---

## Before reading the draft: four facts that should inform the decision

1. **Response base rate is low.** The maintainer's HF profile reports `numDiscussions: 0`.
   Across all eight personal datasets, one Space, and two AltaySec org datasets there are
   exactly four threads — **all automated parquet-converter bots**, all with
   `repoOwner.isParticipating = false`. The one inbound human request on record (GitHub
   issue #5 on their profile repo, opened 2026-08-14) sat **unanswered for 10 days** while
   the author actively pushed code elsewhere. Plan for silence.

2. **Scoop risk is real and was not previously on the table.** The maintainer is an active
   researcher in exactly this area — AltaySec CTO, publishes prompt-injection corpora, and
   shipped `injectbench`, a companion evaluation harness, minutes before this dataset.
   Disclosing that a preregistered *dependence* study is evaluating this corpus hands a
   direct competitor strategic information for free. **The draft below therefore discloses
   no estimand, no preregistration, no power analysis, and no blocker.**

3. **It is a one-way door.** The thread cannot be deleted by us. Edits leave a public
   edit-history link. The maintainer can hide comments **irreversibly**, lock the thread, or
   disable discussions. It would be the first and only thread on the repo — permanently
   top of the Community tab for every future visitor.

4. **It is legible as public criticism of a third party's work.** Even worded neutrally.
   Weigh that against a private route (LinkedIn / altaysec.com.tr — no public email exists),
   accepting the real trade-off: **a private answer is much weaker evidence**, because it is
   not publicly verifiable, timestamped, or citable in the study record.

---

## Draft — minimal disclosure (recommended)

> **Title:** Provenance question: is there a per-record scenario / generation identifier?

> Hi — thanks for publishing this corpus, and for putting a CITATION.cff and a clear
> attack-family taxonomy on it.
>
> I'm looking at using this dataset for an evaluation and I'm trying to get the
> provenance right before I do anything with it. The card says items were authored with
> slot variables and expanded deterministically, and that splits are
> "scenario-isolated — no template leaks across splits."
>
> That suggests records may share an underlying scenario or template parent. My question
> is just whether that parent is recoverable from the release:
>
> 1. Is there a stable per-record identifier linking each row to its scenario / template /
>    seed — analogous to the `scenario_id` and `pair_id` you publish in
>    `agentic-prompt-injection-boundary-pairs`? Or a published count of distinct
>    scenarios or templates, like the `scenario_count` in that dataset's
>    `metadata/stats.json`?
> 2. The released parquet has a `source` column that isn't in the card's schema table.
>    In `turkish-daily-dialogues-5k` the equivalent field carries generation metadata
>    (`scenario_family`, seeds). Does `source` here retain anything like that, or is it a
>    flat provenance string?
>
> If the generation doesn't work that way — e.g. records are independently parameterised
> with no stable shared parent — that's genuinely just as useful an answer for me, and I'd
> rather know than assume. I'm not asking for the templates or the generator, only for
> whatever ancestry metadata already exists (or confirmation that none does).
>
> Either way, thanks for making the corpus available.

---

## Why it is worded this way

| Choice | Reason |
|---|---|
| Asks about **scenario / template / seed**, not `template_id` | The probe returned **Branch D** — the unit sits *above* the template. Asking for a `template_id` would have been the wrong ask and would have anchored the answer at the wrong level |
| Cites **their own** `boundary-pairs` and `turkish-daily-dialogues-5k` | Asks only for something they already publish elsewhere. Concrete, low-effort, and implicitly respectful of their better practice rather than critical of this release |
| Explicitly welcomes *"no shared parent"* | The frozen question requires that Branch C be as easy to give as any other. A request that signals a preferred answer produces a worthless answer |
| Names the undocumented `source` column | The single cheapest thing that could resolve the design, and unanswerable by us without reading rows |
| Says *"I'm not asking for the templates or the generator"* | Keeps the ask narrow, avoids requesting anything the author may consider proprietary, and keeps us clear of row content |
| Mentions **no** estimand, dependence, covariance, power, preregistration, or blocker | Scoop risk (§2 above) |
| No deadline, no urgency, no pressure | Nothing entitles us to a reply from a volunteer maintainer |

## What must NOT be added if this is revised

- Any mention of Δ, covariance, dependence, joint failure, or paired detector outcomes.
- Any statement that a study is blocked or that an answer would unblock it.
- Any request for record text, examples, previews, or the templates themselves.
- Any framing implying the dataset is defective.
- Any deadline or follow-up threat.

## Recording rule once (and if) an answer arrives

The answer is recorded against the **sealed** branch table in
[`E2_METADATA_QUESTION_FROZEN.md`](E2_METADATA_QUESTION_FROZEN.md) §5 **without amendment**.
Silence, partial answers, and ambiguity all map to **remain RED**. No inference is drawn
from non-response.
