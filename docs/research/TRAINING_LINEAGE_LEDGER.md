# Training lineage ledger — E2 candidates and population

**Compiled:** 2026-08-24, from primary sources only (model cards at pinned revisions,
repository trees, API metadata, associated papers). **No model was retrieved or loaded and
no dataset row was opened.**

## Classification scales

**Lineage class** — L0 complete sources+versions known · L1 large public portion disclosed,
some private possible · L2 some sources described, substantial remainder unknown ·
L3 composition materially unknown.

**Benchmark relation** — O1 documented exclusion (rare; requires strong evidence) ·
O2 no documented sharing (**not** proof of non-overlap) · O3 source families plausibly
intersect · O4 known inclusion · O5 undeterminable from public evidence.

> **O1 is never translated as "clean." O2 is never translated as "clean."**
> A repository publication date after a model release dates the **artifact**, not the
> **origin of its text**. The operative quantity is **EARLIEST PROVABLE AVAILABILITY**.

---

## Guard A — `securelayer7/PROMPTPurify` @ `8a035c79…` (tag v0.0.1, MIT)

**Lineage class: L3 — materially unknown. Benchmark relation: O3 — plausible intersection.**

| Field | Finding |
|---|---|
| Training data | *"Built from scratch by SecureLayer7 on curated internal corpora."* **That is the entire disclosure.** Zero datasets named, no counts, no sampling rule |
| Training cutoff | `NO PUBLIC EVIDENCE FOUND` across all docs at the pinned revision |
| Pretraining corpus | `l5e.json` points to `models/l5e/CORPUS_LICENSES.json` — **HTTP 404 at every revision and every mirror.** The only pointer to the only claimed pretraining disclosure resolves to nothing |
| Distillation | version string `v13-struq+supcon+distill` names a **distillation stage with no teacher identified anywhere**. Repo denials are precisely hedged — "no teacher weights are *redistributed*", "no teacher weights *shipped*" — neither denies **use** |
| Backbone | `/Users/ssk/SSK/projects/promptpurify/models/l5e/_pretrained/latest` — a **private local path** ending in a **mutable** pointer. The vendor's own build record does not pin its backbone |
| Architecture | 4-layer encoder, 13,723,140 params, INT8 ONNX (~13.6 MiB); "int8 ELECTRA-small"; 32k WordPiece + special `[UNTRUSTED]` token at id 32000 |
| Output | **Continuous** [0,1] softmax at config-attested `injection_logit_index=1` |
| Threshold | **No vendor-mandated threshold** — *"you decide the threshold and the policy."* Ships a **ternary** default: block ≥0.95, advisory ≥0.85 |
| Artifact integrity | **Genuinely strong.** `model.int8.onnx` = `6609fea5…` attested by `SHA256SUMS` and independently matched by the HF mirror's LFS oid — bit-identical across two channels |
| **Maintenance — CONTRADICTS THE RECORDED PASS** | The pinned revision is an **orphan commit**: GitHub's compare API returns *"No common ancestor between `8a035c79…` and `main`."* It has `parents = []`; `main` has its own disjoint root. Reachable **only** via the owner-mutable annotated tag `v0.0.1`, and **GC-eligible if that tag moves.** This is exactly the "silent-update path" the maintenance criterion requires be absent |
| **Feasibility — UNDISCLOSED IN THE SCREEN** | Its no-input loader **already failed** in E2-FIELD-TEST-001 three hours before the VNext preflight: `createL5eRunner()` resolved the model relative to its own npm install rather than the frozen workdir, because the frozen runner omitted the documented `modelDir` override |

**Contamination-relevant positive evidence, not mere absence:** `struq` denotes StruQ
(arXiv 2402.06363, USENIX Security 2025), whose training method is to augment
instruction-tuning data with examples placing instructions in the **data** portion and
fine-tune the model to ignore them. That construction is **structurally identical to the
study population's declared `rag_document_embedded` family.** `docs/HONEST-LIMITS.md` also
declares a **live feedback corpus** fed by a public adversarial CTF.

**Earliest provable availability:** Guard A's own artifacts, 2026-05-30. Relative to the
study population: **UNKNOWN and structurally undeterminable** — the guard publishes no
cutoff and the corpus publishes no per-record first-appearance evidence.

---

## Guard B — `patronus-studio/wolf-defender-prompt-injection-small` @ `eff31df5…` (Apache-2.0)

**Lineage class: L2 — partial. Benchmark relation: O3 — plausible intersection.**

| Field | Finding |
|---|---|
| Training data | **Unusually forthcoming** — enumerates ~26–36 named public corpora with links, plus an itemized augmentation recipe (unicode variants, homoglyphs, encodings, tag wrappers, leetspeak, spacing/case noise) — **then adds "internally generated examples"** with no manifest |
| Training cutoff | **NONE stated.** This single omission is what fails the strict gate |
| **Direct family overlap** | Four named training corpora sit **directly on the study's declared families**: `AlignmentLab-AI/agentcode` and `interstellarninja/tool-calls-multiturn` (`tool_function_abuse`); `alwaysfurther/agent-social-eng-attacks-sample` (`approval_authority_escalation`); `Microsoft/llmail-inject-challenge` (`rag_document_embedded`) |
| **Ontology contamination** | Training list includes **`S-Labs/prompt-injection-dataset` — the exact source this protocol REJECTED** for mixing jailbreak personas — plus `allenai/wildjailbreak`, `jackhhao/jailbreak-classification`, `Lakera/mosscap_prompt_injection`. Its decision boundary is fit to injection ∪ jailbreak and **cannot be un-fit by narrowing the frame** |
| Output | **Binary only.** Card declares benign(0)/injection(1); sole documented recipe is `argmax(logits)`. Metrics are F1/P/R/FPR at one fixed point — **no ROC, no AUC, no score semantics** |
| Artifact self-attestation | `config.json` has **no `id2label`, `label2id`, `num_labels`, or `problem_type`** — the artifact does not attest which logit is "injection" |
| **Not one detector** | The frozen tree ships `model.safetensors` (562 MB) **plus** fp16 ONNX **plus** mixed-precision ONNX **plus** an L2 cascade (`aggregator.onnx`, `promote_router.onnx`, four heads `{jayavibhav, mirror, necent, patronus}`) **using a different tokenizer** than the root. `cascade_config.json` declares `t_bert = 0.9` and `bert_calls: 884` — the L3 transformer is **router-gated**, invoked on a deliberately selected subset |
| **Memorization channel with a filename** | One cascade head is literally named **`jayavibhav`** — after a corpus in Guard B's own training list |
| Maintenance | **v2 shipped 2026-08-18** — the pinned revision was already superseded when the screen was dated 2026-08-24 |

**Earliest provable availability:** guard-side ingestion boundary is provable from the
pinned artifact; relative to the population, **UNKNOWN**.

---

## Population — `3nesdeniz/agentic-prompt-injection-5k` @ `e658d94d…` (CC-BY-4.0)

**Lineage class: L3 — materially unknown. Benchmark relation: O3.**
Full verdict in [`E2_VNEXT_ADMISSIBILITY_COURT.md`](E2_VNEXT_ADMISSIBILITY_COURT.md) §4.

| Field | Finding |
|---|---|
| Provenance | Human-designed attack templates, **"authored with slot variables and expanded deterministically with scattered mixed-radix sampling,"** then "deduplicated and **split-isolated by template**" |
| **Template count** | **DISCLOSED NOWHERE.** No count, no expansion factor |
| **Template identifier** | **ABSENT from the released schema** `[id, text, label, class, attack_family, technique, severity, language]`. The author used template identity as the split-isolation key and **withheld it from the release** |
| Repo created | 2026-08-11T23:43:28Z — **after** both model revisions, which dates the **repository, not the text** |
| Label semantics | `label=1` is the **author's assertion** that the text represents an injection. It is **not** an attack-success oracle against any application |
| Attack base rate | 2,750 / 4,935 = **55.7%** — no relation to any deployment prevalence |
| Declared lineage | "in the spirit of garak, PyRIT and promptfoo" — families public in the literature since ~2022–23 |

**Earliest provable availability of the underlying template text: UNKNOWN.** Deterministic
expansion of long-public templates can reproduce strings predating every candidate guard.
**The out-of-sample gate fails on the corpus side, not only the guard side — so no guard
swap can clear it.**

---

## Adjudication — what unknown overlap actually invalidates

| Intended claim | Lineage verdict | Note |
|---|---|---|
| **Descriptive** — these frozen systems, this frozen frame | **IRRELEVANT** | a census is a measurement, not an inference; contamination changes what the number means, not whether it is the number |
| **Generalization** to unseen injections | **FATAL** | affirmatively contra-indicated, not merely unsupported |
| **Mechanistic** — shared failure ⇒ shared vulnerability | **FATAL ×3** | (a) shared public injection vocabulary + StruQ; (b) template expansion breaks it even with perfect lineage; (c) harness asymmetry manufactures Cov(A,B) from item length |
| **Deployment-rate** estimation | **LIMITING only because already dead** | protocol already forbids it, correctly |

**Therefore: unknown lineage is a LIMITATION on E2, not its blocker.** The narrow
descriptive claim survives it under an explicit no-generalization,
no-mechanistic-attribution scope. What blocks E2 is the empty runtime lock, the
unidentified unit of analysis, and the absent common event.

## Field-wide finding

Of **24** candidate guards screened on primary sources, **zero** publish record-level
training data **and** a training cutoff. One (`leolee99/PIGuard`, ACL 2025, MIT) publishes
its actual training rows as a file — the highest transparency found anywhere in the field.
One (`guardion/ModernGuard-1`) publishes a training window; its data is proprietary.

**The strict out-of-sample gate is therefore unsatisfiable by the entire category, and the
E2 stop must not be repaired by substitution.** The gate also discriminates on
**disclosure rather than independence**: a vendor publishing a manifest showing 40% overlap
would pass, while a vendor with no disclosure and genuinely zero overlap fails.
