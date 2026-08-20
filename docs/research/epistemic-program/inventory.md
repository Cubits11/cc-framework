# Epistemic Inventory

## How to read this

One row per artifact that contains an actual intellectual move. The
qualifying question is **not** "is this impressive enough?" It is:

> What question did this artifact attempt to answer?

Small things count. A test is an artifact. A refusal is an artifact. A failed
idea is an artifact, and often a better one than a success.

The `Limit` column is mandatory. An artifact with no stated limit has not been
inventoried; it has been advertised.

## Kernel and mathematics

| Artifact | Question | Claim | Evidence | Limit | Interesting because | Public |
| --- | --- | --- | --- | --- | --- | --- |
| `cc.kernel.sensitivity` finite-atom LP | What does evidence about guardrail failures identify when the joint law is unknown? | Sharp lower and upper values for a declared linear composition query over feasible finite binary atom distributions. | `tests/unit/kernel/test_sensitivity.py`, monotonic tightening tests, theorem ledger T1. | Says nothing about upstream data collection, semantic safety, or representativeness. | The answer is an interval, and its width is itself the finding. | Research |
| `cc.kernel.frechet_classes` | What do classical Fréchet-Hoeffding bounds give for composed binary failure? | Bounds under exact singleton marginals with no side constraints. | `test_classical_frechet_special_cases.py`, theorem ledger T2. | Assumes nothing about dependence, and therefore proves nothing about a deployment. | The independence calculation is exposed as a *baseline*, never as truth. | Research |
| `cc.kernel.sample_complexity` | When counts replace probabilities, what survives? | Under stated iid Bernoulli sampling and simultaneous moment coverage, count-derived intervals form an outer confidence interval. | `test_finite_sample_constraints.py`, `test_sample_complexity.py`, theorem ledger T6. | Invalid after uncorrected adaptive target selection. Not a deployment certificate. | Names the exact condition under which its own guarantee dissolves. | Research |
| Endpoint witnesses | Is a reported bound actually attainable? | The returned joint law attains the reported endpoint under declared constraints. | Witness verification table in the paper artifacts. | Witnesses the mathematics of the bound, not any real system's behavior. | A bound that cannot exhibit a witness is not reported. | Research |
| `cc.kernel.cliff` | Do composed risks jump discontinuously as dependence changes? | Characterizes dependence-driven jumps under declared models. | Correlation-cliff experiments under `experiments/`. | Exploratory. Not a confirmatory certificate. | The failure mode is a cliff, not a slope — averages hide it. | Experimental |

## Evidence governance

| Artifact | Question | Claim | Evidence | Limit | Interesting because | Public |
| --- | --- | --- | --- | --- | --- | --- |
| `cc.evidence.role_ontology` | Should all evidence be allowed to say the same things? | Role determines what a piece of evidence may support. | `tests/unit/evidence/test_role_ontology.py`. | A correct role assignment does not make the underlying measurement correct. | Typing evidence is a stronger idea than scoring it. | Research |
| `cc.evidence.confirmatory_protocol` | When does an exploratory finding become a confirmatory one? | Only under a separately registered protocol; never by reinterpretation. | `test_confirmatory_protocol.py`. | Registration does not validate. It only prevents silent revision. | It makes the exploratory/confirmatory boundary machine-checkable. | Research |
| `cc.evidence.decay` | Do claims expire? | A claim carries a policy defining when it must be rechecked, degraded, or expired. | `tests/unit/evidence/` decay tests. | Does not establish that the system is currently safe. It defines when to look again. | Treats claims as mortal by construction. | Research |
| `cc.evidence.merkle_log` / `anchoring` | Can a reviewer detect a silent rewrite of history? | Tamper evidence over recorded bytes under canonical serialization. | `test_transparency_log_adversarial.py`. | Integrity only. Never statistical validity, never safety. | The adversarial test is the artifact; the log is just the subject. | Research |
| `cc.evidence.claim_governance` | Can a claim package be checked for internal consistency? | PASS means internally consistent under verifier rules. | `test_claim_governance.py`, capsule integration. | PASS is not deployment safety. The verdict says so in its own payload. | The success message carries its own limitation. | Research |
| `cc.reporting.report` strict models | Can a report be repaired quietly into validity? | Fail-closed validation: missing boundaries, extra fields, and reserved overclaim vocabulary are rejected. | `tests/unit/reporting/test_reporting.py` round-trip and rejection tests. | Schema validity is not semantic truth. | The word list means a report literally cannot say "proves". | Research |
| Deterministic claim-governance capsule | Can a whole governance chain be reproduced byte for byte? | Regeneration from declared inputs reproduces the manifest exactly. | `tests/integration/test_claim_governance_capsule.py`. | **Reproducibility from declared inputs is not measurement.** See [CH-001](challenges.md). | The strongest guard in the repository, and its boundary is now demonstrated rather than asserted. | Public |

## Boundary and refusal artifacts

These are the ones a portfolio would omit. They are the most characteristic.

| Artifact | Question | Claim | Evidence | Limit | Interesting because | Public |
| --- | --- | --- | --- | --- | --- | --- |
| `docs/research/NON_CLAIMS.md` | What will this project never say? | Ten enumerated non-claims with reasons and substitutes. | The document, plus verifier-mandated boundaries it lists. | Cannot prevent bad-faith actors from overclaiming outside the framework. | A repository that publishes its own refusals is unusual. | Public |
| `CLAIM_BOUNDARY_MANIFEST` + validator | Can public language be checked against evidence? | Every active claim maps to a level, lane, support, test, and non-claim. | `tests/unit/docs/test_claim_boundary_manifest.py`. | Documents boundaries; creates no authority. | Forbidden *upgrades* are enumerated, not just forbidden words. | Public |
| Paper-core language quarantine | Can legacy vocabulary leak into the paper? | Named terms cannot appear in paper-core texts. | `tests/unit/kernel/test_paper_core_language_quarantine.py`. | Only covers the listed files and terms. | A unit test whose subject is prose. | Public |
| Generated-artifact boundary checker | Which outputs may be tracked as evidence? | Runtime, fixture, archive, and release artifacts are separated and enforced. | `scripts/check_artifact_boundary.py --static`, CI. | Static rules; does not judge artifact quality. | Stops generated results from quietly becoming evidence. | Public |
| `cc_max` deprecation warning | What happens to a metric that outlived its justification? | Legacy exploratory metric, not a partial-identification claim, not evidence of safety. | The `FutureWarning` raised at call time. | Preserved for compatibility, which is itself a risk. | The code apologizes for itself at runtime. | Public |

## Communication and interface

| Artifact | Question | Claim | Evidence | Limit | Interesting because | Public |
| --- | --- | --- | --- | --- | --- | --- |
| Claim Observatory World V3 | Can epistemic constraints be spatial? | Eleven chambers mapping claim integrity stages to physical metaphors. | Blender scenes, renders, `visual_world_manifest_v3.json`, world bible. | A teaching environment. Not a dashboard and not a certification surface. | Law 10: the world must make overclaiming uncomfortable. | Public |
| *Before You See It* (15s film) | Can epistemic restraint be compelling? | None. It is a statement of method. | The film itself. | Establishes no technical claim; labels its own unverified frame. | It says INCONCLUSIVE on purpose, in its own advertising. | Public |
| Canonical page | Can a visitor encounter the philosophy and test it in the same minute? | The embedded bytes hash to the digest recorded in the capsule manifest. | Live WebCrypto recomputation in the visitor's browser. | Integrity only. Says nothing about whether the number in the file is true. | The claim can actually fail in front of the reader. | Public |
| Drift guard | Can an honest screen go stale and start lying? | The page's embedded fixture and digest match the capsule. | `tests/unit/docs/test_canonical_page_fixture.py`. | Enforces internal consistency, not authenticity. Demonstrated by [CH-001](challenges.md). | The guard's own limit was found by attacking it, not by reasoning about it. | Public |
| Source ledger | What may be inferred from a narrative? | Nothing, until translated, predeclared, and observed. | The ledger itself, applied to a real source. | Does not establish that its source is accurate or inaccurate. | It takes the story seriously enough to test it, and refuses to mock it. | Public |
| Eight-week plan + prompt library | Can a method be handed to someone else? | A schedule with gates, and eleven prompts that produce artifacts rather than predictions. | The documents. | Completing them produces no certainty. | Every week has a stop condition. | Public |

## Adjacent, institutionally separate

| Artifact | Question | Relationship |
| --- | --- | --- |
| Ghost-Ark | Can a hostile reviewer independently determine what a system does and does not prove — by replaying receipts, inspecting malicious corpora, and validating evidence windows? | Informs the thinking. Not a Cubits11 product, not a university endorsement, not a commercial credential. Referenced only as a labelled research case study. |

## What the inventory shows

Ninety Python modules, one hundred and six test files, ninety-six documents —
and the through-line is not any of them. It is that a surprising fraction of the
artifacts exist **to prevent a conclusion**, not to produce one: a quarantine on
vocabulary, a boundary on generated files, a warning attached to a metric, a
verdict that carries its own caveat, a film that labels its own footage.

That is the invariant. It is developed in the [research graph](research-graph.md).
