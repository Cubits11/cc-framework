# E1 — Epistemic Decision Record

> Not a system that accumulates claims. A system that forces every claim to
> declare which world it belongs to, what evidence carries it, what could defeat
> it, and what must happen when reality disagrees.

**Decision: Narrow.** Companion to the frozen wager in
[E1_DEPENDENCE_EVIDENCE_STUDY.md](E1_DEPENDENCE_EVIDENCE_STUDY.md). Successor
contract: [E2_MEASUREMENT_CONTRACT.md](E2_MEASUREMENT_CONTRACT.md).

Every quantitative statement in this record is regenerated from
`artifacts/empirical/e1/study.json` and was re-checked against that artifact
before publication. Where a claim could not be verified, it is marked
**unverified** rather than softened.

---

## 1. Verdict

E1 is complete as a controlled synthetic dependence-evidence study. The claim
survived, but only at the mathematical, implementation, and generator-scoped
simulation layers. The evidence does not cross into real guardrail measurement,
deployment safety, adaptive robustness, or operational decision value.

The strongest justified conclusion:

> Within the declared finite binary setting and the tested synthetic
> generators, CC-Framework computes witness-carrying identified intervals under
> progressively richer moment evidence. The study constructively demonstrates
> that identical singleton and pairwise moments can remain compatible with
> different three-way failure probabilities.

The strongest **unjustified** conclusion would be:

> CC-Framework has validated the safety, robustness, or practical value of
> composed real-world guardrails.

**Narrow is not a retreat.** It is the correct result: the experiment located
the boundary between what the evidence establishes and what remains unknown.

## 2. What E1 actually reached

A research claim can fail at several layers. Passing an earlier layer confers
no authority at a later one.

| Claim layer | E1 status |
| --- | --- |
| Formal specification | Supported |
| Optimization semantics | Supported over the tested E1 scope |
| Witness validity | Supported — 20/20 regimes carry verified endpoint witnesses |
| Implementation fidelity | Supported by the declared tests and an independent LP |
| Synthetic behaviour | Supported |
| Finite-sample simulation | Supported, generator-scoped |
| Measurement validity | **Untested** |
| Population validity | **Untested** |
| Instrument validity | **Untested** |
| Adaptive robustness | **Untested** |
| Decision value | **Untested** |
| External reproducibility | **Untested** |
| Cross-context generalization | **Untested** |
| Deployment persistence | **Untested** |

E1 advances the first six rows and supplies no evidence for the rest.

## 3. Controlled results

"Truth" below means the exact probability under the frozen synthetic generator.
It does not mean real-world truth.

| Scenario | Truth | B0 product | I0 | I1 | I2 | I3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S1_independent_reference` | 0.125 | 0.125 | [0, 0.5] | [0, 0.25] | [0, 0.25] | [0.125, 0.125] |
| `S2_common_cause` | 0.5 | 0.125 | [0, 0.5] | [0, 0.5] | [0.5, 0.5] | [0.5, 0.5] |
| `S3_mutually_exclusive` | 0 | 0.015625 | [0, 0.25] | [0, 0] | [0, 0] | [0, 0] |
| `S4_parity_even` | 0 | 0.125 | [0, 0.5] | [0, 0.25] | [0, 0.25] | [0, 0] |
| `S4_parity_odd` | 0.25 | 0.125 | [0, 0.5] | [0, 0.25] | [0, 0.25] | [0.25, 0.25] |

All 20 regime results report `solver_status: optimal`, valid lower and upper
endpoint witnesses, and containment of the declared generator truth.

### Product coupling is not directionally safe

Under `S1`, `B0 = 0.5 × 0.5 × 0.5 = 0.125` exactly matches the generator. That
is a positive control for the assumption B0 makes — not a licence to use it
elsewhere.

Under `S2`, the generator truth is `0.5` while `B0 = 0.125`: the product
baseline is **one quarter** of the real value. Under `S3`, the generator truth
is `0` while `B0 = 0.015625`: the same baseline now invents risk that is absent.

> Product coupling is neither inherently conservative nor inherently
> optimistic. Its direction of error depends on the unestablished joint
> structure.

### All pairwise evidence can still be insufficient

`S4_parity_even` and `S4_parity_odd` share **identical** singleton moments
(`0.5, 0.5, 0.5`) and **identical** pairwise moments (`0.25, 0.25, 0.25`), yet
assign three-way probabilities `0` and `0.25`. Under I2 both remain compatible
with `[0, 0.25]`.

> In the declared three-variable binary model, complete pairwise moment
> information need not identify a three-way event.

This defeats the inference *"we measured every pair, so we know the stack."* It
does **not** establish that pairwise evidence is useless: under `S2` and `S3`,
I2 point-identifies the target. The correct statement is conditional — pairwise
evidence may identify a higher-order event in some feasible regions, but not in
general.

## 4. Finite-sample result

The frozen grid contains 60 cells: 5 laws × 2 events × 3 regimes (I0/I1/I2) ×
2 sample sizes, at 128 replicates each. All 60 passed the predeclared rule.

```text
minimum observed coverage:       0.984375   (= 126/128)
minimum Wilson 95% lower bound:  0.944819
nominal coverage:                0.90
frozen acceptance floor:         0.84       (= nominal − 0.06)
seed: 20260821    delta: 0.10    replicates: 128    n ∈ {64, 256}
```

Within that exact generator grid, sample-size grid, evidence construction,
seed, and implementation, the simulated outer intervals met the frozen
criterion. It establishes nothing about arbitrary binary joint laws, arbitrary
missingness, adaptive model selection, post-hoc threshold tuning, non-IID
samples, drifting populations, real-world label correctness, or adaptive
attackers.

### On the precision of `0.984375`

An earlier draft of this record warned that `0.984375` "must not be presented
as six-decimal empirical knowledge." **That warning was wrong, and it is
withdrawn.**

`0.984375` is exactly `126/128` — an exact dyadic rational arising from 128
replicates. It is not a rounded estimate, and writing it to six places is the
most faithful available representation, not false precision. Rounding it would
have *destroyed* information.

The same draft listed seven items a report "should disclose." Six were already
disclosed in the artifact before the criticism was written:

| Disclosure | Where |
| --- | --- |
| Replicates per cell | `design.replicates = 128` |
| Nominal coverage target | `design.nominal_coverage = 0.9` |
| Interval construction | `design.confidence_method` |
| Exact grid | `design.sample_sizes`, `coverage_rows` |
| Seed | `design.seed = 20260821` |
| Covered count per cell | `coverage_rows[].covered` |
| Rationale for the `0.84` floor | `design.coverage_decision_rule` (nominal − 0.06) |

This is recorded rather than silently fixed because the failure mode is
instructive: a critique aimed at the wrong level is itself a claim that
exceeded its evidence. "Precision" can mean numerical precision, estimator
uncertainty, reporting precision, or epistemic specificity of the claim. Those
must never be conflated — see the record ladder in
[E2_MEASUREMENT_CONTRACT.md](E2_MEASUREMENT_CONTRACT.md#the-four-level-record-ladder).

The acceptance floor remains a protocol decision, not a theorem. Sixty grid
cells are 60 selected design conditions, not 60 independent validations of
arbitrary future use.

## 5. Strongest challenge result

The parity construction is the strongest result in E1 because it attacks the
framework's own possible future rhetoric rather than testing a happy path. It
prevents the repository from ever implying that measuring all pairwise
dependence determines the system-level event.

It is also the result least dependent on our implementation: the same
conclusion follows from an independently written eight-atom LP. That matters,
because a framework can trivially look useful when only its own code produces
the desired conclusion. Separating

```text
phenomenon   ≠   framework implementation artifact
```

is exactly what an evaluation framework must do to be believed.

## 6. Strongest surviving objection

E1 has not shown that its binary variables correspond to valid measurements of
meaningful guardrail failure. **This objection dominates every remaining
technical refinement.**

A mathematically correct identified interval can still be decision-useless when
the event is operationally misdefined, labels are unreliable, the dataset is
unrepresentative, guardrails are evaluated on different items, outputs are
missing non-randomly, thresholds were tuned after inspecting results, nominally
distinct guardrails share a base model, the attack population is too weak, or
no decision-maker has a loss function under which interval narrowing matters.

The next frontier is therefore not a larger LP. It is the world-to-measurement
bridge, which is why [E2](E2_MEASUREMENT_CONTRACT.md) is frozen before any
dataset is chosen.

## 7. What changed

| Path | Role |
| --- | --- |
| `src/cc/evals/dependence_benchmark.py` | E1 scenarios, evidence-regime analysis, artifact I/O |
| `docs/research/E1_DEPENDENCE_EVIDENCE_STUDY.md` | Frozen wager contract |
| `artifacts/empirical/e1/{study.json,coverage.csv,manifest.json}` | Regenerable, hashed artifacts |
| `scripts/reproduce_e1_dependence_evidence.py` | Replay entry point |
| `scripts/verify_e1_dependence_evidence.py` | Verifier entry point |
| `make reproduce-empirical-e1 / verify-empirical-e1 / test-empirical-e1` | Reproduction surface |

### Test surface, stated exactly

An earlier draft of this record listed seven categories of "new tests." **That
was an overstatement and is corrected here.** The actual counts:

| File | Tests | Covers |
| --- | ---: | --- |
| `tests/unit/evals/test_dependence_evidence.py` | 3 | parity oracle, malformed-design rejection, tamper detection |
| `tests/unit/evals/test_e1_artifact_identity.py` | 10 | relocation invariance and mutation sensitivity of artifact identity |
| `tests/integration/test_e1_dependence_evidence.py` | 1 | clean reproduce-and-verify round trip |

Endpoint-witness validation, parity checking, CSV/JSON cross-consistency, and
manifest hashing are capabilities of `verify_e1_artifacts`, exercised through
those tests; artifact-boundary enforcement lives in
`scripts/check_artifact_boundary.py`. They are real, but they are not
additional test functions, and the record should not have implied a larger test
surface than exists.

The numerical difference is small. The rhetorical difference is not: if a study
says it performed *N* tests when the evidence supports *N − k*, the correct
move is to correct it, because the invariant

```text
claim strength  ≤  verified evidence strength
```

applies to the record's description of itself, not only to its results.

## 8. Verification record

Verified in the local environment at the time of writing:

- E1 artifact verification (`verify_e1_dependence_evidence.py`, exit 0);
- deterministic replay: `study.json` and `coverage.csv` regenerate **byte-identical**;
- relocation invariance: identical `manifest_payload_sha256` across output roots;
- tamper detection: a single flipped byte yields exit 1 and two independent errors;
- 14 focused E1 tests pass (3 + 10 + 1);
- endpoint witnesses valid and containment holds for all 20 regimes;
- independent eight-atom LP reproduces every interval endpoint and every B0 value;
- `ruff check`, `ruff format`, `isort`, `black` clean on changed files;
- `mkdocs build --strict` passes.

The complete repository suite was also verified, but only on the second
attempt: **958 collected tests, all passing, with exactly seven skips** — all
optional-dependency or opt-in gates (`CC_RUN_EXPERIMENTS`, `CC_RUN_PERF`,
guardrails, fastavro, protobuf ×2, SQLAlchemy).

The first attempt aborted on `OSError: [Errno 28] No space left on device` in an
unrelated enterprise smoke test. That was an environment limitation, not a
result. The rerun also exposed one genuine failure worth recording: the
committed evidence cards pinned the pre-upgrade `README.md` size and hash and
had to be regenerated. Only three values moved across the eight cards and the
site manifest — README's byte count, README's sha256, and `source_revision` —
with no change to any verdict, evidence state, assumption, or claim text.

This supports *"the declared E1 checks passed in the tested local
environment."* It does **not** support *"the result has been independently
reproduced."* A single investigator using a second local environment is still
an internal replay. External reproduction requires an outside investigator, an
independently controlled environment, or an independently written
implementation.

## 9. Claim ledger

| Claim | Status | Boundary |
| --- | --- | --- |
| The E1 artifact set replays and verifies. | Supported within scope | Internal environments only |
| Endpoint witnesses satisfy the declared constraints. | Supported within scope | 20/20 tested regimes |
| The implementation agrees with an independent LP. | Supported constructively | Declared finite binary construction |
| Equal singleton and pairwise moments can coexist with different three-way probabilities. | Supported constructively | Three-variable binary model |
| Artifact identity is invariant under relocation. | Supported | Manifest schema v2 |
| Complete pairwise evidence always identifies the all-three event. | **Contradicted** | Defeated by the parity pair |
| Product coupling is always conservative. | **Contradicted** | `S2` and `S3` show opposite error directions |
| Product coupling is always optimistic. | **Contradicted** | Same controls |
| The finite-sample procedure met the frozen simulation rule. | Simulation-supported | 60-cell generator grid |
| The finite-sample result generalizes to arbitrary joint laws. | **Untested** | No such experiment or theorem |
| The binary variables validly measure real guardrail failure. | **Untested** | No real measurement study |
| The benchmark population represents deployment. | **Untested** | Synthetic IID only |
| CC improves real guardrail safety. | **Not supported** | No intervention or field evidence |
| CC predicts adaptive attack success. | **Not supported** | No adaptive attacker implemented |
| Interval narrowing is worth its measurement cost. | **Untested** | No value-of-information analysis |
| The results are independently reproduced. | **Untested** | Internal replay only |
| The full repository suite passes. | Supported within scope | 958 tests, 7 optional skips, single local environment |
| Artifact hashes prove empirical truth. | **False by category** | Hashes establish identity, not validity |

## 10. The three bridges

The ecosystem is not one pipeline. It contains three logically independent
bridges, and E1 crosses only the middle one.

| Bridge | Question | E1 |
| --- | --- | --- |
| **I — World to measurement** | What does the binary variable mean, and why should anyone believe it tracks a real failure? | Not crossed |
| **II — Evidence to identified claim** | Given the measurements and assumptions, what joint claim is supported? | **Crossed** |
| **III — Identified claim to bounded action** | What should a person or institution do with the interval? | Not crossed |

A mathematically sharp interval can be operationally useless. A wide interval
can be extremely useful when it defeats an unjustified certification or reveals
that current evidence cannot support a decision.

## 11. World-expansion gates

| Gate | Permitted claim once passed | Status |
| --- | --- | --- |
| **A — Synthetic correctness** | The implementation behaves as declared over the tested synthetic scope. | **Reached** |
| **B — Measurement bridge** | A real, bounded evaluation population was measured under this protocol. | Contract frozen, closed |
| **C — Independent replay** | At least one outside investigator reproduced the declared result. | Open |
| **D — Adaptive challenge** | The result survived or failed under this exact adaptive attacker. | Open |
| **E — Decision value** | The additional evidence changed or clarified this bounded decision. | Open |
| **F — Cross-context replication** | The result recurred across these declared contexts. | Open |
| **G — Longitudinal persistence** | The result persisted over this declared interval and update history. | Open |
| **H — Public standard** | The method is mature enough for broader public reuse within explicit boundaries. | Open |

Gate A's forbidden expansion is *"the framework is validated for real
guardrails."* Nothing in E1 licenses it.

## 12. Corrections to the first draft of this record

Recorded rather than silently amended, because a system that claims to control
claim expansion must apply that discipline to itself.

| # | Draft statement | Correction |
| --- | --- | --- |
| 1 | `0.984375` is spurious six-decimal precision. | It is exactly `126/128`. The criticism was aimed at the wrong level and is withdrawn. |
| 2 | The report should disclose seed, replicates, nominal coverage, grid, covered counts, and the `0.84` rationale. | All six were already disclosed in `study.json` before the criticism was written. |
| 3 | Seven categories of new tests were added. | 14 test functions across three files. Verifier capabilities were miscounted as tests. |
| 4 | The complete `pytest -q` suite passed with seven expected skips. | Unevidenced when written — that run aborted on disk exhaustion. Since confirmed: 958 tests, exactly seven skips. The statement was true, but it was not verified at the time it was made, which is the same failure as an overclaim. |
| 5 | The artifact set is manifest-hashed and replayable. | True, but the manifest digest was path-dependent until it was fixed; external replay would have appeared to fail. |

## 13. Public research summary

> E1 tested what dependence evidence can and cannot justify in a controlled
> three-component binary system. It compared a product-coupling baseline
> against intervals supported by singleton, selected pairwise, all-pair, and
> full-joint information. The product baseline matched the target only under
> the independent generator; it understated the common-cause case, overstated
> the mutually exclusive case, and was therefore not directionally safe without
> a coupling assumption.
>
> The central counterexample uses even- and odd-parity distributions. They
> share every singleton and pairwise moment but assign different probabilities
> to simultaneous three-component failure. Complete pairwise evidence therefore
> does not generally identify a three-way event.
>
> CC-Framework reproduced the frozen synthetic artifacts, verified endpoint
> witnesses, and met the study's generator-scoped finite-sample simulation
> rule.
>
> E1 does not evaluate real guardrails, deployment safety, measurement quality,
> representativeness, or adaptive robustness. The next gate is a licensed,
> shared-item guardrail pilot with frozen labels, versions, thresholds,
> missingness rules, and row-level joint outcomes.

## 14. Final judgment

E1 did not validate an assurance system. It validated a narrower and more
foundational discipline:

> When evidence leaves multiple joint worlds possible, the system can preserve
> those worlds instead of collapsing them into a convenient point estimate.

The parity construction is the intellectual centre. Relocation-invariant
artifact identity is the engineering centre. **Narrow** is the epistemic
centre.

The next advance will not come from adding synthetic scenarios until the result
looks unbreakable. It will come from entering the world where failures are
contested, labels are imperfect, datasets are selective, guardrails share
hidden dependencies, attackers adapt, decisions have asymmetric costs, and
every public claim must eventually expire unless reality renews it.

> **Not a system that asks the world to trust its rigor. A research organism in
> which every claim must earn passage from formal possibility, through valid
> measurement and hostile replay, into bounded action — and can lose that
> passage whenever new evidence arrives.**
