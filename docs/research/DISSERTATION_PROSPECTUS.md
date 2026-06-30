# Dissertation Prospectus

## Working Title

Dependence-Aware Partial Identification for Composed AI Safety Systems

## Thesis

Layered AI safety systems cannot be evaluated by isolated guardrail scores
alone. When guardrail failures are dependent, composed risk is generally only
partially identified. This dissertation develops theory, algorithms, empirical
protocols, and proof-carrying artifacts that compute sharp risk intervals,
expose endpoint witnesses, propagate finite-sample uncertainty, search for
dependence amplification, and package claim-bounded evidence without claiming
deployment safety certification.

## Research Principles

- Identification before optimization: the composition event and feasible
  dependence class must be explicit before tuning or comparing a guardrail
  stack.
- Proof-carrying evaluation: reported intervals should carry labels, atom
  order, constraints, assumption hashes, query coefficients, tolerance, and
  endpoint witnesses.
- Finite-sample propagation: estimated singleton and pairwise rates should
  become simultaneous confidence intervals before they become composition
  constraints.
- Dependence stress testing: product coupling is a named baseline, not the
  default truth.
- Claim-bounded assurance: receipts preserve evidence integrity and scope; they
  do not certify real-world safety.

## Chapter Claims

1. Sharp Static Bounds: finite binary guardrail composition admits sharp LP
   bounds over atom distributions, with classical Frechet AND/OR formulas as
   special cases.
2. Finite-Sample Identification: Bernoulli count evidence can be converted into
   simultaneous interval constraints, yielding valid outer identified sets under
   the stated sampling assumptions.
3. Dependence Stress and Cliffs: fixed marginals can hide large composition
   swings, and stress tests should report interval width, endpoint witnesses,
   and product-baseline regret separately.
4. Empirical Guardrail Evaluation: deterministic benchmark protocols can ingest
   binary failure matrices from keyword, regex, semantic, LlamaGuard-style, and
   optional harness outputs without calling external model APIs.
5. Adversarial Dependence Amplification: red-team search should optimize
   co-failure overlap under bounded perturbations, held-out baselines, safety
   gates, and multiplicity-aware reporting.
6. Sequential and Agentic Monitoring: time-indexed guardrail failures require
   anytime-valid monitoring and path-level estimands rather than static batch
   claims alone.
7. Proof-Carrying Assurance Artifacts: reports, Merkle logs, receipts, and
   SACM/GSN-style cases can make evidence auditable while keeping non-claims
   explicit.

## Theorem Inventory

| Theorem | Status | Evidence |
| --- | --- | --- |
| Atom LP sharpness | Implemented | `cc.kernel.strict` and `tests/unit/kernel/test_sensitivity.py` |
| Classical Frechet special cases | Implemented | `artifacts/paper/table_1_classical_frechet_bounds.csv` |
| Monotonic tightening under valid added constraints | Implemented | `tests/unit/kernel/test_monotonic_tightening.py` |
| Endpoint witness verification | Implemented | `scripts/verify_paper_artifacts.py` |
| Product-coupling baseline label-order invariance | Implemented | `tests/unit/kernel/test_independent_probability_label_order.py` |
| Finite-sample interval propagation | Prototype | `cc.kernel.sample_complexity` count-to-constraint helpers |
| Sequential anytime-valid extension | Prototype | `cc.kernel.sequential` |

## Artifact Inventory

| Artifact | Status | Command or verifier |
| --- | --- | --- |
| Paper tables and figures | Implemented | `make reproduce-paper`; `make verify-paper-artifacts` |
| Minimal bounds and witnesses | Implemented | `artifacts/paper/minimal_bounds.json`; `minimal_witnesses.json` |
| Proof-context bundle and manifest | Implemented | `artifacts/paper/minimal_bundle.json`; `manifest.json` |
| CC report receipts | Implemented | `make test-reporting` |
| Dependence benchmark summary | Implemented smoke | `tests/integration/test_dependence_benchmark_example.py` |
| Active dependence red-team protocol | Planned | Requires held-out baseline and redaction policy |
| Semantic fault-line analysis | Planned | Requires slice labels or cluster IDs |
| Inspect-compatible ingestion | Planned optional | File-based adapter only, no external API calls |

## Empirical Protocols

- Static benchmark protocol: fix singleton marginals, vary feasible dependence,
  and report FH width plus independence regret.
- Finite-sample protocol: convert singleton and pairwise counts into
  simultaneous intervals, then solve composition bounds under those interval
  constraints.
- Red-team protocol: search for co-failure amplification while holding marginal
  movement within declared tolerances; evaluate on held-out prompts.
- Slice protocol: compute per-slice intervals for declared groups or clusters,
  apply multiplicity controls, and state bounded empirical claims only.
- Sequential protocol: monitor time-indexed failure events with anytime-valid
  tests and path-level composition estimands.

## Non-Claims

- No deployment safety certification.
- No guarantee of dataset representativeness.
- No causal conclusion without causal assumptions.
- No production-readiness claim for dashboards, cloud infrastructure, or vendor
  integrations.
- No claim that product coupling is the true joint law.

## Twelve-Month Milestone Calendar

| Month | Milestone |
| --- | --- |
| 1 | Freeze Paper 1 strict kernel API, artifact verifier, and release metadata. |
| 2 | Complete finite-sample constraint propagation examples and theorem text. |
| 3 | Produce dependence benchmark V1 with fixed marginals and varied couplings. |
| 4 | Submit Paper 1 reviewer package with one-command reproduction. |
| 5 | Build active dependence red-team protocol with deterministic smoke fixture. |
| 6 | Add semantic fault-line prototype and multiplicity-aware reporting. |
| 7 | Add optional Inspect-compatible file ingestion and adapter tests. |
| 8 | Draft dependence stress and cliff chapter with benchmark evidence. |
| 9 | Extend sequential monitoring experiments and document anytime-valid limits. |
| 10 | Package proof-carrying assurance receipts and SACM/GSN-style cases. |
| 11 | Integrate chapters, theorem ledger, artifact ledger, and limitations. |
| 12 | Dissertation polish, defense materials, and reproducibility audit. |
