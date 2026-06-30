# CC-Framework Discovery Report

Date: 2026-06-27

Scope: research-discovery pass only. No feature implementation was performed.
This report is grounded in repository inspection, existing validation commands,
existing scripts, existing artifacts, and observed command results from this
checkout.

## 1. Executive Summary

This repository's strongest contribution is an implemented finite-binary
partial-identification kernel for composed guardrail failures: declare binary
failure variables, encode marginal/pairwise/linear evidence as atom-space
constraints, compute sharp Frechet-style lower and upper bounds for a declared
Boolean or linear composition query, and expose endpoint witness distributions
for reproducible verification. The strongest paper is not about "AI safety
certification"; it is about dependence-aware composition under unknown joint
failure structure, plus artifact discipline that makes the mathematical claim
inspectable.

Top findings:

1. Implemented: `src/cc/kernel/sensitivity.py`, `src/cc/kernel/metrics.py`, and
   `src/cc/kernel/frechet_classes.py` form a real, tested scientific core.
   `make test-kernel` passed, including 118 kernel tests, a focused 60-test
   sensitivity/Frechet subset, strict mypy for `src/cc/kernel`, and ruff.
2. Implemented but fragile: paper artifact generation and verification exist
   (`scripts/reproduce_paper.py`, `scripts/verify_paper_artifacts.py`,
   `artifacts/paper/*`), and `make reproduce-paper` plus
   `make verify-paper-artifacts` passed. However, the committed
   `artifacts/paper/minimal_bounds.json` was initially inconsistent: regeneration
   changed `schema_version_tampered` to `schema_version`.
3. Partially implemented: evidence and audit primitives are real
   (`src/cc/evidence/merkle_log.py`, `src/cc/evidence/anchoring.py`,
   `src/cc/evidence/assurance_schema.py`), but they support integrity and
   claim review, not statistical validity.
4. Partially implemented / prototype: correlation cliffs, CCF point models,
   systemic stress tests, anytime-valid tests, causal two-world estimators, and
   red-team dependence search are plausible research directions, but they are
   not all equally mature and should not be bundled into the first paper as
   coequal core contributions.
5. Avoid / dangerous overclaim: older paper, audit, and experiment surfaces
   still contain `CC_max`, `delta_add`, "certification", "production-ready",
   "actionable audit", and "no prior art" style language. These conflict with
   the newer non-claim discipline in `README.md`,
   `docs/research/NON_CLAIMS.md`, and `docs/theory/metric_taxonomy.md`.

Brutal bottom line: the project is credible if it narrows. The finite atom LP,
metric taxonomy, witness artifacts, and verifier can support a strong first
paper. The project becomes fragile when it presents old two-world CC metrics,
dashboards, enterprise infrastructure, or exploratory correlation-cliff
experiments as if they were the same level of mathematical evidence.

## 2. Repository Reality Map

### Implemented: strict mathematical kernel

- `src/cc/kernel/sensitivity.py`: finite binary atom LP, `AssumptionSet`,
  `LinearConstraint`, `LinearQuery`, `identified_region`, infeasibility errors,
  stable assumption hashes, endpoint solutions, active constraints.
- `src/cc/kernel/metrics.py`: canonical estimand-layer diagnostics:
  `fh_width`, `fh_position`, `independent_event_probability`,
  `independence_regret`, `cc_gain`, `cc_shift`, with domain validation.
- `src/cc/kernel/frechet_classes.py`: classical Frechet special cases,
  pairwise joint constraints, rank/phi conversion for binary indicators,
  feasible distributions, sampling, monotonic tightening tests.
- Tests: `tests/unit/kernel/test_sensitivity.py`,
  `tests/unit/kernel/test_frechet_classes.py`,
  `tests/unit/kernel/test_classical_frechet_special_cases.py`,
  `tests/unit/kernel/test_monotonic_tightening.py`,
  `tests/unit/kernel/test_metric_*`.
- Docs: `docs/theory/metric_taxonomy.md`,
  `docs/theory/theorem_ledger.md`, `docs/research/PAPER_CORE.md`,
  `docs/research/NON_CLAIMS.md`.

### Implemented: paper artifact pipeline

- `scripts/reproduce_paper.py`: deterministic generation of minimal tables,
  figures, witness JSON, environment JSON, bundle JSON, and manifest hashes.
- `scripts/verify_paper_artifacts.py`: required-file checks, JSON schemas,
  manifest hashes and byte sizes, witness verification, recomputed LP optima,
  PNG signatures, table checks.
- `artifacts/paper/`: currently contains `table_1_classical_frechet_bounds.csv`,
  `table_2_metric_examples.csv`, `table_3_witness_verification.csv`,
  `minimal_bounds.json`, `minimal_witnesses.json`, `minimal_bundle.json`,
  `environment.json`, three PNG figures, and `manifest.json`.
- Tests: `tests/integration/test_reproduce_paper.py` and
  `tests/integration/test_verify_paper_artifacts.py`.

### Partially implemented: audit/evidence layer

- `src/cc/evidence/merkle_log.py`: RFC-6962-style Merkle roots, inclusion
  proofs, consistency proofs.
- `src/cc/evidence/anchoring.py`: root checkpoints and Ed25519 witness
  signatures.
- `src/cc/evidence/assurance_schema.py`: GSN-inspired assurance cases in which
  claims, assumptions, and defeaters default to human review.
- Tests: `tests/unit/evidence/test_assurance_schema.py`,
  `tests/unit/evidence/test_transparency_log_adversarial.py`.
- Status: useful support layer; not a deployment-safety certificate.

### Partially implemented: expanded kernel modules

- `src/cc/kernel/cliff.py`: tail-dependence estimates, bootstrap CIs, copula
  fitting, and `cliff_certificate`.
- `src/cc/kernel/stress.py`: budgeted dependence stress tests over finite atom
  laws using Wasserstein or regularized KL neighborhoods.
- `src/cc/kernel/ccf_models.py`: beta-factor, alpha-factor, MGL common-cause
  point models, FH consistency checks, and recommendation logic.
- `src/cc/kernel/sequential.py`: anytime-valid Bernoulli e-process for a
  pre-registered null miss rate.
- `src/cc/kernel/causal.py`: clustered two-world ATE utilities.
- These are mathematically interesting, but they broaden the namespace beyond
  the narrow first-paper core.

### Prototype / experimental surfaces

- `experiments/correlation_cliff/*`: extensive simulation stack, README,
  legacy and current paths, unit tests outside normal `tests/`.
- `experiments/fh_atlas/*`: Frechet atlas generation prototypes.
- `src/cc/redteam/dependence_search.py`: bounded active dependence search with
  safety gate and redacted artifacts.
- `scripts/generate_correlation_cliff_copula_simulation.py`,
  `scripts/generate_systemic_risk_case_study.py`,
  `scripts/calibrate_anytime_valid.py`.
- Status: promising research scaffolds; not all should be first-paper claims.

### Application/demo surfaces

- `src/cc/adapters/*`, `src/cc/guardrails/*`, `src/cc/enterprise/*`.
- `apps/dashboard/*`, `infra/*`, `README_ENTERPRISE.md`.
- `examples/minimal/run_bounds.py` is paper-facing and useful.
- Other examples are application/demo material.

### Legacy / stale / overgrown surfaces

- `src/cc/core/metrics.py`, `src/cc/cartographer/stats.py`,
  `src/cc/analysis/*`, and `src/cc/exp/run_two_world.py` preserve legacy
  `CC_max`, `cc_rel`, `delta_add`, and `delta_mult` pathways.
- `paper/main.tex` references missing section files (`02_setup.tex`,
  `03_theory.tex`, `04_evaluation.tex`, `05_discussion.tex`,
  `91_impossibility.tex`, `92_reproducibility.tex`) while the tracked
  `paper/sections/` directory contains only `01_introduction.tex`,
  `02_foundation.tex`, `02_framework.tex`, `03_identification.tex`, and
  `03_prereg.tex`.
- Top-level `theory/` is not importable as `cc.theory` from `PYTHONPATH=src`;
  its tests are not in the configured pytest `testpaths`.
- `docs/design-specs/strict-kernel-teardown-contracts-plan.md` contains stale
  claims such as "no mkdocs.yml" and locates the strongest FH implementation in
  experiments, despite the current `mkdocs.yml` and `src/cc/kernel`.

### Generated and local artifacts

- Ignored local bloat exists: `.venv`, `.venv-enterprise`,
  `apps/dashboard/node_modules`, `apps/dashboard/.next`, `site`, caches.
- Tracked or visible generated artifacts include `artifacts/paper`,
  `paper/figures`, `results/week5_scan`, `results/smoke`,
  `checkpoints/*`, `runs/*`.
- Full local pytest generated tracked diffs and a new checkpoint directory.
  This is an artifact hygiene problem.

## 3. Validation Snapshot

Initial state:

- `git status --short` initially returned clean.
- `find . -maxdepth 3 -type f | sort`, `find src -maxdepth 4 -type f | sort`,
  `find docs -maxdepth 4 -type f | sort`, and
  `find tests -maxdepth 4 -type f | sort` were run for reconnaissance.

Validation commands:

| Command | Result | Runtime | Important output |
| --- | --- | ---: | --- |
| `make test-kernel` | Pass | 7.38s | 118 kernel tests passed; focused sensitivity/Frechet subset passed; strict mypy passed for 9 kernel files; ruff passed. |
| `make test-release` | Pass | 14.38s | Re-ran kernel lane, ran `examples/minimal/run_bounds.py`, and passed 5 paper integration tests. |
| `make reproduce-paper` | Pass | 1.05s | Wrote paper artifacts to `artifacts/paper`. |
| `make verify-paper-artifacts` | Pass | 0.66s | Verified paper artifacts in `artifacts/paper`. |
| `.venv/bin/mkdocs build --strict --site-dir /tmp/cc-framework-mkdocs-site` | Pass | 0.83s | Built docs to `/tmp`; emitted only an upstream MkDocs Material warning. |
| `/usr/bin/time -p PYTHONPATH=src .venv/bin/pytest -q` | Fail before pytest | 0.00s | Shell invocation error: `time: PYTHONPATH=src: No such file or directory`. Non-core operator error. |
| `/usr/bin/time -p env PYTHONPATH=src .venv/bin/pytest -q` | Pass | 51.87s | Full local suite passed with 9 skips and 2 warnings. |
| `env PYTHONPATH=src .venv/bin/python -c "import importlib.util; print(importlib.util.find_spec('cc.theory'))"` | Pass | tiny | Printed `None`, confirming top-level `theory/` is not packaged as `cc.theory`. |

Full pytest skips:

- `tests/e2e/test_enterprise_smoke.py`: `moto` not installed.
- `tests/integration/test_enterprise_aws_emulation.py`: `moto` not installed.
- `tests/experiments/test_experiment_leak_metrics.py`: requires
  `CC_RUN_EXPERIMENTS=1`.
- `tests/performance/test_adapter_perf.py`: requires `CC_RUN_PERF=1`.
- `tests/unit/adapters/test_guardrails_ai_adapter.py`: optional guardrails
  dependency missing.
- `tests/unit/core/models/test_models_base.py`: optional `fastavro`,
  `protobuf`, and SQLAlchemy dependencies missing.

Observed validation side effects:

- `make reproduce-paper` changed
  `artifacts/paper/minimal_bounds.json` from
  `schema_version_tampered` to `schema_version`.
- Full pytest later modified:
  `paper/figures/cc_convergence.pdf`,
  `paper/figures/phase_diagram.pdf`,
  `paper/figures/roc_comparison.pdf`,
  `results/smoke/aggregates/summary.csv`,
  `results/week5_scan/analysis.json`,
  `results/week5_scan/calibration_summary.json`,
  and created `checkpoints/exp_1782534088/`.
- These are not code behavior changes, but they are repository state changes
  caused by validation. Tests that write tracked artifacts should be moved to
  temp directories or clearly gated.

## 4. Scientific Core

### Primary mathematical object

Implemented: the primary mathematical object is a finite atom distribution
over binary failure variables:

```text
Z = (Z_1, ..., Z_m), Z_i in {0,1}
pi(z) = P(Z = z), z in {0,1}^m.
```

Repository evidence:

- `README.md` defines `Z_i = 1` as guardrail failure / unsafe pass.
- `src/cc/kernel/sensitivity.py` enumerates atoms and solves LPs over the
  atom probability vector.
- `src/cc/kernel/frechet_classes.py` defines `atom_matrix`, classical
  Frechet bounds, side-constrained LPs, and distribution moments.

### Central estimand

Implemented: for a declared linear or Boolean composition query `phi`, the
central estimand is the sharp identified interval:

```text
L_phi = inf_{pi in F} E_pi[phi(Z)]
U_phi = sup_{pi in F} E_pi[phi(Z)]
```

where `F` is the feasible set implied by declared constraints.

Repository evidence:

- `AssumptionSet.identify()` in `src/cc/kernel/sensitivity.py`.
- `LinearQuery.intersection`, `LinearQuery.union`,
  `LinearQuery.from_predicate`.
- `docs/theory/theorem_ledger.md` theorem T1.

### Strongest theorem stack

Implemented / documented:

1. Finite LP sharpness: compact atom simplex plus linear objective.
2. Classical Frechet special cases for AND and OR.
3. Monotonic tightening under feasible refinement.
4. Infeasibility detection for contradictory constraints.
5. Endpoint witness verification.
6. Audit integrity is not statistical validity.

Repository evidence:

- `docs/theory/theorem_ledger.md`.
- `tests/unit/kernel/test_sensitivity.py`,
  `tests/unit/kernel/test_classical_frechet_special_cases.py`,
  `tests/unit/kernel/test_monotonic_tightening.py`.
- `tests/integration/test_verify_paper_artifacts.py`.

### Main algorithms

Implemented:

- Atom enumeration in little-endian order: `enumerate_atoms`,
  `atom_matrix`.
- LP identification via SciPy HiGHS: `identified_region`.
- Classical Frechet closed forms: `classical_frechet_bounds`.
- Pairwise side-information conversion: `dependence_to_joint_probability`,
  `joint_probability_to_dependence`.
- Product-coupling baseline evaluation:
  `independent_event_probability`.
- Witness export and verification:
  `scripts/reproduce_paper.py`, `scripts/verify_paper_artifacts.py`.

Partially implemented / prototype:

- Tail-dependence cliffs: `src/cc/kernel/cliff.py`.
- Budgeted stress tests: `src/cc/kernel/stress.py`.
- CCF point models: `src/cc/kernel/ccf_models.py`.
- Sequential e-processes: `src/cc/kernel/sequential.py`.

### Canonical metrics

Implemented canonical metrics:

- Identified-set diagnostics: `fh_width`, `fh_position`.
- Product-coupling diagnostics: `independent_event_probability`,
  `independence_regret`.
- One-world normalization: `cc_gain`.
- Two-world movement: `cc_shift`.

Documented deprecations:

- `cc_max`, `cc_rel`, `delta_add`, `delta_mult` are deprecated in
  `docs/theory/metric_taxonomy.md` and warn in `src/cc/core/metrics.py`.

### Evidence and witness story

Implemented core:

- `IdentificationResult.lower_solution` and `.upper_solution` expose endpoint
  distributions.
- `scripts/reproduce_paper.py` serializes witnesses with constraints, query
  coefficients, active constraints, and assumption hashes.
- `scripts/verify_paper_artifacts.py` recomputes LP optima and validates
  witness distributions.

Non-claim:

- Witnesses verify feasibility and endpoint optimality under declared
  constraints. They do not validate the data collection process, semantic
  coverage, causal validity, or future deployment behavior.

### Reproducibility story

Partially implemented:

- Make targets exist for kernel tests, release tests, paper reproduction, and
  artifact verification.
- Artifact manifest records filenames, byte sizes, SHA-256 hashes,
  package version, generation command, and schema version.
- Integration tests verify fresh temporary artifacts.

Fragile:

- The committed artifact state was inconsistent before regeneration.
- Full pytest writes tracked generated artifacts and new checkpoints.
- Some docs claim a final reproduce-paper pipeline is planned even though a
  pipeline now exists.

### Strongest first-paper contribution

Implemented and defensible:

> A finite partial-identification framework for composed AI guardrail failure
> events under unknown dependence, with sharp atom-LP bounds, product-coupling
> diagnostics, endpoint witness distributions, and a verifier that separates
> mathematical feasibility from empirical validity.

### Claims that must remain non-claims

Avoid / dangerous overclaim:

- Does not prove deployed systems are safe.
- Does not certify AI systems or deployed models.
- Does not infer causality without causal assumptions.
- Does not prove dataset representativeness.
- Does not guarantee future performance.
- Does not make cryptographic integrity equivalent to statistical truth.
- Does not make worst-case couplings automatically realizable attacks.
- Does not provide a full sequential agentic safety framework.

## 5. Overclaim and Fragility Audit

| Severity | Status | File path | Why it matters | Recommended action | Fix now/later |
| --- | --- | --- | --- | --- | --- |
| High | Implemented but fragile | `artifacts/paper/minimal_bounds.json` | Committed artifact initially used `schema_version_tampered`, and regeneration changed it. This means the checked-in paper artifact state was not self-consistent. | Add CI that verifies committed `artifacts/paper` without regenerating first, or stop tracking generated artifacts except release bundles. | Now |
| High | Partially implemented | `tests/*`, generated files under `paper/figures`, `results/*`, `checkpoints/*` | Full pytest modified tracked artifacts and created a new checkpoint. Tests should not mutate tracked outputs by default. | Redirect integration/regression outputs to `tmp_path`; gate artifact-refresh tests; ignore or untrack checkpoint outputs. | Now |
| High | Stale | `paper/main.tex` | LaTeX main references missing section files, so it is not a reliable current paper source. | Replace with a first-paper outline matching `docs/research/PAPER_CORE.md`, or archive current `paper/` as historical. | Now |
| High | Avoid / dangerous overclaim | `paper/sections/03_identification.tex`, `paper/sections/03_prereg.tex` | Uses "certifying complementarity", "certifying acceptable benign inflation", and "certificate of stability" language. This conflicts with repo non-claims. | Replace with "rejecting a pre-registered null under stated assumptions" or "finite-sample evidence for a threshold claim." | Now |
| High | Avoid / dangerous overclaim | `experiments/correlation_cliff/README.md` | Claims "Production-ready", "PhD-grade", "actionable insight", "no prior art", and deployed-system audit outputs. The experiment is useful but not validated enough for those claims. | Rewrite as exploratory/prototype; move strong language to hypotheses and planned validation. | Now |
| Medium | Partially implemented | `src/cc/kernel/__init__.py` | Public `cc.kernel` exports core LP/metrics plus causal, cliff, CCF, sequential, and stress modules, making experimental breadth look like one stable kernel. | Define a narrow `cc.kernel.strict` or docs-facing import surface for paper claims. | Now |
| Medium | Legacy | `src/cc/core/metrics.py`, `src/cc/cartographer/stats.py`, `src/cc/analysis/*`, `src/cc/exp/run_two_world.py` | `CC_max` and `delta_add` persist in many reports and experiments despite canonical replacement guidance. | Keep compatibility warnings, but prevent these names from appearing in first-paper artifacts except in a legacy appendix. | Now |
| Medium | Updated | `docs/design-specs/strict-kernel-teardown-contracts-plan.md` | Earlier text had stale "verified" claims about docs configuration, the Python floor, and the strongest implementation location. | Keep historical planning docs aligned with the active validation matrix when they mention current tooling. | Now |
| Medium | Stale / duplicate | `docs/research/THEOREM_LEDGER.md` and `docs/theory/theorem_ledger.md` | Two theorem ledgers present different theorem stacks. The theory ledger is better aligned with current kernel. | Consolidate into one canonical theorem ledger; archive the other. | Later |
| Medium | Legacy / orphaned | `theory/fh_bounds.py`, `theory/test_fh_bounds.py` | Top-level `theory/` is not importable as `cc.theory` and not in pytest `testpaths`. | Archive, migrate required pieces into `src/cc/kernel`, or add explicit sandbox labeling. | Later |
| Medium | Stale docs | `docs/index.md` | Says adding metrics means defining them in `cc.analysis.metrics`, but canonical metrics now live in `src/cc/kernel/metrics.py`. | Update docs front page to point to strict kernel and canonical metric taxonomy. | Now |
| Medium | Avoid / language risk | `docs/STACK_ROLE.md` | "Ghost Protocol ecosystem" and "law of physics" language is not paper-safe and can be read as theatrical. | Keep private if desired, but exclude from public research docs; rewrite public-facing copy. | Later |
| Medium | Addressed | full pytest skips | Enterprise, guardrails, avro/protobuf/sql tests skip when optional deps are missing. That is acceptable, but release claims should state which lane was run. | Use `docs/validation_matrix.md` to distinguish Paper Core, Full Python, Enterprise Reference, Dashboard, Docs, Security, and Optional Vendor lanes. | Now |
| Low | Stale | `README.md` | Witness/reproducibility section says final reproduce-paper pipeline is planned, while scripts and Makefile targets now exist. | Update wording from "planned" to "implemented but still maturing." | Now |
| Low | Generated workspace bloat | `.venv`, `.venv-enterprise`, `apps/dashboard/node_modules`, `apps/dashboard/.next`, `site` | Ignored, not tracked, but noisy for discovery and repo size. | Add cleanup note or `make clean-local` for generated local directories. | Later |

## 6. Research Direction Atlas

### 1. Finite-Atom Frechet Cartography

Name: Finite-Atom Frechet Cartography
Status: Implemented
Repository evidence: `src/cc/kernel/sensitivity.py`,
`src/cc/kernel/frechet_classes.py`, `docs/theory/theorem_ledger.md`,
`tests/unit/kernel/test_sensitivity.py`, `tests/unit/kernel/test_frechet_classes.py`,
`artifacts/paper/table_1_classical_frechet_bounds.csv`.
Mathematical foundation: finite Bernoulli coupling polytope; linear programming;
Frechet-Hoeffding bounds; partial identification.
Core research question: What composition-risk interval is identified by
singleton and optional dependence evidence for a declared guardrail failure
event?
Why this is nontrivial: Marginal guardrail rates generally underidentify joint
failure risk; pairwise evidence may tighten but not identify higher-order law.
Minimal implementation path: Stabilize the strict kernel API, add more canonical
examples for custom `LinearQuery` and pairwise intervals, and publish runtime
scaling for explicit atom enumeration.
Possible paper/benchmark/artifact: First paper core; reviewer-reproducible
minimal artifact bundle with Frechet tables, LP witnesses, and manifest hashes.
Required tests: property tests for sharpness, monotonic tightening, infeasible
constraints, pairwise interval consistency, solver tolerance stability.
Main risks: exponential atom scaling; users treating intervals as point
estimates; false precision from exact constraints estimated from finite data.
Non-claims: Does not prove safety, dataset representativeness, or future
performance.
Priority: Highest.

### 2. Witness Distributions and Proof-Carrying Bounds

Name: Witness Distributions and Proof-Carrying Bounds
Status: Partially implemented
Repository evidence: `IdentificationResult.lower_solution`,
`IdentificationResult.upper_solution`, `scripts/reproduce_paper.py`,
`scripts/verify_paper_artifacts.py`, `artifacts/paper/minimal_witnesses.json`,
`tests/integration/test_verify_paper_artifacts.py`.
Mathematical foundation: LP optimal solutions as endpoint witnesses; primal
feasibility plus objective attainment.
Core research question: Can every reported bound ship with enough witness data
for an independent verifier to reconstruct constraints and endpoints?
Why this is nontrivial: A witness artifact must bind atom order, labels,
constraints, assumptions hash, solver tolerances, objective coefficients, and
schema version without implying empirical truth.
Minimal implementation path: Harden schemas, add committed-artifact verification
CI, create golden fixtures, and make witness verification independent of the
artifact generator.
Possible paper/benchmark/artifact: "Proof-carrying bounds" artifact track: each
interval has endpoint distributions and a verifier.
Required tests: corrupted witness, hash mismatch, atom-order mismatch,
constraint mutation, tolerance boundary, schema evolution tests.
Main risks: Numerical tolerance disputes; stale committed artifacts; readers
mistaking witnesses for observed distributions.
Non-claims: Witnesses verify mathematical feasibility, not data quality or
deployment validity.
Priority: Highest.

### 3. Independence Regret Benchmark

Name: Independence Regret Benchmark
Status: Partially implemented
Repository evidence: `src/cc/kernel/metrics.py`,
`docs/theory/metric_taxonomy.md`, `artifacts/paper/figure_2_independence_regret.png`,
`tests/unit/kernel/test_metric_composition_diagnostics.py`.
Mathematical foundation: product Bernoulli coupling baseline; signed model
misspecification diagnostic.
Core research question: How wrong is the product-coupling assumption for a
declared event under feasible dependence alternatives?
Why this is nontrivial: Independence regret must be event-specific and
label-order-specific; it is not a global safety scalar.
Minimal implementation path: Generate synthetic benchmark cases where marginals
are fixed and dependence varies across the Frechet class; report endpoint
regret ranges and observed-regret examples.
Possible paper/benchmark/artifact: Benchmark table of product baseline failures
under known witness couplings.
Required tests: label-order invariance where intended, query dimension checks,
known closed-form two-event cases, regret endpoint tests.
Main risks: Reintroducing independence as a default truth model; confusing
regret with causal effect.
Non-claims: Independence regret does not identify the deployed joint law.
Priority: High.

### 4. Correlation-Cliff Formalization

Name: Correlation-Cliff Formalization
Status: Partially implemented
Repository evidence: `src/cc/kernel/cliff.py`,
`docs/theory/correlation_cliffs.md`,
`scripts/generate_correlation_cliff_copula_simulation.py`,
`docs/theory/figures/correlation_cliff_copula_sweep.csv`,
`experiments/correlation_cliff/*`, `tests/unit/kernel/test_cliff.py`.
Mathematical foundation: copula tail-dependence coefficients, finite-threshold
tail estimates, bootstrap CIs, dependence-sensitive event risk.
Core research question: When does small movement in supported tail dependence
produce large movement in guardrail co-failure risk?
Why this is nontrivial: Pearson correlation, finite rare-event overlap, and
asymptotic tail dependence are different objects; operational thresholds must
be pre-registered.
Minimal implementation path: Define one canonical cliff estimand, demote
experimental README claims, connect toy simulation to kernel verifier, and add
finite-sample CI calibration.
Possible paper/benchmark/artifact: Controlled copula simulation showing
Gaussian false cliffs versus Student-t/Clayton/Gumbel tail coupling.
Required tests: theoretical lambda values, bootstrap coverage sanity,
certificate threshold decisions, reproducible simulation outputs.
Main risks: Overclaiming real-world attackability; treating a copula family fit
as evidence of deployment behavior.
Non-claims: A cliff demonstration is not proof of realized harm or attack
feasibility.
Priority: High, but after paper-core artifact hardening.

### 5. Common-Cause Guardrail Failure Analysis

Name: Common-Cause Guardrail Failure Analysis
Status: Partially implemented
Repository evidence: `src/cc/kernel/ccf_models.py`,
`docs/theory/ccf_models.md`, `tests/unit/kernel/test_ccf_models.py`.
Mathematical foundation: reliability common-cause failure models, homogeneous
component groups, partition polynomials, FH admissibility checks.
Core research question: When do distinct guardrails share latent failure causes
that make layered defenses less redundant than singleton scores imply?
Why this is nontrivial: CCF point models add strong exchangeability and
common-cause assumptions; AI guardrails may not satisfy homogeneous component
assumptions.
Minimal implementation path: Keep CCF as sensitivity analysis beside FH
envelopes; define guardrail-specific common-cause evidence types; add examples
that show when CCF point estimates are inadmissible.
Possible paper/benchmark/artifact: Reliability-to-guardrail diagnostic appendix
or second paper.
Required tests: published CCF examples, FH envelope consistency, inadmissible
parameter rejection, recommendation logic.
Main risks: Users treating CCF point estimates as assumption-free guarantees.
Non-claims: The kernel does not infer latent causes or causal mechanisms.
Priority: Medium-high.

### 6. Adversarial Dependence Amplification

Name: Adversarial Dependence Amplification
Status: Prototype
Repository evidence: `src/cc/redteam/dependence_search.py`,
`docs/security/redteam_bounds.md`, `tests/unit/redteam/test_dependence_search.py`.
Mathematical foundation: constrained search over prompt perturbations; empirical
co-failure/tail-dependence metrics; comparison to held-out passive baseline.
Core research question: Can bounded adversarial input selection increase shared
guardrail failures while keeping the search space safe and auditable?
Why this is nontrivial: The attack objective must avoid unsafe content
generation, data leakage, and multiple-testing overclaim; discovered dependence
may be search-space-specific.
Minimal implementation path: Build a small synthetic benchmark with declared
transformations, held-out baselines, redacted artifacts, and pre-registered
metrics.
Possible paper/benchmark/artifact: Dependence-amplification benchmark with
safety gate and redacted proof artifacts.
Required tests: safety gate pre-objective blocking, reproducibility under fixed
seed, redaction-by-default, baseline split integrity, bootstrap CI behavior.
Main risks: Generating harmful content; overgeneralizing from bounded synthetic
perturbations; treating worst-case search as deployment prevalence.
Non-claims: Does not prove an attacker can realize any FH endpoint.
Priority: Medium-high.

### 7. Guardrail Portfolio Optimization Under Unknown Dependence

Name: Guardrail Portfolio Optimization Under Unknown Dependence
Status: Future research
Repository evidence: Existing kernel can evaluate a declared portfolio via
`AssumptionSet` and `LinearQuery`; no optimizer exists. `docs/research/RESEARCH_PROGRAM.md`
marks portfolio optimization as future research.
Mathematical foundation: robust optimization over guardrail subsets, costs,
constraints, and partially identified risk objectives.
Core research question: Which guardrail set minimizes worst-case or
ambiguity-aware composition risk under budget and utility constraints?
Why this is nontrivial: Decision variables change the event space, cost model,
dependence evidence, and computational scaling.
Minimal implementation path: Start with fixed small candidate set, known
singleton marginals, pairwise intervals, simple costs, and worst-case union or
intersection risk.
Possible paper/benchmark/artifact: Follow-up optimization benchmark.
Required tests: brute-force small portfolios, dominance pruning correctness,
infeasible subset handling, monotonic cost/risk behavior.
Main risks: Diluting first paper; making production recommendations without
validated evidence.
Non-claims: Current repo does not recommend deployments.
Priority: Medium.

### 8. Sequential Dependence Bounds for Agentic Systems

Name: Sequential Dependence Bounds for Agentic Systems
Status: Future research with partial sequential utility
Repository evidence: `src/cc/kernel/sequential.py`,
`docs/theory/anytime_valid.md`, `tests/unit/kernel/test_sequential.py`;
`docs/research/RESEARCH_PROGRAM.md` marks sequential dependence bounds as
future research.
Mathematical foundation: stochastic processes, filtrations, stopping times,
e-processes, trajectory-level partial identification.
Core research question: How should dependence-aware composition bounds extend
from one batch of binary events to time-indexed agent trajectories?
Why this is nontrivial: Path dependence, adaptive policies, tool calls, and
stopping rules change the estimand.
Minimal implementation path: Keep current e-process as monitoring utility;
separately define trajectory events and finite-history atom approximations for
small horizons.
Possible paper/benchmark/artifact: Later theory paper on trajectory-level
partial identification.
Required tests: stopping-time Type-I calibration, predictable-bet constraints,
small-horizon exact enumeration, independence from monitored data in `p0`.
Main risks: Overclaiming "agent safety" from static binary bounds.
Non-claims: Current kernel is not a full sequential agent safety framework.
Priority: Medium.

### 9. Semantic Fault-Line / Subgroup Dependence Analysis

Name: Semantic Fault-Line / Subgroup Dependence Analysis
Status: Future research
Repository evidence: `docs/research/RESEARCH_PROGRAM.md` identifies semantic
fault-line analysis; `docs/theory/two_world_estimand.md` discusses evaluation
population and clusters; no slice-discovery implementation exists.
Mathematical foundation: conditional identified sets, subgroup/slice analysis,
multiplicity control, external validity.
Core research question: Do aggregate composition intervals hide concentrated
dependence risk in semantic slices?
Why this is nontrivial: Slice discovery can overfit; semantic labels require
validation; multiple testing must be controlled.
Minimal implementation path: Start with pre-declared slices only; compute
per-slice atom bounds and report hierarchical uncertainty.
Possible paper/benchmark/artifact: Subgroup benchmark after core artifacts are
stable.
Required tests: fixed-slice bound correctness, sample-size warnings,
multiplicity-adjusted intervals, no post-hoc claim promotion.
Main risks: Subgroup safety overclaims; privacy/governance issues in slices.
Non-claims: Aggregate intervals do not prove subgroup safety.
Priority: Medium-low until empirical data exists.

### 10. Continuous-Score / Copula-Free Guardrail Score Bounds

Name: Continuous-Score / Copula-Free Guardrail Score Bounds
Status: Speculative
Repository evidence: Current core reduces to binary events; score-oriented
legacy code exists in `src/cc/core/metrics.py`, `src/cc/cartographer/bounds.py`,
and top-level `theory/fh_bounds.py`, but not as a clean score-bound kernel.
Mathematical foundation: distribution-free bounds on thresholded scores,
ROC envelopes, optimal transport or rearrangement inequalities.
Core research question: Can composition bounds be computed without committing
to a binary threshold too early?
Why this is nontrivial: Score-level dependence is much richer than Bernoulli
atoms and may require empirical process or optimal transport machinery.
Minimal implementation path: Define threshold-indexed binary bounds first;
avoid general continuous claims until a tractable theorem is written.
Possible paper/benchmark/artifact: Later methods paper on score-to-event
uncertainty propagation.
Required tests: threshold grid consistency, reduction to binary case,
monotonicity over nested thresholds, finite-sample envelope calibration.
Main risks: Scope explosion; weak formalism; accidental copula assumptions.
Non-claims: Current repository is not a continuous-score composition theory.
Priority: Low for first paper.

### 11. Finite-Sample Constraint Uncertainty

Name: Finite-Sample Constraint Uncertainty
Status: Partially implemented
Repository evidence: `src/cc/cartographer/intervals.py`,
`src/cc/cartographer/planner.py`, `src/cc/kernel/sequential.py`,
`docs/theory/anytime_valid.md`, `docs/theory/two_world_estimand.md`,
`tests/unit/cartographer/*`, `tests/unit/kernel/test_sequential.py`.
Mathematical foundation: confidence sequences, Wilson/Bernstein intervals,
cluster bootstrap, interval-valued LP constraints.
Core research question: How should uncertainty in estimated marginals and
pairwise constraints propagate through the identified composition interval?
Why this is nontrivial: The LP currently treats supplied constraints as exact
or declared intervals; statistical coverage of those intervals must compose
with optimization and multiple constraints.
Minimal implementation path: Add a paper-core example using interval marginal
constraints from binomial confidence intervals, then verify coverage in
simulation.
Possible paper/benchmark/artifact: Finite-sample uncertainty extension to the
first paper or immediate follow-up.
Required tests: coverage simulations, union-bound accounting, interval
constraint infeasibility, monotone widening with smaller samples.
Main risks: Misstating finite-sample guarantees; mixing heuristic bootstrap
outputs into theorem claims.
Non-claims: Exact LP endpoints under estimated constraints are conditional on
the declared intervals, not automatically valid confidence intervals.
Priority: High after v0.3 hardening.

### 12. Optimal Transport View of Extremal Guardrail Couplings

Name: Optimal Transport View of Extremal Guardrail Couplings
Status: Future research with prototype hooks
Repository evidence: `src/cc/kernel/stress.py` uses Wasserstein distance on
binary atoms; `docs/theory/systemic_risk.md` documents budgeted stress tests.
Mathematical foundation: optimal transport on finite atom spaces, coupling
polytopes, distance-constrained ambiguity sets.
Core research question: Can Frechet endpoints and local stress neighborhoods be
unified as coupling optimization problems?
Why this is nontrivial: OT gives geometry, but the project must avoid adding
formalism that does not improve computation or interpretation.
Minimal implementation path: Formalize finite-atom Hamming-cost Wasserstein
stress as an LP and show FH upper endpoint as infinite-budget limit.
Possible paper/benchmark/artifact: Theory appendix or stress-testing follow-up.
Required tests: primal LP feasibility, distance monotonicity, convergence to FH
limit, known two-rail closed forms.
Main risks: Mathematical decoration without empirical payoff; confusing local
stress with assumption-free bounds.
Non-claims: Budgeted stress outputs are modeling choices, not universal worst
cases unless budget is infinite.
Priority: Medium.

### 13. Claim-Bounded Safety Receipts

Name: Claim-Bounded Safety Receipts
Status: Partially implemented
Repository evidence: `src/cc/evidence/assurance_schema.py`,
`src/cc/evidence/merkle_log.py`, `src/cc/evidence/anchoring.py`,
`docs/security/transparency_log.md`, `tests/unit/evidence/*`,
`src/cc/core/evidence_bundle.py`.
Mathematical foundation: cryptographic integrity, schema validation,
assurance-case decomposition, explicit non-claims.
Core research question: How can mathematical outputs, assumptions, hashes, and
human review status be bound into artifacts without implying safety
certification?
Why this is nontrivial: Auditability is often rhetorically inflated into truth;
this repo must keep byte integrity separate from statistical validity.
Minimal implementation path: Define `cc.paper.bundle.v2` with explicit
claim-status fields, non-claim fields, witness verification status, and
review-required flags.
Possible paper/benchmark/artifact: Artifact appendix and reproducibility
contribution.
Required tests: schema validation, replay/tamper tests, missing-defeater tests,
human-review default tests, manifest mismatch.
Main risks: "Receipt" language sounding like certification; polished reports
hiding missing evidence.
Non-claims: Receipts do not prove safety, conformance, or empirical validity.
Priority: High as artifact discipline, medium as standalone research.

### 14. Mechanistic Localization of Common-Cause Failures

Name: Mechanistic Localization of Common-Cause Failures
Status: Speculative
Repository evidence: CCF diagnostics exist in `src/cc/kernel/ccf_models.py`;
bounded red-team search exists in `src/cc/redteam/dependence_search.py`; no
mechanistic localization implementation exists.
Mathematical foundation: causal discovery, representation analysis, error
taxonomy, latent-variable modeling, controlled interventions.
Core research question: Can observed co-failure dependence be traced to shared
features, policy ambiguities, model blind spots, or retrieval artifacts?
Why this is nontrivial: Dependence is not causation; mechanistic claims need
interventions or strong measurement design.
Minimal implementation path: Start with labeled synthetic common-cause factors
and show whether dependence bounds change when conditioning on those factors.
Possible paper/benchmark/artifact: Future empirical paper after common-cause
benchmarks exist.
Required tests: synthetic known-cause recovery, negative controls,
conditioning/infeasibility checks, held-out validation.
Main risks: Causal overclaim; data dredging; confusing labels with mechanisms.
Non-claims: Current repository does not localize mechanisms.
Priority: Low-medium.

### 15. Decision Theory for Abstain/Escalate Policies Under Dependence Uncertainty

Name: Decision Theory for Abstain/Escalate Policies Under Dependence Uncertainty
Status: Future research
Repository evidence: Guardrail and protocol layers exist
(`src/cc/guardrails/*`, `src/cc/core/protocol.py`), but no decision-theoretic
policy optimizer is implemented.
Mathematical foundation: robust Bayes/minimax decision theory, partial
identification, abstention costs, escalation costs, utility under ambiguity.
Core research question: Given a wide identified interval, when should a system
abstain, escalate, gather more evidence, or accept residual risk?
Why this is nontrivial: Requires explicit utilities, operational constraints,
human review costs, and population assumptions.
Minimal implementation path: Define a tiny decision table over `[L,U]`, utility
losses, and value-of-information for one declared event.
Possible paper/benchmark/artifact: Applied assurance follow-up.
Required tests: dominance cases, interval monotonicity, threshold sensitivity,
no hidden utility defaults.
Main risks: Making implicit policy decisions appear mathematical.
Non-claims: The current kernel computes bounds; it does not decide deployment.
Priority: Medium-low until governance layer stabilizes.

## 7. First-Paper Strategy

Suggested title:

Sharp Composition Bounds for AI Guardrails Under Unknown Dependence

One-paragraph thesis:

Composed guardrail evaluation is a partial-identification problem when only
singleton failure rates and limited dependence evidence are known. This paper
formalizes finite binary guardrail failures as an atom-space joint law, computes
sharp Frechet composition intervals for declared Boolean events by linear
programming, compares those intervals to explicit product-coupling baselines,
and ships endpoint witness distributions plus verifier artifacts so reported
bounds are reproducible and challengeable. The contribution is conditional on
supplied evidence and assumptions; it is not a deployment safety certificate.

Five exact contributions:

1. Formalization of composed guardrail failure evaluation as finite Bernoulli
   partial identification over atom distributions.
2. Sharp atom-LP algorithm for lower/upper event-risk bounds under singleton,
   pairwise, monotonicity, and arbitrary linear constraints.
3. Metric taxonomy separating identified-set diagnostics (`fh_width`,
   `fh_position`) from product-coupling diagnostics
   (`independent_event_probability`, `independence_regret`) and legacy CC names.
4. Endpoint witness distributions and assumption hashes for proof-carrying
   reproducibility of LP bounds.
5. Claim-bounded artifact discipline that verifies mathematical feasibility and
   integrity without claiming empirical validity or deployment safety.

Non-contributions:

- Not a deployment safety certificate.
- Not causal inference without causal assumptions.
- Not dataset representativeness.
- Not a full sequential agent safety framework.
- Not a guardrail portfolio optimizer.
- Not a continuous-score copula theory.
- Not an enterprise/cloud/dashboard product paper.

Section outline:

1. Introduction: dependence problem in composed guardrail evaluation.
2. Problem setup: binary failure variables, atom law, constraints, event query.
3. Sharp finite-atom bounds: LP, classical Frechet recovery, infeasibility.
4. Diagnostics: FH width/position, product coupling, independence regret.
5. Witnesses and reproducibility: endpoint distributions, hashes, verifier.
6. Experiments/artifacts: minimal cases, dependence sensitivity, runtime
   scaling, optional toy correlation-cliff illustration only if demoted.
7. Limitations and non-claims.
8. Related work.

Required experiments:

- Classical AND/OR Frechet recovery for 2 and 3 events.
- Pairwise side-information tightening and infeasibility examples.
- Product-coupling regret examples with fixed marginals and varying witnesses.
- Runtime scaling for explicit atom LP as `m` grows.
- Artifact-verifier corruption tests in paper supplement.

Required figures/tables:

- Table: classical Frechet formulas versus atom LP outputs.
- Table: metric taxonomy and deprecated names.
- Table: witness verification results.
- Figure: FH interval and observed/product baseline position.
- Figure: independence regret across controlled examples.
- Figure: atom-count/runtime scaling.
- Optional figure: toy correlation-cliff sensitivity, labeled as illustrative.

Must be implemented before submission:

- Committed-artifact verifier lane that fails on stale/tampered artifacts.
- Paper source aligned with `docs/research/PAPER_CORE.md`.
- Deterministic artifact generation that does not mutate unrelated tracked files.
- Runtime scaling script and stable output table.
- Final non-claim checklist applied to README, paper, and artifact docs.

Must be deleted, demoted, or excluded:

- Current stale `paper/main.tex` or missing section references.
- `paper/sections/*` certification language.
- `experiments/correlation_cliff/README.md` production/actionable/no-prior-art
  language.
- Enterprise/dashboard/AWS content from first-paper core.
- Legacy `CC_max` and `delta_add` as front-door metrics.
- Top-level `theory/` sandbox unless migrated or archived.

Best-fit venues:

- Research software / artifact-friendly venues and workshops in ML evaluation,
  AI safety evaluation, security measurement, reliability, or reproducibility.
- For a stronger security venue, the project needs empirical adversarial
  dependence benchmarks and stronger external validation beyond the current
  kernel.
- For a statistics venue, it needs deeper finite-sample uncertainty propagation
  and clearer relation to partial identification/optimal transport literature.

## 8. Implementation Roadmap

### Phase 0 -- Stabilize Current Release Candidate

Deliverables:

- Clean artifact state.
- Fresh-venv validation notes.
- Docs build in `/tmp` plus link sanity.
- Validation matrix covering Paper Core, Full Python, Enterprise Reference,
  Dashboard, Docs, Security, and Optional Vendor lanes.
- Release checklist.

Acceptance criteria:

- `git status --short` is clean after `make test-release`,
  `make reproduce-paper`, and `make verify-paper-artifacts`, unless explicitly
  running an artifact-refresh command.
- `verify-paper-artifacts` passes on committed `artifacts/paper` before
  regeneration.
- Full pytest either writes only to temp paths or clearly gates generated
  artifact refresh.
- README validation commands match Makefile targets.

Tests:

- `make test-kernel`
- `make test-release`
- `make reproduce-paper`
- `make verify-paper-artifacts`
- `env PYTHONPATH=src .venv/bin/pytest -q`
- `.venv/bin/mkdocs build --strict --site-dir /tmp/cc-framework-mkdocs-site`

Anti-goals:

- No new research features.
- No dashboard/cloud expansion.
- No metric renaming beyond documentation/deprecation cleanup.

### Phase 1 -- Paper Artifact Maturity

Deliverables:

- Hardened artifact schemas.
- Witness verifier v2.
- Golden paper bundle fixture.
- Deterministic figures/tables.
- Minimal examples with witness checks.

Acceptance criteria:

- Every artifact has schema, hash, generation command, and verifier check.
- Corruption tests cover witness values, constraints, queries, hashes, and
  manifest payload.
- Artifact generation is reproducible from a clean checkout.

Tests:

- Existing integration tests plus committed-bundle verification.
- New tests for query mutation and atom-order mismatch.
- Snapshot/golden manifest test.

Anti-goals:

- No real-world deployment claims.
- No broad empirical benchmarks yet.
- No certification wording.

### Phase 2 -- Empirical Demonstration

Deliverables:

- Synthetic dependence benchmark.
- Realistic guardrail-style dataset with documented provenance.
- Adversarial dependence examples under bounded transformations.
- Slice/subgroup analysis only with controls.

Acceptance criteria:

- Populations, labels, guardrails, and composition events are declared before
  analysis.
- Synthetic examples isolate dependence from marginal-rate changes.
- Adversarial search reports held-out baselines, safety gates, redaction, and
  multiple-testing limitations.
- Subgroup results include multiplicity controls or are explicitly exploratory.

Tests:

- Synthetic known-law recovery.
- Held-out baseline integrity.
- Redaction/safety-gate tests.
- Coverage and calibration simulations.

Anti-goals:

- No benchmark-to-deployment generalization.
- No "real-world audited system" claim without external protocol.
- No product recommendation.

### Phase 3 -- Theory Expansion

Deliverables:

- Finite-sample uncertainty propagation through interval constraints.
- Pairwise-to-higher-order limitation examples.
- Continuous-score or threshold-grid formulation.
- Optimal transport finite-atom stress formalism.

Acceptance criteria:

- Each extension has a formal estimand, theorem statement, proof sketch, tests,
  and non-claims.
- Extensions reduce to current finite binary kernel in special cases.
- Coverage claims are simulation-backed and assumption-labeled.

Tests:

- Interval-valued LP coverage.
- OT stress monotonicity and FH-limit tests.
- Continuous-threshold reduction tests.

Anti-goals:

- No broad agentic safety claim.
- No continuous-score claims without theorem and tests.
- No new terminology without formal object.

### Phase 4 -- Applied Assurance Layer

Deliverables:

- Claim-bounded receipts.
- Audit bundles.
- Governance-facing report templates.
- External validation workflow.
- Optional cloud integration only after core evidence is stable.

Acceptance criteria:

- Receipt schemas distinguish mathematical outputs, empirical assumptions,
  integrity checks, and human review status.
- Governance reports enumerate unresolved defeaters and non-claims.
- Cloud/dashboard tools consume verified bundles rather than duplicating math.

Tests:

- Schema validation.
- Tamper/replay tests.
- Human-review-default tests.
- Dashboard bundle verification tests, if included.

Anti-goals:

- No automated acceptance of safety claims.
- No audit receipt as "truth".
- No cloud architecture as central research contribution.

## 9. Next 5 Codex Prompts

### Prompt 1: v0.3-rc1 Release Candidate Hardening

Context:
You are in the `cc-framework` repository. The current research core is the
finite atom LP kernel in `src/cc/kernel`, with paper artifacts generated by
`scripts/reproduce_paper.py` and verified by `scripts/verify_paper_artifacts.py`.
The previous discovery pass found that validation mutates tracked generated
files and that committed `artifacts/paper/minimal_bounds.json` had drifted.

Goal:
Stabilize the v0.3-rc1 validation state so release checks can run from a clean
checkout without unintended tracked diffs.

Scope limits:
Do not add new research features. Do not change mathematical behavior unless a
test proves a bug. Keep edits to validation, artifact hygiene, docs, and
release checklist surfaces.

Required files:
`Makefile`, `scripts/reproduce_paper.py`, `scripts/verify_paper_artifacts.py`,
`tests/integration/test_reproduce_paper.py`,
`tests/integration/test_verify_paper_artifacts.py`, relevant regression tests
that write `paper/figures`, `results`, or `checkpoints`, `README.md`,
`docs/research/ROADMAP.md`.

Tests:
`make test-kernel`; `make test-release`; `make reproduce-paper`;
`make verify-paper-artifacts`; `env PYTHONPATH=src .venv/bin/pytest -q`;
docs build to `/tmp`.

Acceptance criteria:
After the validation commands, `git status --short` shows only intentional
documentation/report edits. Tests that need generated outputs use `tmp_path` or
an explicit artifact-refresh target. The committed paper bundle verifies before
regeneration.

Final response format:
List files changed, commands run, artifact diffs eliminated, remaining skips,
and any residual release risk.

### Prompt 2: Artifact Schema and Witness Verifier Hardening

Context:
`scripts/reproduce_paper.py` emits paper artifacts under `artifacts/paper`.
`scripts/verify_paper_artifacts.py` validates schemas, hashes, metrics, and LP
witnesses. Existing tests catch corrupted witnesses and manifest hash mismatch.

Goal:
Harden the artifact schema and witness verifier so every bound is
proof-carrying relative to atom order, labels, constraints, query coefficients,
solver tolerance, and assumption hash.

Scope limits:
Do not change the mathematical estimand. Do not add dashboards or enterprise
features. Do not weaken verifier failures into warnings.

Required files:
`scripts/reproduce_paper.py`, `scripts/verify_paper_artifacts.py`,
`tests/integration/test_verify_paper_artifacts.py`,
`tests/integration/test_reproduce_paper.py`, `artifacts/paper/*`,
`docs/research/PAPER_CORE.md`.

Tests:
Existing paper integration tests plus new tests for atom-order mismatch,
query-coefficient mutation, constraint RHS mutation, missing assumptions hash,
and tolerance boundary behavior.

Acceptance criteria:
Verifier fails on corrupted atom order, labels, assumptions, query coefficients,
witness distributions, stale hashes, and schema drift. Fresh artifacts still
verify. Committed artifacts verify without regeneration.

Final response format:
Summarize schema changes, verifier checks added, corruption cases covered,
commands run, and any backward-compatibility notes.

### Prompt 3: Correlation Cliff Formalization

Context:
`src/cc/kernel/cliff.py` and `docs/theory/correlation_cliffs.md` implement and
document tail-dependence cliff concepts, while
`experiments/correlation_cliff/README.md` contains overstrong production and
actionability language.

Goal:
Turn correlation cliffs into a precise, bounded, paper-safe research component:
formal definition, non-claims, toy simulation, and tests.

Scope limits:
Do not claim real-world deployment risk. Do not use "production-ready",
"certify", "no prior art", or actionable deployment recommendations. Keep the
first paper core optional: this can be an illustrative section, not the main
theorem stack.

Required files:
`src/cc/kernel/cliff.py`, `docs/theory/correlation_cliffs.md`,
`experiments/correlation_cliff/README.md`,
`scripts/generate_correlation_cliff_copula_simulation.py`,
`tests/unit/kernel/test_cliff.py`.

Tests:
Unit tests for theoretical tail-dependence values, certificate CI threshold
logic, bootstrap finite-sample sanity, and reproducible simulation output.

Acceptance criteria:
Docs define a single cliff estimand and pre-registered threshold semantics.
Experiment README is demoted to exploratory/prototype. Simulation artifacts are
reproducible and labeled as toy evidence.

Final response format:
List formal definitions changed, language removed/demoted, tests run, and
remaining empirical limitations.

### Prompt 4: Independence Regret Benchmark

Context:
`src/cc/kernel/metrics.py` implements product-coupling event probability and
independence regret. `artifacts/paper/figure_2_independence_regret.png` is a
minimal example, but there is not yet a benchmark that maps regret across
controlled dependence cases.

Goal:
Create a small, deterministic independence regret benchmark for fixed
marginals and varying feasible couplings, producing a table and figure suitable
for the first paper.

Scope limits:
Do not use legacy `delta_add` as the benchmark metric. Do not imply the product
coupling is the true deployment model. Keep data synthetic and fully declared.

Required files:
`src/cc/kernel/metrics.py`, `src/cc/kernel/sensitivity.py`,
`scripts/reproduce_paper.py`, `scripts/verify_paper_artifacts.py`,
`tests/integration/test_reproduce_paper.py`,
`docs/theory/metric_taxonomy.md`, `artifacts/paper/*`.

Tests:
Known two-event cases; label-order checks; product baseline within Frechet
interval; signed regret endpoint tests; artifact verifier checks for new table
and figure.

Acceptance criteria:
Benchmark artifacts regenerate deterministically, verify by schema/hash, and
show product-coupling error without presenting independence as truth.

Final response format:
Summarize benchmark design, artifact files, tests run, and non-claims.

### Prompt 5: Finite-Sample Constraint Uncertainty

Context:
The strict kernel accepts declared exact or interval constraints, but the paper
core still needs a disciplined story for estimated marginals and pairwise
constraints. Supporting code exists in `src/cc/cartographer/intervals.py`,
`src/cc/cartographer/planner.py`, `src/cc/kernel/sequential.py`, and
`docs/theory/anytime_valid.md`.

Goal:
Prototype finite-sample uncertainty propagation by converting estimated
Bernoulli marginals into interval constraints and solving the atom LP over
those intervals, with explicit coverage assumptions.

Scope limits:
Do not claim finite-sample validity without a stated confidence construction.
Do not merge heuristic bootstrap intervals into theorem language. Keep the
prototype small and synthetic.

Required files:
`src/cc/kernel/sensitivity.py`, `src/cc/cartographer/intervals.py`,
`src/cc/cartographer/planner.py`, `docs/theory/anytime_valid.md`,
`docs/research/PAPER_CORE.md`, relevant `tests/unit/kernel` and
`tests/unit/cartographer` files.

Tests:
Interval constraints widen as sample size decreases; coverage simulation for
simple two-event marginals; infeasible interval combinations fail clearly;
exact-constraint behavior is preserved when intervals collapse.

Acceptance criteria:
A minimal example shows estimated marginals -> confidence intervals ->
`AssumptionSet` interval constraints -> composition interval. Tests document
coverage assumptions and non-claims.

Final response format:
List implementation files, mathematical assumptions, tests run, and limitations.

## 10. Final Brutal Verdict

The repository can credibly become a focused research software artifact and
first paper on dependence-aware partial identification of composed guardrail
failures. The core is real. The kernel tests pass. The witness idea is strong.
The artifact verifier is a serious differentiator if hardened.

The project should not try to become, in the first paper, a dashboard, an AWS
reference architecture, a red-team product, a full causal framework, a
continuous-score theory, a sequential agent safety framework, and a governance
receipt system all at once. Those surfaces are useful only if they orbit the
strict kernel rather than compete with it.

Highest-value next move: make the release/artifact state clean and non-mutating,
then rewrite the paper around the finite atom LP, canonical metrics, witnesses,
and non-claims. Everything else should be labeled prototype or future research
until it earns its own theorem, benchmark, and verifier.
