# Research Roadmap

This roadmap stages the project as a research program. It keeps the current
kernel narrow, makes paper artifacts reproducible, and leaves applied assurance
work until the mathematical and empirical boundaries are explicit.

## Phase 1 - Kernel Purification

Objective: make the mathematical core small, inspectable, and publication-ready.

Deliverables:

- formal estimand layer,
- theorem ledger,
- metric taxonomy,
- witness artifacts,
- reproduce-paper pipeline.

Success criteria:

- The finite atom LP, metrics, and witness outputs have stable documented contracts.
- The theorem ledger links every paper-facing mathematical claim to implementation and tests.
- Kernel tests, strict type checks, and focused lint checks pass.
- Paper examples can be regenerated from deterministic commands.

Explicit things not to do yet:

- Do not expand dashboards or user-facing apps.
- Do not add new guardrail adapters.
- Do not introduce enterprise demos or cloud workflows.
- Do not change mathematical behavior without a documented bug and test.

## Phase 2 - Paper-Grade Reproducibility

Objective: turn the core method into deterministic paper artifacts that
reviewers can rerun and challenge.

Deliverables:

- deterministic artifact generation,
- manifest hashing,
- witness verification,
- minimal examples,
- paper tables/figures.

Success criteria:

- Every paper table and figure has a stable generation command.
- Outputs record versions, seeds, assumptions, and hashes.
- Witness tables reconstruct constraints and endpoint objectives.
- Minimal examples run without notebook state or manual edits.

Explicit things not to do yet:

- Do not add product workflows around the artifacts.
- Do not claim paper artifacts certify a deployed system.
- Do not use unverified exploratory plots as final paper figures.
- Do not broaden the first paper to portfolio or agentic settings.

## Phase 3 - Empirical Guardrail Evaluation

Objective: test the framework on controlled and real guardrail evaluations
while preserving the implemented-versus-future boundary.

Deliverables:

- benchmark datasets,
- synthetic dependence experiments,
- adversarial dependence amplification,
- semantic subgroup/fault-line analysis.

Success criteria:

- Empirical protocols define populations, labels, guardrails, and composition events.
- Synthetic experiments isolate dependence effects from marginal-rate changes.
- Adversarial and semantic analyses include uncertainty and multiplicity controls.
- Results state whether they are examples, diagnostics, or validated claims.

Explicit things not to do yet:

- Do not treat benchmark performance as deployment certification.
- Do not present subgroup scans without statistical controls.
- Do not imply adversarial dependence search has solved all threat models.
- Do not merge empirical claims into the kernel contract.

## Phase 4 - Applied Assurance Layer

Objective: package claim-bounded evidence for governance-facing review without
conflating auditability with safety.

Deliverables:

- claim-bounded receipts,
- audit bundles,
- optional cloud integration,
- governance-facing reports.

Success criteria:

- Receipt schemas distinguish mathematical outputs, empirical assumptions, and human review status.
- Audit bundles can be verified for integrity without implying statistical validity.
- Governance reports enumerate non-claims and unresolved defeaters.
- Optional cloud integration remains an implementation detail, not the research claim.

Explicit things not to do yet:

- Do not make receipts sound like proof of safety.
- Do not turn optional cloud plumbing into the center of the project.
- Do not automate acceptance of safety claims that require human judgment.
- Do not obscure missing evidence behind polished report formatting.
