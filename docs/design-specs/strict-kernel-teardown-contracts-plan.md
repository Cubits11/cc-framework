# CC-Framework Strict Kernel Teardown + Contracts Plan (Historical)

> Historical status: this is an archived planning snapshot from an earlier
> strict-kernel extraction pass. It is not the current contributor guide.
> Current readers should start with `README.md`, [docs/index.md](../index.md),
> [docs/research/PAPER_CORE.md](../research/PAPER_CORE.md), and
> [docs/architecture/STRICT_KERNEL_CONTRACT.md](../architecture/STRICT_KERNEL_CONTRACT.md).
>
> Current repo reality: Python `>=3.10`, active project scripts in
> `pyproject.toml`, a root `mkdocs.yml`, and publication-facing kernel code in
> `src/cc/kernel/`. Treat the older `src/cc/core` extraction sketch below as
> historical design context unless a current contract document says otherwise.
>
> Scope note: this document was a **planning + contract-definition pass only**.
> It did not implement kernel refactors.

## 1) Repo Reality Map (Verified)

### 1.1 High-level structure
Top-level areas in the current repository:
- Library/runtime code: `src/cc/`
- Tests: `tests/`
- Experiment pipelines: `experiments/`
- Documentation and design specs: `docs/`
- Paper artifacts: `paper/`
- Notebooks: `notebooks/`
- Operational scripts: `scripts/`
- Build/CI/deploy: `Makefile`, `.github/workflows/`, `deployment/`

### 1.2 Packaging identity
- Project name: `cc-framework`
- Version: `0.2.0`
- Python: `>=3.10`
- Core dependencies are lightweight (`numpy`, `pydantic`, `pyyaml`, `jsonlines`, `cryptography`, `blake3`), with optional extras for heavier stacks.
- Active project scripts are declared in `pyproject.toml`: `cc-bundle`,
  `cc-cartographer`, `cc-dependence-bench`, and `cc-report`.
- `src` layout with package discovery `include = ["cc*"]`.

### 1.3 Source layout + public API surface
- Current publication-facing kernel modules live under `src/cc/kernel/`.
- Public package surface is broad and package-level via `src/cc/__init__.py`.
- `cc.__all__` exports package namespaces (`adapters`, `analysis`,
  `cartographer`, `core`, `exp`, `guardrails`, `io`, `utils`).
- The historical proposal below used a `src/cc/core` API-boundary sketch, but
  current kernel work should follow
  [docs/architecture/STRICT_KERNEL_CONTRACT.md](../architecture/STRICT_KERNEL_CONTRACT.md)
  and the `src/cc/kernel/` modules.

### 1.4 Tests + verification stack
- Pytest is configured in `pyproject.toml` and used across `tests/`.
- Hypothesis is present in unit/property tests (e.g., core model/hash/time parsing paths), but there is no dedicated strict-kernel invariants suite yet.
- CI runs lint and pytest across Python 3.10, 3.11, 3.12, and 3.13, with the
  focused mypy target on Python 3.12.

### 1.5 Docs system and docs build reality
- `Makefile` docs target uses `mkdocs build --strict`.
- CI docs workflow runs on changes to `docs/**`, `README.md`, `mkdocs.yml`, etc.
- Root `mkdocs.yml` is present, and the docs workflow runs `make docs`.

### 1.6 Experiments + determinism
- `experiments/run.py` computes config hash, dataset hash, git SHA, and writes a manifest.
- `src/cc/exp/run_two_world.py` applies deterministic controls (`PYTHONHASHSEED`, RNG seeds, BLAS thread caps).
- `src/cc/io/storage.py` uses deterministic JSON hashing and content-addressed sharded storage paths.

### 1.7 Existing audit and manifest capabilities
- Hash-chained JSONL audit logs are implemented in `src/cc/cartographer/audit.py` (`append_jsonl`, `verify_chain`, `tail_sha`, rehash/truncate utilities).
- Additional run-manifest helpers exist (e.g., `experiments/fh_atlas/manifest.py`).
- There is **no single formalized audit_packet v1 directory schema** (`manifest.json` + `results.json` + `report.md` + `figures/`) enforced as a contract.

### 1.8 Existing assumptions/theory labeling
- Assumptions and failure modes are documented narratively in docs and memos.
- No first-class machine-readable assumption registry with stable IDs + schema validation currently exists under `src/cc/core`.

### Explicit requested answers
- **Current de-facto public API:** package namespace exports in `cc.__init__`,
  active `project.scripts`, and importable kernel modules under
  `src/cc/kernel/`.
- **Core math primitives location:** current paper-facing primitives are in
  `src/cc/kernel/`, especially `sensitivity.py`, `metrics.py`,
  `frechet_classes.py`, and `sample_complexity.py`.
- **Dependence/FH/envelope implementation:** current contributor-facing work
  should use `src/cc/kernel/` and the strict-kernel contract, with older
  cartographer and experiment modules treated as clients or historical
  prototypes.
- **Where CC is computed:** publication-facing CC and identified-set diagnostics
  belong in `cc.kernel.metrics`; legacy/supporting workflows may still exist in
  `cc.core`, `cc.cartographer`, or experiments.

---

## 2) Vision Mismatch (Strict Kernel vs Current)

### A) Blockers (must change)
1. No signed, narrow `cc-core` API boundary with CI snapshot enforcement.
2. Math semantics are duplicated/split across core/cartographer/experiments.
3. No machine-readable assumption registry attached to computations.
4. No canonical audit packet v1 schema + validator + deterministic golden fixture.
5. No contract-change gate requiring changelog/migration notes on API/schema drift.
6. Class-conditional default behavior is not enforced at kernel API boundary.

### B) Non-blocking divergences (defer possible)
1. Existing CLIs and experiment orchestrators can remain as clients during extraction.
2. Existing hash-chained audit JSONL can be adapted as input to audit packet generator.
3. Existing docs/memos can remain and be cross-linked rather than replaced.

### C) Already aligned (retain)
1. Determinism intent exists in runtime controls.
2. Dependency minimization direction already exists at core install level.
3. Rich dependence math exists and can be promoted into kernel modules.
4. CI foundations (lint/type/test/docs hooks) already exist.

---

## 3) Strict Kernel Definition (Historical Contract Sketch)

The current source of truth for strict-kernel semantics is
[docs/architecture/STRICT_KERNEL_CONTRACT.md](../architecture/STRICT_KERNEL_CONTRACT.md).
The artifact list below is preserved to explain the older planning pass.

### 3.1 Historical contract artifact set (not current target file list)

#### API Contract
- `docs/contracts/api.md`
- `src/cc/core/api_surface.py`
- `src/cc/core/schemas.py`

#### Invariant Contract
- `docs/contracts/invariants.md`
- `tests/property/test_core_invariants.py`
- `tests/unit/core/test_kernel_edge_cases.py`

#### Assumption Contract
- `docs/contracts/assumptions.md`
- `src/cc/core/assumptions_schema.py`
- `src/cc/core/assumptions_registry.yaml`

#### Repro Contract
- `docs/contracts/reproducibility.md`
- `src/cc/core/repro.py`

#### Audit Contract
- `docs/contracts/audit_packet_v1.md`
- `src/cc/core/audit_packet.py`
- `src/cc/core/audit_packet_schema.py`

### 3.2 Signed-off operational definition
A contract is "signed-off" only if:
1. The contract exists in versioned doc + code files.
2. CI fails on unauthorized drift.
3. Contract changes require changelog entry (and migration note for breaking changes).

### 3.3 Required CI contract checks
- `tests/contracts/test_api_snapshot.py`
- `tests/contracts/test_schema_validation.py`
- `tests/contracts/test_contract_change_requires_changelog.py`

---

## 4) Step-by-step Execution Plan (Historical Steps 1-6)

### Step 1 — Freeze kernel boundary (historical cc-core contract sketch)
**Current note:** the repo now uses `src/cc/kernel/` as the paper-facing kernel
home. The file list below is retained as historical planning context, not as
current implementation guidance.

**Existing assets now:** finite-atom and metric primitives in `src/cc/kernel/`,
with legacy/supporting primitives in `src/cc/core/metrics.py` and
`src/cc/cartographer/bounds.py`.

**Historical create/change proposal:**
- Add `src/cc/core/api_surface.py` (stable re-exports only).
- Add `src/cc/core/schemas.py` (typed contracts).
- Add `docs/contracts/api.md` (one-page signed API).
- Add deprecation shims in legacy import paths.

**Done criteria:** API snapshot test green, deprecated imports warn, no duplicated math entrypoints in signed surface.

### Step 2 — Operators as types (OR / AND / Sequential)
**Historical create/change proposal:**
- Add `src/cc/core/operators.py` dataclass-based operator objects.
- Modes: `observed_joint`, `fh_worst`, `fh_best`, `copula(param)`.
- Inputs accepted as either paired outcomes or marginals + dependence model.

**Done criteria:** class-conditional outputs mandatory; invariants executed before result emission.

### Step 3 — Machine-readable assumption registry
**Historical create/change proposal:**
- Add registry YAML + schema + loader/validator.
- Attach `assumption_ids` to kernel outputs and audit exports.

**Done criteria:** CI validates registry schema and every verdict-producing output includes IDs.

### Step 4 — Audit Packet v1 format
**Historical create/change proposal:**
- Canonical deterministic packet bundle: `audit_packet/{manifest.json,results.json,report.md,figures/}`.
- Pure deterministic packet builder + schema validation.

**Done criteria:** golden packet fixture reproducible; CI checks deterministic regen + schema conformance.

### Step 5 — Upgrade GCE into thin client of cc-core
**Historical create/change proposal:**
- Define dependency boundary: tag-pinned kernel release consumption.
- Define required API entrypoints for GCE UX.
- Enforce "no duplicated math in client" policy.

**Done criteria:** integration tests assert client invokes cc-core API, not copied formulas.

### Step 6 — Build cc-academy (separate repo)
**Historical create/change proposal:**
- Define tutorial/docs/notebook API surface from cc-core.
- Add headless notebook execution CI recipe (papermill/nbclient).
- Provide minimal reproducible notebook set.

**Done criteria:** notebooks run headless in CI and consume pinned cc-core version.

---

## 5) API Spec v0 (historical sketch)

Current readers should not add a new `src/cc/core/api_surface.py` from this
sketch. Use the current `src/cc/kernel/` modules and
[docs/architecture/STRICT_KERNEL_CONTRACT.md](../architecture/STRICT_KERNEL_CONTRACT.md)
for kernel semantics.

```python
# Historical proposal: src/cc/core/api_surface.py
from .schemas import (
    OperatorRequest,
    OperatorResult,
    CCBoundsRequest,
    CCBoundsResult,
    AssumptionValidationResult,
)

def evaluate_operator(req: OperatorRequest) -> OperatorResult: ...
def compute_cc_bounds(req: CCBoundsRequest) -> CCBoundsResult: ...
def validate_assumptions(ids: list[str]) -> AssumptionValidationResult: ...
def build_audit_packet(result: CCBoundsResult, out_dir: str) -> str: ...
```

### Numeric-only machine meta rule
- `results.json` and machine payload structs allow only finite numeric/bool scalar leaves.
- Free-form narrative text is restricted to `report.md`.

### Error policy
- Strict kernel defaults to typed exceptions (`InputValidationError`, `FeasibilityError`, `InvariantViolationError`).
- Non-strict wrappers may exist outside signed kernel.

### Stability promise
- SemVer-stable: `cc.core.api_surface` exported names/signatures + schemas + audit packet v1 schema.
- Experimental: analysis notebooks and exploratory experiment modules.

---

## 6) Invariants & Property Tests (Executable)

### Required invariants
1. FH feasibility (`p11 ∈ [L,U]`) always holds.
2. OR/AND extrema occur at FH endpoints.
3. CC/JC bounds are ordered and nonnegative.
4. CC degeneracy policy handles `J_best=0` without unhandled NaN/Inf.
5. Class-conditional separation is mandatory.
6. Numeric-only rule enforced for machine payloads.
7. Assumption IDs must be valid registry IDs.
8. Deterministic mode yields repeatable outputs.
9. Audit packet schema always validates.
10. Operator mode is explicit and round-trippable.

### Proposed property tests (10)
1. `test_p11_within_fh_bounds`
2. `test_and_bounds_match_endpoints`
3. `test_or_bounds_match_endpoints`
4. `test_jc_bounds_ordered`
5. `test_cc_degenerate_policy`
6. `test_class_conditional_required`
7. `test_assumption_registry_roundtrip`
8. `test_numeric_meta_enforcement`
9. `test_operator_mode_roundtrip`
10. `test_audit_packet_numeric_results`

### Proposed edge-case tests (10)
1. `pA=0,pB=0`
2. `pA=1,pB=1`
3. `pA=1,pB=0`
4. epsilon-near boundaries
5. infeasible `p11<L`
6. infeasible `p11>U`
7. empty class partition
8. invalid copula parameter
9. `J_best=0` branch
10. deterministic packet regeneration byte-equality

---

## 7) Assumption Registry Spec + Initial Entries

### Proposed schema
```yaml
- id: ASM-001
  name: binary_outcomes
  scope: [operator_eval, cc_bounds]
  falsifiability:
    check: all_outcomes_binary
    severity: error
  implications:
    if_violated: abort_strict_run
```

### Initial 12 assumptions
1. `ASM-001` Binary outcomes.
2. `ASM-002` Valid class partition (Y=1/Y=0).
3. `ASM-003` Sample representativeness to deployment slice.
4. `ASM-004` Intra-window stationarity.
5. `ASM-005` Threshold realizability.
6. `ASM-006` Independence used only when explicitly selected.
7. `ASM-007` FH feasibility of provided marginals/joints.
8. `ASM-008` Copula family adequacy when copula mode selected.
9. `ASM-009` Missingness mechanism recorded.
10. `ASM-010` Label noise bounded/monitored.
11. `ASM-011` Deterministic seed policy honored.
12. `ASM-012` Calibration window validity.

---

## 8) Audit Packet v1 Spec

### `manifest.json` (required fields)
- `packet_version`
- `git_commit`
- `python_version`
- `os`
- `deps_hash`
- `random_seed`
- `dataset_fingerprint`
- `run_id`
- `created_at_utc`
- `assumption_ids`
- `input_hash`

### `results.json`
- Machine-readable finite numeric outcomes only.
- Includes CC/JC point + bounds + mode/operator codes + diagnostics.

### `report.md`
1. Run context
2. Assumptions engaged
3. Operator/mode
4. Primary outcomes
5. Violations/warnings
6. Repro instructions

### `figures/` naming convention
- `01_roc_envelope.png`
- `02_cc_path.png`
- `03_sensitivity.png`

---

## 9) Refactor Plan (historical file-move sketch)

Current repo direction: paper-facing kernel code lives under `src/cc/kernel/`.
The older proposed layout below is retained only to explain the planning
history.

### Historical proposed kernel layout under `src/cc/core/`
- `api_surface.py`
- `schemas.py`
- `operators.py`
- `dependence.py`
- `cc_metrics.py`
- `assumptions_schema.py`
- `assumptions_registry.yaml`
- `audit_packet.py`
- `audit_packet_schema.py`
- `repro.py`
- `errors.py`

### Deprecation strategy
- Keep wrappers in existing modules for one minor release.
- Emit clear `DeprecationWarning` migration targets.

### Duplication elimination
- Move validated dependence math from legacy/supporting modules to current
  kernel modules.
- Make `experiments/` and `cartographer/` consume kernel API only.

---

## 10) CI Gates & Release Discipline

### Required gates
1. Unit + property tests.
2. Lint/format/type checks.
3. Deterministic golden artifact checks.
4. API snapshot drift + changelog guard.
5. Assumption + audit schema validation checks.
6. Docs build check (recommended keep).

### SemVer policy
- **Major:** signed API/schema breaking change.
- **Minor:** additive API/schema expansion.
- **Patch:** bugfix/perf/docs with no signed contract break.

### Audit schema evolution
- Additive optional fields => minor.
- Required field mutation/removal => major + migration note.

---

## 11) Open Questions (only unanswerable from repo)

1. Should canonical CC in strict kernel be ratio-only or include canonical log-ratio variant?
2. Should strict API always `raise`, or permit policy-selectable `clip` in signed surface?
3. Should machine payload enums be strict integer codes only, or bounded strings allowed?
4. For external consumers (e.g., GCE), should version pinning be exact tag or compatible range?

---

## Market/Positioning Number Credibility Check (for sales narrative)

This repository contains strong technical claims, but several external-facing market numbers in pitch narrative are **not currently source-linked in repo artifacts**. Specifically:
- Funding totals/deal counts for AI guardrails market.
- Percentage of enterprises lacking AI security frameworks.
- 2025 breach-rate and breach-cost deltas.
- 2026 enterprise guarded-agent adoption percentages.
- 2025 M&A totals and acquirer-specific rollups.

**Recommendation:** before go-to-market deck publication, create `docs/go-to-market/claims_registry.md` with per-claim fields: claim text, source URL, source date, geography scope, methodology note, and confidence tier.

---

## Evidence Index (files inspected)
- `pyproject.toml`
- `src/cc/__init__.py`
- `src/cc/core/metrics.py`
- `src/cc/cartographer/bounds.py`
- `src/cc/cartographer/audit.py`
- `src/cc/exp/run_two_world.py`
- `src/cc/io/storage.py`
- `experiments/run.py`
- Legacy correlation-cliff experiment implementation
- `experiments/fh_atlas/manifest.py`
- `README.md`
- `docs/reproducibility.md`
- `.github/workflows/ci.yml`
- `.github/workflows/docs.yml`
- `Makefile`
